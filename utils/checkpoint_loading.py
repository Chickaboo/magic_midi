from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
from safetensors import safe_open as safetensors_safe_open
from safetensors.torch import load_file as safetensors_load_file

from config import ModelConfig
from data.tokenizer import CustomDeltaTokenizer
from model.factory import build_model
from model.variant_a import VariantAConfig, VariantAModel
from model.variant_b import VariantBConfig, VariantBModel
from model.variant_c import VariantCConfig, VariantCModel
from model.variant_d import VariantDConfig, VariantDModel
from model.variant_e import VariantEConfig, VariantEModel
from model.variant_f import VariantFConfig, VariantFModel
from utils.config_compat import normalize_model_config_payload


@dataclass(frozen=True)
class LoadedModelBundle:
    """Container for a strictly validated checkpoint/model load."""

    model: Any
    model_class: str
    model_config: Dict[str, Any]
    checkpoint_path: Path
    sidecar_path: Path | None
    checkpoint_metadata: Dict[str, Any]
    missing_keys: int
    unexpected_keys: int


def coerce_mapping(value: Any) -> Dict[str, Any]:
    """Coerce dict-like payloads or JSON strings into dictionaries."""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except Exception:
            return {}
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def resolve_sidecar_path(model_path: Path) -> Path | None:
    """Resolve companion .pt sidecar for a checkpoint file."""

    if model_path.suffix == ".pt":
        return model_path if model_path.exists() else None
    if model_path.suffix != ".safetensors":
        return None

    candidates = []
    stem = model_path.stem
    candidates.append(model_path.with_name(f"{stem}_state.pt"))

    if stem.endswith("_model"):
        candidates.append(model_path.with_name(f"{stem[:-6]}_state.pt"))

    if model_path.name.endswith("_model.safetensors"):
        candidates.append(
            model_path.with_name(
                model_path.name.replace("_model.safetensors", "_state.pt")
            )
        )

    candidates.append(model_path.with_name("latest_state.pt"))
    candidates.append(model_path.with_name("best_state.pt"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_safetensors_metadata(model_path: Path) -> Dict[str, Any]:
    """Load metadata dictionary from safetensors file when available."""

    if model_path.suffix != ".safetensors" or not model_path.exists():
        return {}
    try:
        with safetensors_safe_open(str(model_path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        if isinstance(metadata, Mapping):
            return dict(metadata)
    except Exception:
        return {}
    return {}


def _metadata_from_sidecar_payload(sidecar_path: Path | None) -> Dict[str, Any]:
    """Extract metadata-like fields from a sidecar state payload."""

    if (
        sidecar_path is None
        or not sidecar_path.exists()
        or sidecar_path.suffix != ".pt"
    ):
        return {}

    try:
        payload = torch.load(sidecar_path, map_location="cpu")
    except Exception:
        return {}
    if not isinstance(payload, Mapping):
        return {}

    metadata: Dict[str, Any] = {}
    for key in ("epoch", "val_loss", "train_config", "data_config", "model_config"):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list, str, int, float, bool)):
            metadata[key] = value
            continue
        try:
            metadata[key] = dict(value.__dict__)
        except Exception:
            continue
    return metadata


def load_checkpoint_metadata(
    model_path: Path,
    sidecar_path: Path | None = None,
) -> Dict[str, Any]:
    """Load checkpoint metadata with safetensors-first, sidecar fallback."""

    metadata = load_safetensors_metadata(model_path)
    if metadata:
        return metadata

    if sidecar_path is None:
        sidecar_path = resolve_sidecar_path(model_path)
    return _metadata_from_sidecar_payload(sidecar_path)


def extract_data_config(checkpoint_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data_config payload from checkpoint metadata."""

    return coerce_mapping(checkpoint_metadata.get("data_config"))


def extract_model_config(checkpoint_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model_config payload from checkpoint metadata."""

    return coerce_mapping(checkpoint_metadata.get("model_config"))


def strip_dataparallel_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Strip module. prefixes produced by DataParallel checkpoints."""

    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module.") :]] = value
        else:
            stripped[key] = value
    return stripped


def load_state_dict(model_path: Path) -> Dict[str, torch.Tensor]:
    """Load checkpoint weights from safetensors or torch .pt files."""

    if model_path.suffix == ".safetensors":
        raw = safetensors_load_file(str(model_path), device="cpu")
        return strip_dataparallel_prefix(raw)

    payload = torch.load(model_path, map_location="cpu")
    if isinstance(payload, Mapping):
        state_dict = payload.get("state_dict")
        if isinstance(state_dict, Mapping):
            return strip_dataparallel_prefix(dict(state_dict))

        tensor_dict = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
        if tensor_dict:
            return strip_dataparallel_prefix(tensor_dict)

    raise RuntimeError(f"Unsupported checkpoint format: {model_path}")


def infer_model_architecture(model_config_payload: Dict[str, Any]) -> str:
    """Infer model family from checkpoint model_config keys."""

    payload = dict(model_config_payload)
    keys = set(payload.keys())

    has_gdn = bool(
        {
            "gdn_inner_dim",
            "gdn_num_heads",
            "gqa_num_heads",
            "gqa_groups",
        }
        & keys
    )
    has_cfc = bool({"cfc_units", "cfc_backbone_units", "cfc_backbone_layers"} & keys)
    has_variant_b_attn = "num_attention_heads" in keys
    has_variant_c_ffn = "ffn_expansion" in keys
    has_hybrid_markers = bool(
        {
            "d_state",
            "d_conv",
            "expand",
            "use_mamba",
            "use_cfc",
            "use_continuous_time_encoding",
        }
        & keys
    )
    has_variant_f_markers = bool(
        {
            "event_size",
            "harmonic_ratio",
            "temporal_ratio",
            "structural_num_heads",
            "cross_stream_every_n_layers",
        }
        & keys
    )

    if has_variant_f_markers:
        return "variant_f"

    if has_gdn and has_cfc:
        return "variant_a"
    if has_gdn and not has_cfc:
        return "variant_e"
    if (
        has_variant_b_attn
        and has_variant_c_ffn
        and not has_cfc
        and not has_gdn
        and not has_hybrid_markers
    ):
        return "variant_c"
    if has_variant_b_attn and has_cfc and not has_gdn and not has_hybrid_markers:
        return "variant_b"
    if has_cfc and not has_variant_b_attn and not has_gdn and not has_hybrid_markers:
        return "variant_d"
    return "hybrid"


def _filter_payload(payload: Dict[str, Any], allowed_keys: set[str]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if key in allowed_keys}


def _build_model_from_payload(
    model_config_payload: Dict[str, Any],
) -> Tuple[Any, Any, str]:
    """Instantiate model and config object from raw checkpoint payload."""

    arch = infer_model_architecture(model_config_payload)
    payload = dict(model_config_payload)

    if arch == "variant_a":
        allowed = set(VariantAConfig.__annotations__.keys())
        cfg = VariantAConfig(**_filter_payload(payload, allowed))
        return VariantAModel(cfg), cfg, arch

    if arch == "variant_b":
        allowed = set(VariantBConfig.__annotations__.keys())
        cfg = VariantBConfig(**_filter_payload(payload, allowed))
        return VariantBModel(cfg), cfg, arch

    if arch == "variant_c":
        allowed = set(VariantCConfig.__annotations__.keys())
        cfg = VariantCConfig(**_filter_payload(payload, allowed))
        return VariantCModel(cfg), cfg, arch

    if arch == "variant_d":
        allowed = set(VariantDConfig.__annotations__.keys())
        cfg = VariantDConfig(**_filter_payload(payload, allowed))
        return VariantDModel(cfg), cfg, arch

    if arch == "variant_e":
        allowed = set(VariantEConfig.__annotations__.keys())
        cfg = VariantEConfig(**_filter_payload(payload, allowed))
        return VariantEModel(cfg), cfg, arch

    if arch == "variant_f":
        allowed = set(VariantFConfig.__annotations__.keys())
        cfg = VariantFConfig(**_filter_payload(payload, allowed))
        return VariantFModel(cfg), cfg, arch

    normalized = normalize_model_config_payload(dict(payload))
    allowed = set(ModelConfig.__annotations__.keys())
    cfg = ModelConfig(**_filter_payload(normalized, allowed))
    return build_model(cfg), cfg, arch


def _validate_config_against_state(
    model_config: Any,
    state_dict: Dict[str, torch.Tensor],
    model_path: Path,
) -> None:
    """Validate core dimensions before loading state to prevent silent mismatches."""

    token_weight = state_dict.get("token_embedding.weight")
    if token_weight is None:
        raise RuntimeError(
            f"Checkpoint {model_path.name} is missing token_embedding.weight; "
            "cannot validate model dimensions."
        )

    expected_vocab = int(getattr(model_config, "vocab_size", -1))
    expected_d_model = int(getattr(model_config, "d_model", -1))
    inferred_vocab = int(token_weight.shape[0])
    inferred_d_model = int(token_weight.shape[1])
    if expected_vocab != inferred_vocab or expected_d_model != inferred_d_model:
        raise RuntimeError(
            f"Checkpoint {model_path.name} config mismatch: metadata says "
            f"vocab_size={expected_vocab}, d_model={expected_d_model}, but "
            f"weights are vocab_size={inferred_vocab}, d_model={inferred_d_model}."
        )


def load_model_from_checkpoint(
    model_path: Path,
    *,
    sidecar_path: Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> LoadedModelBundle:
    """Load checkpoint model with architecture auto-detection and strict validation."""

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    resolved_sidecar = sidecar_path or resolve_sidecar_path(model_path)
    checkpoint_metadata = load_checkpoint_metadata(
        model_path=model_path,
        sidecar_path=resolved_sidecar,
    )

    model_config_payload = extract_model_config(checkpoint_metadata)
    if not model_config_payload and resolved_sidecar is not None and resolved_sidecar.exists():
        try:
            sidecar_payload = torch.load(resolved_sidecar, map_location="cpu")
        except Exception:
            sidecar_payload = {}
        if isinstance(sidecar_payload, Mapping):
            model_config_payload = coerce_mapping(sidecar_payload.get("model_config"))

    if not model_config_payload:
        raise RuntimeError(
            f"Checkpoint {model_path.name} is missing model_config metadata and sidecar model config. "
            "Re-save the checkpoint with training metadata or provide the matching *_state.pt sidecar."
        )

    state_dict = load_state_dict(model_path)
    model, model_cfg, _arch = _build_model_from_payload(model_config_payload)
    _validate_config_against_state(model_cfg, state_dict, model_path)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if strict and (missing_keys or unexpected_keys):
        missing_preview = ", ".join(missing_keys[:8]) if missing_keys else "none"
        unexpected_preview = ", ".join(unexpected_keys[:8]) if unexpected_keys else "none"
        raise RuntimeError(
            "Checkpoint/model mismatch detected. "
            f"missing={len(missing_keys)} ({missing_preview}) | "
            f"unexpected={len(unexpected_keys)} ({unexpected_preview}). "
            "Use a matching checkpoint bundle (weights + state + tokenizer)."
        )

    device_obj = torch.device(device)
    model.eval()
    model.to(device_obj)

    return LoadedModelBundle(
        model=model,
        model_class=type(model).__name__,
        model_config=dict(getattr(model_cfg, "__dict__", {})),
        checkpoint_path=model_path,
        sidecar_path=resolved_sidecar,
        checkpoint_metadata=checkpoint_metadata,
        missing_keys=int(len(missing_keys)),
        unexpected_keys=int(len(unexpected_keys)),
    )


def detect_tokenizer_kind(tokenizer_path: Path, data_config: Dict[str, Any]) -> str:
    """Validate and detect unified tokenizer kind (custom_delta only)."""

    strategy = str(data_config.get("tokenization_strategy", "")).strip().lower()
    if strategy and strategy != "custom_delta":
        raise ValueError(
            "Unsupported checkpoint tokenization strategy "
            f"'{strategy}'. Only custom_delta is supported."
        )

    if tokenizer_path.suffix.lower() == ".json":
        try:
            payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        payload_type = str(payload.get("type", "")).strip()
        if payload_type and payload_type != "CustomDeltaTokenizer":
            raise ValueError(
                "Unsupported tokenizer payload type "
                f"'{payload_type}'. Only CustomDeltaTokenizer is supported."
            )

    return "custom_delta"


def resolve_tokenizer_path(
    data_config: Dict[str, Any],
    search_paths: Iterable[Path],
) -> Path:
    """Resolve tokenizer path from data_config metadata and fallback search paths."""

    configured = str(data_config.get("tokenizer_path", "")).strip()

    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
        candidates.append(Path(configured))

    fallback_paths = [Path(path) for path in search_paths]
    custom_first = [
        path for path in fallback_paths if path.name.lower() == "custom_tokenizer.json"
    ]
    others = [
        path for path in fallback_paths if path.name.lower() != "custom_tokenizer.json"
    ]
    fallback_paths = custom_first + others

    candidates.extend(fallback_paths)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate

    if candidates:
        raise FileNotFoundError(
            "Tokenizer not found. Searched: "
            + ", ".join(str(path) for path in candidates)
        )
    raise FileNotFoundError("Tokenizer not found and no candidate paths were provided.")


def load_tokenizer(
    tokenizer_path: Path,
    data_config: Dict[str, Any],
) -> tuple[Any, Dict[str, Any]]:
    """Load tokenizer object with automatic class selection."""

    tokenizer_kind = detect_tokenizer_kind(tokenizer_path, data_config)
    tokenizer = CustomDeltaTokenizer.load(str(tokenizer_path))

    meta = {
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_kind": tokenizer_kind,
        "data_config": dict(data_config),
    }
    return tokenizer, meta


def load_tokenizer_for_checkpoint(
    checkpoint_metadata: Dict[str, Any],
    *,
    search_paths: Iterable[Path],
) -> tuple[Any, Dict[str, Any]]:
    """Resolve and load tokenizer from checkpoint metadata + fallback paths."""

    data_config = extract_data_config(checkpoint_metadata)
    tokenizer_path = resolve_tokenizer_path(
        data_config=data_config,
        search_paths=search_paths,
    )
    return load_tokenizer(tokenizer_path, data_config)


__all__ = [
    "LoadedModelBundle",
    "coerce_mapping",
    "detect_tokenizer_kind",
    "extract_data_config",
    "extract_model_config",
    "infer_model_architecture",
    "load_checkpoint_metadata",
    "load_model_from_checkpoint",
    "load_state_dict",
    "load_tokenizer",
    "load_tokenizer_for_checkpoint",
    "load_safetensors_metadata",
    "resolve_sidecar_path",
    "resolve_tokenizer_path",
    "strip_dataparallel_prefix",
]
