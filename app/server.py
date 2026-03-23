from __future__ import annotations

import io
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file
from safetensors.torch import load_file as safetensors_load_file
from safetensors import safe_open as safetensors_safe_open


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_DRIVE_DIR = REPO_ROOT / "local_drive" / "piano_model"
LOCAL_DRIVE_MODELS_DIR = LOCAL_DRIVE_DIR / "checkpoints"
LOCAL_DRIVE_TOKENIZER_PATH = LOCAL_DRIVE_DIR / "tokenizer" / "tokenizer.json"

from config import ModelConfig
from data.tokenizer import PianoTokenizer
from model.factory import build_model
from utils.config_compat import normalize_model_config_payload


app = Flask(__name__, template_folder=str(APP_DIR / "templates"))

MODELS_DIR = APP_DIR / "models"
TOKENIZER_PATH = APP_DIR / "tokenizer" / "tokenizer.json"
RUNTIME_DIR = APP_DIR / "runtime"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
MODEL_SEARCH_DIRS = (MODELS_DIR, LOCAL_DRIVE_MODELS_DIR)
TOKENIZER_SEARCH_PATHS = (TOKENIZER_PATH, LOCAL_DRIVE_TOKENIZER_PATH)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_model_cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
_tokenizer_cache: PianoTokenizer | None = None


def _strip_dataparallel_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Strip `module.` prefixes produced by DataParallel checkpoints."""

    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module.") :]] = value
        else:
            stripped[key] = value
    return stripped


def _resolve_sidecar(model_path: Path) -> Path | None:
    """Resolve checkpoint sidecar state path for a model file."""

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

    for sidecar in candidates:
        if sidecar.exists():
            return sidecar
    return None


def _load_safetensors_metadata(model_path: Path) -> Dict[str, Any]:
    """Load safetensors metadata dictionary when available."""

    if model_path.suffix != ".safetensors" or not model_path.exists():
        return {}
    try:
        with safetensors_safe_open(str(model_path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        if isinstance(metadata, dict):
            return dict(metadata)
    except Exception:
        return {}
    return {}


def _metadata_from_sidecar_payload(sidecar_path: Path | None) -> Dict[str, Any]:
    """Extract metadata-like fields from a `.pt` sidecar payload."""

    if (
        sidecar_path is None
        or not sidecar_path.exists()
        or sidecar_path.suffix != ".pt"
    ):
        return {}
    try:
        state = torch.load(sidecar_path, map_location="cpu")
    except Exception:
        return {}
    if not isinstance(state, dict):
        return {}

    metadata: Dict[str, Any] = {}
    for key in ("epoch", "val_loss", "train_config", "data_config", "model_config"):
        value = state.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list, str, int, float, bool)):
            metadata[key] = value
        else:
            try:
                metadata[key] = value.__dict__
            except Exception:
                continue
    return metadata


def _resolve_checkpoint_metadata(
    model_path: Path, sidecar_path: Path | None
) -> Dict[str, Any]:
    """Load checkpoint metadata from safetensors or sidecar fallback."""

    metadata = _load_safetensors_metadata(model_path)
    if metadata:
        return metadata

    sidecar_metadata = _metadata_from_sidecar_payload(sidecar_path)
    if sidecar_metadata:
        return sidecar_metadata

    return {}


def _parse_json_metadata_value(
    metadata: Dict[str, Any], key: str
) -> Dict[str, Any] | None:
    """Parse JSON-encoded metadata payload for a specific key."""

    raw = metadata.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _load_model_config_from_sidecar(sidecar_path: Path | None) -> ModelConfig | None:
    """Load model configuration from sidecar state if present."""

    if sidecar_path is None or sidecar_path.suffix != ".pt":
        return None
    try:
        state = torch.load(sidecar_path, map_location="cpu")
    except Exception:
        return None
    payload = state.get("model_config") if isinstance(state, dict) else None
    if not isinstance(payload, dict):
        return None
    try:
        return ModelConfig(**normalize_model_config_payload(dict(payload)))
    except Exception:
        return None


def _load_model_config_from_checkpoint_metadata(model_path: Path) -> ModelConfig | None:
    """Load model configuration from safetensors metadata payload."""

    metadata = _load_safetensors_metadata(model_path)
    payload = _parse_json_metadata_value(metadata, "model_config")
    if payload is None:
        return None
    try:
        return ModelConfig(**normalize_model_config_payload(dict(payload)))
    except Exception:
        return None


def _validate_model_config_against_state(
    model_cfg: ModelConfig, state_dict: Dict[str, torch.Tensor], model_path: Path
) -> None:
    """Validate config dimensions match checkpoint tensor shapes."""

    token_weight = state_dict.get("token_embedding.weight")
    if token_weight is None:
        raise RuntimeError(
            f"Checkpoint {model_path.name} is missing token_embedding.weight; "
            "cannot validate model dimensions."
        )

    inferred_vocab = int(token_weight.shape[0])
    inferred_d_model = int(token_weight.shape[1])
    if model_cfg.vocab_size != inferred_vocab or model_cfg.d_model != inferred_d_model:
        raise RuntimeError(
            f"Checkpoint {model_path.name} config mismatch: metadata/sidecar says "
            f"vocab_size={model_cfg.vocab_size}, d_model={model_cfg.d_model}, but "
            f"weights are vocab_size={inferred_vocab}, d_model={inferred_d_model}."
        )


def _load_state_dict(model_path: Path) -> Dict[str, torch.Tensor]:
    """Load model weights from `.safetensors` or `.pt` checkpoint."""

    if model_path.suffix == ".safetensors":
        raw = safetensors_load_file(str(model_path), device="cpu")
        return _strip_dataparallel_prefix(raw)

    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return _strip_dataparallel_prefix(state["state_dict"])
        tensor_dict = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
        if tensor_dict:
            return _strip_dataparallel_prefix(tensor_dict)
    raise RuntimeError(f"Unsupported checkpoint format: {model_path}")


def _load_model(model_name: str) -> Tuple[Any, Dict[str, Any]]:
    """Load and cache a model by checkpoint filename."""

    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = None
    for directory in MODEL_SEARCH_DIRS:
        candidate = directory / model_name
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError(f"Checkpoint not found in app/models: {model_name}")

    sidecar = _resolve_sidecar(model_path)
    checkpoint_metadata = _resolve_checkpoint_metadata(model_path, sidecar)
    model_cfg = _load_model_config_from_checkpoint_metadata(model_path)
    if model_cfg is None:
        model_cfg = _load_model_config_from_sidecar(sidecar)

    state_dict = _load_state_dict(model_path)
    if model_cfg is None:
        raise RuntimeError(
            f"Checkpoint {model_path.name} is missing model_config metadata and sidecar "
            "model config. Re-save the checkpoint with training metadata or provide "
            "the matching *_state.pt sidecar."
        )
    model_cfg = ModelConfig(**normalize_model_config_payload(dict(model_cfg.__dict__)))
    _validate_model_config_against_state(model_cfg, state_dict, model_path)

    model = build_model(model_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(torch.device("cpu"))

    meta = {
        "missing_keys": int(len(missing)),
        "unexpected_keys": int(len(unexpected)),
        "checkpoint": str(model_path.name),
        "model_config": dict(model_cfg.__dict__),
        "checkpoint_metadata": checkpoint_metadata,
    }
    _model_cache[model_name] = (model, meta)
    return model, meta


def _load_tokenizer() -> PianoTokenizer:
    """Load tokenizer from known search paths with cache."""

    global _tokenizer_cache
    if _tokenizer_cache is not None:
        return _tokenizer_cache
    for tokenizer_path in TOKENIZER_SEARCH_PATHS:
        if tokenizer_path.exists():
            _tokenizer_cache = PianoTokenizer.load(str(tokenizer_path))
            return _tokenizer_cache
    raise FileNotFoundError(
        f"Tokenizer not found in: {', '.join(str(p) for p in TOKENIZER_SEARCH_PATHS)}"
    )


def _list_models() -> list[str]:
    """List available checkpoint files in app model directories."""

    allowed = {".safetensors", ".pt"}
    ignored_pt_names = {"latest_state.pt", "best_state.pt"}
    models = []
    seen: set[str] = set()
    for directory in MODEL_SEARCH_DIRS:
        if not directory.exists():
            continue
        for p in sorted(directory.iterdir()):
            if not p.is_file() or p.suffix.lower() not in allowed:
                continue
            if p.suffix.lower() == ".pt":
                if p.name in ignored_pt_names or p.name.endswith("_state.pt"):
                    continue
            if p.name in seen:
                continue
            seen.add(p.name)
            models.append(p.name)
    return models


def _compute_repetition_warning(tokens: list[int]) -> str | None:
    """Return warning text when generated token repetition is too high."""

    if not tokens:
        return None
    counts: Dict[int, int] = {}
    for token in tokens:
        t = int(token)
        counts[t] = counts.get(t, 0) + 1
    ratio = max(counts.values()) / max(1, len(tokens))
    if ratio > 0.60:
        return f"Generated continuation has high repetition ({ratio * 100:.1f}% identical token share)."
    return None


def _decode_and_write(
    tokenizer: PianoTokenizer, tokens: list[int], out_path: Path
) -> None:
    """Decode generated tokens and write output MIDI to disk."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.decode(tokens, out_path)


@app.get("/")
def index() -> str:
    """Serve web UI homepage."""

    return render_template("index.html")


@app.get("/api/models")
def api_models() -> Any:
    """Return list of available checkpoints for UI selection."""

    return jsonify({"models": _list_models()})


@app.post("/api/generate")
def api_generate() -> Any:
    """Run generation request from uploaded seed MIDI."""

    try:
        model_name = str(request.form.get("model_name", "")).strip()
        if not model_name:
            return jsonify({"error": "model_name is required"}), 400

        if "seed_file" not in request.files:
            return jsonify({"error": "seed_file is required"}), 400
        seed_file = request.files["seed_file"]
        if not seed_file or not seed_file.filename:
            return jsonify({"error": "seed_file is empty"}), 400

        temperature = max(0.1, float(request.form.get("temperature", 0.9)))
        length = max(1, int(request.form.get("length", 512)))
        top_p = float(request.form.get("top_p", 0.95))
        top_k = max(3, int(request.form.get("top_k", 50)))

        model, meta = _load_model(model_name)
        tokenizer = _load_tokenizer()

        seed_tmp = RUNTIME_DIR / f"seed_{uuid.uuid4().hex}.mid"
        seed_file.save(seed_tmp)

        seed_tokens = tokenizer.encode(seed_tmp)
        if not seed_tokens:
            return jsonify({"error": "Tokenizer produced no tokens for seed file"}), 400

        generated = model.generate(
            seed_tokens=seed_tokens,
            max_new_tokens=length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,
            repetition_window=64,
            min_tokens_to_keep=3,
        )

        new_tokens = generated[len(seed_tokens) :]
        warning = _compute_repetition_warning(new_tokens)

        health_report = model.generation_health_check(
            seed_tokens=seed_tokens,
            steps=min(20, max(1, int(length))),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,
            repetition_window=64,
            min_tokens_to_keep=3,
            top1_threshold=0.95,
            raise_on_failure=False,
        )

        out_name = f"continuation_{uuid.uuid4().hex}.mid"
        out_path = OUTPUT_DIR / out_name
        _decode_and_write(tokenizer, generated, out_path)

        payload = {
            "ok": True,
            "download_url": f"/api/download/{out_name}",
            "generated_tokens": int(len(new_tokens)),
            "generation_stats": dict(getattr(model, "last_generation_stats", {})),
            "health_report": health_report,
            "health_warning": warning,
            "checkpoint_meta": meta,
        }
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/download/<path:file_name>")
def api_download(file_name: str) -> Any:
    """Download generated MIDI artifact by file name."""

    target = OUTPUT_DIR / file_name
    if not target.exists() or not target.is_file():
        return jsonify({"error": "File not found"}), 404
    with target.open("rb") as f:
        data = io.BytesIO(f.read())
    data.seek(0)
    return send_file(
        data,
        mimetype="audio/midi",
        as_attachment=True,
        download_name=target.name,
    )


if __name__ == "__main__":
    host = os.environ.get("IBP_HOST", "127.0.0.1")
    port = int(os.environ.get("IBP_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
