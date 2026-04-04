from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from model.variant_a import VariantAConfig, VariantAModel
    from model.variant_b import VariantBConfig, VariantBModel
    from model.variant_c import VariantCConfig, VariantCModel
    from model.variant_d import VariantDConfig, VariantDModel
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from model.variant_a import VariantAConfig, VariantAModel
    from model.variant_b import VariantBConfig, VariantBModel
    from model.variant_c import VariantCConfig, VariantCModel
    from model.variant_d import VariantDConfig, VariantDModel


ARCH_LABELS: Dict[str, str] = {
    "variant_a": "gated_delta_cfc_attention_hybrid",
    "variant_b": "transformer_cfc_hybrid",
    "variant_c": "pure_attention_transformer_baseline",
    "variant_d": "pure_cfc_recurrent_baseline",
}

BALANCED_SMALL_PROFILES: Dict[str, Dict[str, int]] = {
    "variant_a": {"d_model": 544, "n_layers": 4},
    "variant_b": {"d_model": 544, "n_layers": 5},
    "variant_c": {"d_model": 480, "n_layers": 4},
    "variant_d": {"d_model": 608, "n_layers": 8},
}


@dataclass
class CheckItem:
    name: str
    status: str
    detail: str


def _resolve_num_heads(d_model: int, requested_heads: int) -> int:
    heads = max(1, min(int(requested_heads), int(d_model)))
    while heads > 1 and (int(d_model) % heads) != 0:
        heads -= 1
    return max(1, heads)


def _parse_variants(raw: str) -> List[str]:
    mapping = {
        "a": "variant_a",
        "variant_a": "variant_a",
        "b": "variant_b",
        "variant_b": "variant_b",
        "c": "variant_c",
        "variant_c": "variant_c",
        "baseline": "variant_c",
        "d": "variant_d",
        "variant_d": "variant_d",
        "pure_cfc": "variant_d",
        "cfc_only": "variant_d",
    }
    out: List[str] = []
    seen = set()
    for token in str(raw).split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Unsupported variant token '{token}'")
        name = mapping[key]
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    if not out:
        raise ValueError("No variants selected.")
    return out


def _variant_backend_status(model: torch.nn.Module) -> Dict[str, bool]:
    status = {
        "gdn_using_fallback": False,
        "cfc_using_fallback": False,
    }
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name == "GatedDeltaNetBlock" and bool(
            getattr(module, "using_fallback", False)
        ):
            status["gdn_using_fallback"] = True
        if cls_name == "CfCBlock" and bool(getattr(module, "using_fallback", False)):
            status["cfc_using_fallback"] = True
    return status


def _dependency_checks() -> List[CheckItem]:
    checks: List[CheckItem] = []

    def check_import(module_name: str, label: str) -> None:
        try:
            __import__(module_name)
            checks.append(CheckItem(label, "PASS", f"import {module_name} ok"))
        except Exception as exc:
            checks.append(CheckItem(label, "WARN", f"import {module_name} failed: {exc}"))

    check_import("ncps", "CfC runtime")
    check_import("symusic", "Symusic tokenizer runtime")

    try:
        fla_spec = importlib.util.find_spec("fla.layers")
    except ModuleNotFoundError:
        fla_spec = None
    if fla_spec is not None:
        checks.append(
            CheckItem("GatedDeltaNet kernel", "PASS", "flash-linear-attention available")
        )
    else:
        checks.append(
            CheckItem(
                "GatedDeltaNet kernel",
                "WARN",
                "flash-linear-attention unavailable; Variant A uses fallback",
            )
        )

    try:
        mamba_spec = importlib.util.find_spec("mamba_ssm")
    except ModuleNotFoundError:
        mamba_spec = None
    if mamba_spec is not None:
        checks.append(CheckItem("Mamba kernel", "PASS", "mamba_ssm available"))
    else:
        checks.append(
            CheckItem(
                "Mamba kernel",
                "PASS",
                "mamba_ssm unavailable (optional for architecture ablation)",
            )
        )

    return checks


def _build_variant(
    variant_name: str,
    profile: Dict[str, int],
    max_sequence_length: int,
) -> Tuple[torch.nn.Module, Dict[str, int]]:
    d_model = int(profile["d_model"])
    n_layers = int(profile["n_layers"])
    attn_heads = _resolve_num_heads(d_model=d_model, requested_heads=8)
    gdn_heads = _resolve_num_heads(d_model=d_model, requested_heads=4)
    gqa_groups = 4 if attn_heads % 4 == 0 else (2 if attn_heads % 2 == 0 else 1)
    gdn_inner_dim = max(128, d_model // 2)
    cfc_backbone_units = max(128, int(d_model * 0.75))

    if variant_name == "variant_a":
        model = VariantAModel(
            VariantAConfig(
                vocab_size=155,
                d_model=d_model,
                n_layers=n_layers,
                max_sequence_length=max_sequence_length,
                gdn_inner_dim=gdn_inner_dim,
                gdn_num_heads=gdn_heads,
                cfc_backbone_units=cfc_backbone_units,
                gqa_num_heads=attn_heads,
                gqa_groups=gqa_groups,
            )
        )
    elif variant_name == "variant_b":
        model = VariantBModel(
            VariantBConfig(
                vocab_size=155,
                d_model=d_model,
                n_layers=n_layers,
                max_sequence_length=max_sequence_length,
                num_attention_heads=attn_heads,
                cfc_backbone_units=cfc_backbone_units,
            )
        )
    elif variant_name == "variant_c":
        model = VariantCModel(
            VariantCConfig(
                vocab_size=155,
                d_model=d_model,
                n_layers=n_layers,
                max_sequence_length=max_sequence_length,
                num_attention_heads=attn_heads,
            )
        )
    elif variant_name == "variant_d":
        model = VariantDModel(
            VariantDConfig(
                vocab_size=155,
                d_model=d_model,
                n_layers=n_layers,
                max_sequence_length=max_sequence_length,
                cfc_backbone_units=cfc_backbone_units,
            )
        )
    else:
        raise ValueError(f"Unsupported variant {variant_name}")

    shape = {
        "d_model": d_model,
        "n_layers": n_layers,
        "attention_heads": int(attn_heads),
    }
    return model, shape


def _variant_checks(
    variants: List[str],
    *,
    size_mode: str,
    shared_d_model: int,
    shared_n_layers: int,
) -> Tuple[List[CheckItem], Dict[str, Dict[str, Any]]]:
    checks: List[CheckItem] = []
    details: Dict[str, Dict[str, Any]] = {}

    profiles: Dict[str, Dict[str, int]] = {}
    for name in variants:
        if size_mode == "balanced_small":
            base = BALANCED_SMALL_PROFILES[name]
            profiles[name] = {
                "d_model": int(base["d_model"]),
                "n_layers": int(base["n_layers"]),
            }
        else:
            profiles[name] = {
                "d_model": int(shared_d_model),
                "n_layers": int(shared_n_layers),
            }

    if size_mode == "balanced_small" and "variant_a" in variants:
        baseline_params: List[int] = []
        for baseline_name in ("variant_b", "variant_c"):
            if baseline_name not in variants:
                continue
            baseline_model, _ = _build_variant(
                variant_name=baseline_name,
                profile=profiles[baseline_name],
                max_sequence_length=1024,
            )
            baseline_params.append(int(sum(p.numel() for p in baseline_model.parameters())))
            del baseline_model

        target_params = (
            int(sum(baseline_params) / len(baseline_params))
            if baseline_params
            else 12_000_000
        )
        original = dict(profiles["variant_a"])

        candidates: List[Tuple[Tuple[int, int, int, int], Dict[str, int], int]] = []
        for d_model in range(416, 577, 32):
            for n_layers in (3, 4, 5):
                profile = {"d_model": int(d_model), "n_layers": int(n_layers)}
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cand_model, _ = _build_variant(
                            variant_name="variant_a",
                            profile=profile,
                            max_sequence_length=1024,
                        )
                    params = int(sum(p.numel() for p in cand_model.parameters()))
                    del cand_model
                except Exception:
                    continue

                in_budget = 10_000_000 <= params <= 15_000_000
                score = (
                    0 if in_budget else 1,
                    abs(params - int(target_params)),
                    abs(int(n_layers) - int(original["n_layers"])),
                    abs(int(d_model) - int(original["d_model"])),
                )
                candidates.append((score, profile, int(params)))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            _score, best_profile, _params = candidates[0]
            profiles["variant_a"] = best_profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in variants:
        profile = profiles[name]

        model, shape = _build_variant(
            variant_name=name,
            profile=profile,
            max_sequence_length=1024,
        )
        model = model.to(device)
        params = int(sum(p.numel() for p in model.parameters()))
        backend = _variant_backend_status(model)

        # Quick forward smoke check.
        token_ids = torch.randint(
            low=0,
            high=155,
            size=(2, 96),
            dtype=torch.long,
            device=device,
        )
        onsets = (
            torch.arange(96, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(2, 1)
            * 0.1
        )
        try:
            with torch.no_grad():
                out = model(
                    token_ids=token_ids,
                    onset_times=onsets,
                    memory=None,
                    return_memory=False,
                    position_offset=0,
                )
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            ok = isinstance(logits, torch.Tensor) and tuple(logits.shape) == (2, 96, 155)
            if ok:
                checks.append(
                    CheckItem(
                        f"{name} forward",
                        "PASS",
                        f"shape={tuple(logits.shape)} params={params / 1e6:.2f}M",
                    )
                )
            else:
                checks.append(
                    CheckItem(
                        f"{name} forward",
                        "WARN",
                        f"unexpected output format, params={params / 1e6:.2f}M",
                    )
                )
        except Exception as exc:
            checks.append(CheckItem(f"{name} forward", "WARN", f"forward failed: {exc}"))

        details[name] = {
            "architecture": ARCH_LABELS[name],
            "params": params,
            "shape": shape,
            "backend_status": backend,
        }
        del model

    if details:
        values = [int(v["params"]) for v in details.values()]
        min_v = min(values)
        max_v = max(values)
        ratio = float(max_v) / float(max(1, min_v))
        status = "PASS" if ratio <= 1.20 else "WARN"
        checks.append(
            CheckItem(
                "Variant parameter comparability",
                status,
                f"min={min_v / 1e6:.2f}M max={max_v / 1e6:.2f}M ratio={ratio:.3f}",
            )
        )

    return checks, details


def _manifest_checks(
    manifest_path: Optional[Path],
    pretokenized_root: Optional[Path],
) -> List[CheckItem]:
    checks: List[CheckItem] = []
    if manifest_path is None:
        checks.append(
            CheckItem(
                "Pre-tokenized manifest",
                "WARN",
                "not provided (audit skipped for tokenized data integrity)",
            )
        )
        return checks

    if not manifest_path.exists():
        checks.append(CheckItem("Pre-tokenized manifest", "WARN", f"not found: {manifest_path}"))
        return checks

    try:
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        checks.append(CheckItem("Pre-tokenized manifest", "WARN", f"failed to parse JSON: {exc}"))
        return checks

    if not isinstance(rows, list) or not rows:
        checks.append(CheckItem("Pre-tokenized manifest", "WARN", "manifest is empty or invalid"))
        return checks

    checks.append(CheckItem("Pre-tokenized manifest", "PASS", f"entries={len(rows):,}"))

    sample_ok = 0
    sample_bad = 0
    checked = 0
    parent = manifest_path.parent
    for row in rows:
        if checked >= 5:
            break
        if not isinstance(row, dict):
            continue
        raw = str(row.get("npz_path", "")).strip()
        if not raw:
            continue

        path = Path(raw)
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            if pretokenized_root is not None:
                candidates.append(pretokenized_root / path)
                candidates.append(pretokenized_root / path.name)
            candidates.append(parent / path)
            candidates.append(parent.parent / path)
            candidates.append(parent.parent / "data" / path.name)

        resolved = next((p for p in candidates if p.exists()), None)
        checked += 1
        if resolved is None:
            sample_bad += 1
            continue

        try:
            with np.load(resolved, allow_pickle=False) as pack:
                _ = pack["tokens"]
                _ = pack["onsets"] if "onsets" in pack else pack["onset_times"]
                _ = pack["durations"]
            sample_ok += 1
        except Exception:
            sample_bad += 1

    status = "PASS" if sample_bad == 0 and sample_ok > 0 else "WARN"
    checks.append(
        CheckItem(
            "Pre-tokenized NPZ sample",
            status,
            f"sample_ok={sample_ok} sample_bad={sample_bad}",
        )
    )
    return checks


def _render_report(
    *,
    output_path: Path,
    env_checks: List[CheckItem],
    dep_checks: List[CheckItem],
    variant_checks: List[CheckItem],
    manifest_checks: List[CheckItem],
    variant_details: Dict[str, Dict[str, Any]],
) -> None:
    all_checks = env_checks + dep_checks + variant_checks + manifest_checks
    has_warn = any(c.status == "WARN" for c in all_checks)
    overall = "WARN" if has_warn else "PASS"

    lines: List[str] = []
    lines.append("# Architecture Readiness Audit")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Overall: {overall}")
    lines.append("")

    def add_section(title: str, items: List[CheckItem]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Check | Status | Detail |")
        lines.append("|---|---|---|")
        for item in items:
            lines.append(f"| {item.name} | {item.status} | {item.detail} |")
        lines.append("")

    add_section("Environment", env_checks)
    add_section("Dependencies", dep_checks)
    add_section("Variant Smoke Checks", variant_checks)
    add_section("Tokenized Data", manifest_checks)

    lines.append("## Variant Details")
    lines.append("")
    lines.append("| Variant | Architecture | Params (M) | d_model | n_layers | Backend Status |")
    lines.append("|---|---|---:|---:|---:|---|")
    for name in sorted(variant_details.keys()):
        row = variant_details[name]
        shape = row["shape"]
        backend = row["backend_status"]
        lines.append(
            "| "
            + f"{name} | {row['architecture']} | {row['params'] / 1e6:.2f} | "
            + f"{shape['d_model']} | {shape['n_layers']} | {backend} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit readiness for small architecture ablation runs.")
    parser.add_argument("--variants", type=str, default="a,b,c")
    parser.add_argument(
        "--size_mode",
        type=str,
        choices=["balanced_small", "shared"],
        default="balanced_small",
    )
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--pretokenized_manifest", type=str, default="")
    parser.add_argument("--pretokenized_root", type=str, default="")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ablation_audit_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = _parse_variants(args.variants)

    env_checks = [
        CheckItem("Python", "PASS", platform.python_version()),
        CheckItem("PyTorch", "PASS", torch.__version__),
    ]

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            env_checks.append(CheckItem("CUDA", "PASS", f"available ({name})"))
        except Exception:
            env_checks.append(CheckItem("CUDA", "PASS", "available"))
    else:
        env_checks.append(CheckItem("CUDA", "WARN", "not available"))

    dep_checks = _dependency_checks()
    variant_checks, variant_details = _variant_checks(
        variants,
        size_mode=str(args.size_mode),
        shared_d_model=int(args.d_model),
        shared_n_layers=int(args.n_layers),
    )

    manifest_path = Path(args.pretokenized_manifest) if str(args.pretokenized_manifest).strip() else None
    pretokenized_root = Path(args.pretokenized_root) if str(args.pretokenized_root).strip() else None
    manifest_checks = _manifest_checks(manifest_path, pretokenized_root)

    output_path = Path(args.output)
    _render_report(
        output_path=output_path,
        env_checks=env_checks,
        dep_checks=dep_checks,
        variant_checks=variant_checks,
        manifest_checks=manifest_checks,
        variant_details=variant_details,
    )

    print(f"Audit report written: {output_path.resolve()}")


if __name__ == "__main__":
    main()
