"""
Smoke test for Model v2 architecture and generation health.
Run with: python smoke_test_architecture.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from safetensors.torch import load_file as safetensors_load_file

from config import ModelConfig
from model.attention_block import ALiBiPositionBias, MusicAttentionBlock
from model.hybrid import PianoHybridModel
from scale_config import SCALE_PRESETS, verify_preset_params
from utils.config_compat import normalize_model_config_payload


ROOT = Path(__file__).resolve().parent
LOCAL_CKPT = ROOT / "local_drive" / "piano_model" / "checkpoints"


def _print_ok(label: str, payload: str) -> None:
    print(f"  {label}: OK - {payload}")


def test_alibi_bias() -> None:
    print("Testing ALiBiPositionBias...")
    bias = ALiBiPositionBias(num_heads=4)
    out = bias(seq_len=128, device=torch.device("cpu"))
    assert out.shape == (4, 128, 128), f"Wrong shape: {out.shape}"
    assert torch.all(out[:, torch.arange(128), torch.arange(128)] == 0), (
        "ALiBi diagonal should be zero"
    )
    _print_ok("ALiBiPositionBias", f"shape={tuple(out.shape)}")


def test_attention_block() -> None:
    print("Testing MusicAttentionBlock (ALiBi)...")
    block = MusicAttentionBlock(
        d_model=128,
        num_heads=4,
        max_relative_distance=64,
        use_relative_bias=True,
        bias_type="alibi",
    )
    x = torch.randn(2, 64, 128)
    out = block(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    _print_ok("MusicAttentionBlock", f"input/output={tuple(x.shape)}")


def _build_test_config(use_cfc: bool) -> ModelConfig:
    return ModelConfig(
        vocab_size=2000,
        d_model=192,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        cfc_units=192,
        cfc_backbone_units=96,
        cfc_backbone_layers=2,
        cfc_every_n_layers=2,
        num_attention_heads=4,
        attention_every_n_layers=2,
        attention_dropout=0.1,
        use_relative_attention=True,
        attention_bias_type="alibi",
        max_relative_distance=128,
        dropout=0.1,
        use_cfc=use_cfc,
        use_mamba=True,
        ffn_expansion=4,
        tie_embeddings=True,
        embedding_init_std=0.02,
        output_logit_scale=None,
        max_sequence_length=1024,
        debug=False,
    )


def test_hybrid_forward() -> None:
    print("Testing PianoHybridModel forward (CfC off)...")
    cfg = _build_test_config(use_cfc=False)
    model = PianoHybridModel(cfg).eval()
    with torch.no_grad():
        input_ids = torch.randint(0, cfg.vocab_size, (2, 64))
        logits, hidden = model(input_ids)
    assert logits.shape == (2, 64, cfg.vocab_size)
    assert hidden is None
    _print_ok("Forward CfC off", f"logits={tuple(logits.shape)}")

    print("Testing PianoHybridModel forward (CfC on)...")
    cfg_cfc = _build_test_config(use_cfc=True)
    model_cfc = PianoHybridModel(cfg_cfc).eval()
    with torch.no_grad():
        input_ids = torch.randint(0, cfg_cfc.vocab_size, (2, 64))
        logits, hidden = model_cfc(input_ids)
    assert logits.shape == (2, 64, cfg_cfc.vocab_size)
    assert isinstance(hidden, list)
    expected_cfc_layers = len(
        [i for i in range(cfg_cfc.n_layers) if i % cfg_cfc.cfc_every_n_layers == 0]
    )
    assert len(hidden) == expected_cfc_layers
    _print_ok("Forward CfC on", f"hidden_states={len(hidden)}")


def test_generation_health_random_weights() -> None:
    print("Testing generation distribution with random weights...")
    cfg = _build_test_config(use_cfc=False)
    model = PianoHybridModel(cfg).eval()
    seed = torch.randint(0, cfg.vocab_size, (64,))
    report = model.generation_health_check(
        seed_tokens=seed,
        steps=20,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        min_tokens_to_keep=3,
        top1_threshold=0.95,
        raise_on_failure=True,
    )
    _print_ok(
        "Generation health random",
        f"max_final_top1={report['max_final_top1_prob']:.4f}, "
        f"max_raw_top1={report['max_raw_top1_prob']:.4f}",
    )


def _load_v1_checkpoint_report() -> Tuple[bool, Dict[str, Any]]:
    state_path = LOCAL_CKPT / "latest_state.pt"
    model_path = LOCAL_CKPT / "latest_model.safetensors"
    if not state_path.exists() or not model_path.exists():
        return False, {}

    state = torch.load(state_path, map_location="cpu")
    cfg_payload = state.get("model_config")
    if not isinstance(cfg_payload, dict):
        return False, {}

    cfg = ModelConfig(**normalize_model_config_payload(dict(cfg_payload)))
    model = PianoHybridModel(cfg).eval()
    model_state = safetensors_load_file(str(model_path), device="cpu")
    missing, unexpected = model.load_state_dict(model_state, strict=False)

    seed = torch.randint(0, cfg.vocab_size, (64,))
    report = model.generation_health_check(
        seed_tokens=seed,
        steps=20,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        min_tokens_to_keep=3,
        top1_threshold=0.95,
        raise_on_failure=False,
    )
    report.update(
        {
            "checkpoint_epoch": state.get("epoch"),
            "checkpoint_val_loss": state.get("val_loss"),
            "missing_keys": int(len(missing)),
            "unexpected_keys": int(len(unexpected)),
            "used_checkpoint": True,
        }
    )
    return True, report


def test_v1_checkpoint_generation_report() -> None:
    print("Testing existing v1 checkpoint with new generate()...")
    ok, report = _load_v1_checkpoint_report()
    if not ok:
        print("  v1 checkpoint: SKIPPED - no local checkpoint found")
        return

    assert bool(report.get("passed", False)), (
        "v1 checkpoint failed generation health check: "
        f"max_final_top1={float(report.get('max_final_top1_prob', 0.0)):.4f}"
    )

    print(
        "  v1 checkpoint report: "
        f"epoch={report.get('checkpoint_epoch')} "
        f"val_loss={report.get('checkpoint_val_loss')} "
        f"max_final_top1={float(report.get('max_final_top1_prob', 0.0)):.4f} "
        f"max_raw_top1={float(report.get('max_raw_top1_prob', 0.0)):.4f} "
        f"passed={bool(report.get('passed', False))} "
        f"missing={int(report.get('missing_keys', 0))} "
        f"unexpected={int(report.get('unexpected_keys', 0))}"
    )


def test_parameter_counts() -> None:
    print("Verifying parameter counts for all presets...")
    verify_preset_params()


if __name__ == "__main__":
    print("=" * 56)
    print("Model v2 smoke test")
    print("=" * 56)
    test_alibi_bias()
    test_attention_block()
    test_hybrid_forward()
    test_generation_health_random_weights()
    test_v1_checkpoint_generation_report()
    test_parameter_counts()
    print("=" * 56)
    print("All smoke tests completed.")
    print("=" * 56)
