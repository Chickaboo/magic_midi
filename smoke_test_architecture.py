"""
Smoke test for upgraded architecture.
Run with: python smoke_test_architecture.py
Tests all four scale presets for correct parameter counts and forward pass.
"""

import sys
import time

import torch

sys.path.insert(0, ".")

from config import ModelConfig
from model.attention_block import MusicAttentionBlock, RelativePositionBias
from model.hybrid import PianoHybridModel
from scale_config import SCALE_PRESETS, verify_preset_params


def test_relative_position_bias() -> None:
    print("Testing RelativePositionBias...")
    rpb = RelativePositionBias(max_distance=64, num_heads=4)
    bias = rpb(seq_len=128, device=torch.device("cpu"))
    assert bias.shape == (4, 128, 128), f"Wrong shape: {bias.shape}"
    expected_table_shape = (2 * 64 + 1, 4)
    actual_table_shape = tuple(rpb.embeddings.weight.shape)
    assert actual_table_shape == expected_table_shape, (
        f"Wrong embedding table shape: {actual_table_shape} vs {expected_table_shape}"
    )
    print(f"  RelativePositionBias: OK - shape {bias.shape}")
    print(f"  Bias table: OK - shape {actual_table_shape}")


def test_attention_block() -> None:
    print("Testing MusicAttentionBlock...")
    block = MusicAttentionBlock(d_model=128, num_heads=4, max_relative_distance=64)
    x = torch.randn(2, 64, 128)
    out = block(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"  MusicAttentionBlock: OK - input/output shape {x.shape}")


def test_hybrid_model_forward() -> None:
    print("Testing PianoHybridModel forward pass (nano preset)...")
    preset = SCALE_PRESETS["nano"]
    model = PianoHybridModel(preset["model"])
    model.eval()
    with torch.no_grad():
        input_ids = torch.randint(0, preset["model"].vocab_size, (2, 64))
        logits, hidden = model(input_ids)

    assert logits.shape == (2, 64, preset["model"].vocab_size)
    if preset["model"].use_cfc:
        assert hidden is not None, "Expected CfC hidden states"
        expected_cfc_layers = len(
            [
                i
                for i in range(preset["model"].n_layers)
                if i % preset["model"].cfc_every_n_layers == 0
            ]
        )
        assert len(hidden) == expected_cfc_layers
    print(f"  Forward pass: OK - logits shape {logits.shape}")


def test_generation() -> None:
    print("Testing generation (nano preset)...")
    preset = SCALE_PRESETS["nano"]
    model = PianoHybridModel(preset["model"])
    model.eval()
    seed = torch.randint(0, preset["model"].vocab_size, (10,))
    with torch.no_grad():
        generated = model.generate(seed, max_new_tokens=20, temperature=0.9)
    assert len(generated) == 30, f"Expected 30 tokens, got {len(generated)}"
    print(f"  Generation: OK - {len(generated)} tokens generated")


def test_parameter_counts() -> None:
    print("Verifying parameter counts for all presets...")
    verify_preset_params()


def profile_cfc_cost(d_model: int = 128, seq_len: int = 384, batch: int = 4) -> None:
    print("Profiling CfC cost...")

    cfg_with = ModelConfig(
        vocab_size=2000,
        d_model=d_model,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        cfc_units=d_model,
        cfc_backbone_units=max(32, d_model // 2),
        cfc_backbone_layers=2,
        cfc_every_n_layers=1,
        num_attention_heads=4,
        attention_every_n_layers=2,
        max_sequence_length=max(1024, seq_len),
        use_cfc=True,
    )
    cfg_without = ModelConfig(
        vocab_size=2000,
        d_model=d_model,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        cfc_units=d_model,
        cfc_backbone_units=max(32, d_model // 2),
        cfc_backbone_layers=2,
        cfc_every_n_layers=1,
        num_attention_heads=4,
        attention_every_n_layers=2,
        max_sequence_length=max(1024, seq_len),
        use_cfc=False,
    )

    model_with = PianoHybridModel(cfg_with).eval()
    model_without = PianoHybridModel(cfg_without).eval()
    input_ids = torch.randint(0, cfg_with.vocab_size, (batch, seq_len))

    with torch.no_grad():
        for _ in range(2):
            model_with(input_ids)
            model_without(input_ids)

        t0 = time.perf_counter()
        for _ in range(10):
            model_with(input_ids)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        for _ in range(10):
            model_without(input_ids)
        t3 = time.perf_counter()

    with_cfc = t1 - t0
    without_cfc = t3 - t2
    ratio = with_cfc / max(without_cfc, 1e-8)
    print(
        f"  10 fwd passes with CfC:    {with_cfc:.3f}s\n"
        f"  10 fwd passes without CfC: {without_cfc:.3f}s\n"
        f"  Slowdown ratio (with/without): {ratio:.2f}x"
    )


if __name__ == "__main__":
    print("=" * 50)
    print("Architecture smoke test")
    print("=" * 50)
    test_relative_position_bias()
    test_attention_block()
    test_hybrid_model_forward()
    test_generation()
    test_parameter_counts()
    profile_cfc_cost()
    print("=" * 50)
    print("All tests passed.")
    print("=" * 50)
