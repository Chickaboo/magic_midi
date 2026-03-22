"""
Smoke test for Model v3 architecture and generation health.
Run with: python smoke_test_architecture.py
"""

from __future__ import annotations

import json
from pathlib import Path
import math
import tempfile
from typing import Any, Dict, Tuple, cast
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file as safetensors_load_file

from config import ModelConfig
from data.dataset import create_dataloaders
from data.preprocess import MultiDatasetPreprocessor
from data.tokenizer import PianoTokenizer
from model.attention_block import ALiBiPositionBias, MusicAttentionBlock
from model.factory import build_model
from model.hybrid import PianoHybridModel
from model.hybrid_v2 import IttyBittyPianoV2
from scale_config import SCALE_PRESETS, verify_preset_params
from training import trainer as trainer_module
from training.trainer import CHECKPOINT_KEEP_POLICY, rotate_kaggle_checkpoint_dir
from utils.config_compat import normalize_model_config_payload


ROOT = Path(__file__).resolve().parent
LOCAL_CKPT = ROOT / "local_drive" / "piano_model" / "checkpoints"


def _print_ok(label: str, payload: str) -> None:
    """Print a standardized success line for smoke tests."""

    print(f"  {label}: OK - {payload}")


def test_alibi_bias() -> None:
    """Validate ALiBi bias shape and diagonal values."""

    print("Testing ALiBiPositionBias...")
    bias = ALiBiPositionBias(num_heads=4)
    out = bias(seq_len=128, device=torch.device("cpu"))
    assert out.shape == (4, 128, 128), f"Wrong shape: {out.shape}"
    assert torch.all(out[:, torch.arange(128), torch.arange(128)] == 0), (
        "ALiBi diagonal should be zero"
    )
    _print_ok("ALiBiPositionBias", f"shape={tuple(out.shape)}")


def test_attention_block() -> None:
    """Validate sparse attention block forward shape."""

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
    """Create compact test config with optional CfC blocks."""

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
    """Validate forward pass with and without CfC enabled."""

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
    """Ensure generation health check runs on random weights."""

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
    """Load local legacy checkpoint and run generation-health report."""

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
    """Validate generation-health compatibility with legacy checkpoints."""

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
    """Print parameter verification report for all presets."""

    print("Verifying parameter counts for all presets...")
    verify_preset_params()


def test_cfc_generation_state_threading() -> None:
    """Ensure CfC generation path with recurrent hidden state works end-to-end."""

    print("Testing generation with CfC hidden-state threading...")
    cfg = _build_test_config(use_cfc=True)
    model = PianoHybridModel(cfg).eval()
    seed = torch.randint(0, cfg.vocab_size, (64,))
    out = model.generate(
        seed_tokens=seed,
        max_new_tokens=16,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        min_tokens_to_keep=3,
    )
    assert len(out) == 80
    _print_ok("CfC generation threading", f"generated_tokens={len(out)}")


def test_checkpoint_rotation_policy() -> None:
    """Simulate epoch checkpoints and verify aggressive retention policy."""

    print("Testing checkpoint rotation policy...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir)

        for epoch in range(1, 31):
            model_path = ckpt_dir / f"epoch_{epoch:03d}.safetensors"
            state_path = ckpt_dir / f"epoch_{epoch:03d}_state.pt"
            model_path.write_bytes(b"x")
            state_path.write_bytes(b"x")

        for name in (
            "best.safetensors",
            "best_state.pt",
            "latest.safetensors",
            "latest_state.pt",
        ):
            (ckpt_dir / name).write_bytes(b"x")

        stats = rotate_kaggle_checkpoint_dir(
            ckpt_dir,
            keep_every_n_epochs=int(CHECKPOINT_KEEP_POLICY["milestone_every_n"]),
            max_total_checkpoints=int(CHECKPOINT_KEEP_POLICY["max_total_checkpoints"]),
        )

        remaining_models = sorted(p.name for p in ckpt_dir.glob("*.safetensors"))
        assert len(remaining_models) <= int(
            CHECKPOINT_KEEP_POLICY["max_total_checkpoints"]
        )
        assert "best.safetensors" in remaining_models
        assert "latest.safetensors" in remaining_models
        assert "epoch_025.safetensors" in remaining_models

        _print_ok(
            "Checkpoint rotation",
            f"remaining={len(remaining_models)}, deleted={int(stats['deleted_files'])}",
        )


def test_low_disk_emergency_rotation_trigger() -> None:
    """Ensure low-disk pre-save check triggers emergency rotation once."""

    print("Testing low-disk emergency rotation trigger...")

    class _DummyTrainer:
        def __init__(self) -> None:
            self.rotations: list[int] = []
            self.checkpoint_dir = Path(tempfile.gettempdir())

        def _rotate_checkpoints(self, reserve_slots: int = 0) -> Dict[str, float]:
            self.rotations.append(int(reserve_slots))
            return {
                "deleted_files": 0.0,
                "remaining_models": 0.0,
                "max_total_checkpoints": 8.0,
                "reserve_slots": float(reserve_slots),
            }

    dummy = _DummyTrainer()
    with patch("training.trainer.kaggle_free_space_gb", side_effect=[2.0, 5.0]):
        trainer_module.Trainer._pre_save_disk_check(cast(Any, dummy), reserve_slots=1)

    assert dummy.rotations == [1]
    _print_ok("Low-disk trigger", "emergency rotation invoked")


def test_v2_model() -> None:
    """Run required v2 architecture smoke checks."""

    print("Testing v2 model architecture...")
    cfg = SCALE_PRESETS["large_v2"]["model"]
    model = IttyBittyPianoV2(cfg).eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,} ({params / 1e6:.1f}M)")
    assert 80_000_000 < params < 120_000_000, f"Parameter count out of range: {params}"

    batch_size, seq_len = 2, 256
    token_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    onset_times = torch.linspace(0.0, 60.0, seq_len).unsqueeze(0).repeat(batch_size, 1)

    with torch.no_grad():
        logits, memory = model(
            token_ids,
            onset_times,
            memory=None,
            return_memory=True,
        )
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert memory is not None
    _print_ok("V2 forward", f"logits={tuple(logits.shape)}")

    harmonic_params = sum(p.numel() for p in model.harmonic_layers.parameters())
    temporal_params = sum(p.numel() for p in model.temporal_layers.parameters())
    assert harmonic_params > 0
    assert temporal_params > 0
    _print_ok(
        "V2 dual stream",
        f"harmonic_params={harmonic_params:,}, temporal_params={temporal_params:,}",
    )

    expected_cross = math.ceil(cfg.n_layers / cfg.cross_stream_every_n_layers)
    assert len(model.cross_stream_layers) == expected_cross
    _print_ok(
        "V2 cross-stream cadence",
        f"layers={len(model.cross_stream_layers)}, every={cfg.cross_stream_every_n_layers}",
    )

    phrase_repr = model.phrase_summarizer(torch.randn(batch_size, seq_len, cfg.d_model))
    expected_phrases = math.ceil(seq_len / cfg.tokens_per_phrase)
    assert phrase_repr.shape == (
        batch_size,
        expected_phrases,
        int(cfg.phrase_dim or cfg.d_model),
    )
    _print_ok("V2 phrase summarizer", f"shape={tuple(phrase_repr.shape)}")

    with torch.no_grad():
        _, memory_2 = model(
            token_ids,
            onset_times,
            memory=memory,
            return_memory=True,
        )
    assert memory_2 is not None
    assert int(memory_2.shape[1]) <= int(cfg.memory_size)
    _print_ok("V2 memory accumulation", f"memory_slots={int(memory_2.shape[1])}")

    seed = torch.randint(0, cfg.vocab_size, (128,))
    # V2 uses a stricter confidence ceiling because the larger dual-stream model with
    # label smoothing should maintain a more distributed next-token distribution.
    report = model.generation_health_check(
        seed_tokens=seed,
        steps=20,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        repetition_window=64,
        min_tokens_to_keep=3,
        top1_threshold=0.85,
        raise_on_failure=True,
    )
    _print_ok(
        "V2 generation health",
        f"max_final_top1={float(report['max_final_top1_prob']):.4f}",
    )

    memory_state = model.theme_memory.reset()
    assert memory_state is None
    _print_ok("V2 memory reset", "theme_memory.reset() -> None")


def test_model_factory_switch() -> None:
    """Verify model factory returns correct class for v1/v2 presets."""

    print("Testing model factory v1/v2 routing...")
    small_cfg = SCALE_PRESETS["small"]["model"]
    large_v2_cfg = SCALE_PRESETS["large_v2"]["model"]
    m1 = build_model(small_cfg)
    m2 = build_model(large_v2_cfg)
    assert isinstance(m1, PianoHybridModel)
    assert isinstance(m2, IttyBittyPianoV2)
    _print_ok("Model factory", "v1->PianoHybridModel, v2->IttyBittyPianoV2")


def test_time_features_and_weighted_sampling() -> None:
    """Smoke-test onset/duration extraction and dataset weight sampler wiring."""

    print("Testing preprocessing time features and weighted sampling...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        midi_path = tmp / "seed.mid"
        midi_path_2 = tmp / "seed_2.mid"

        import pretty_midi

        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        note_starts = [0.0, 0.5, 1.0, 1.5]
        for rep in range(8):
            for idx, start in enumerate(note_starts):
                shift = float(rep) * 2.0
                piano.notes.append(
                    pretty_midi.Note(
                        velocity=80 + (idx % 16),
                        pitch=60 + idx,
                        start=start + shift,
                        end=start + shift + 0.4,
                    )
                )
        midi.instruments.append(piano)
        midi.write(str(midi_path))
        midi.write(str(midi_path_2))

        from config import DataConfig, TrainConfig

        cfg = DataConfig(
            maestro_path=str(tmp),
            tokenizer_path=str(tmp / "tokenizer.json"),
            processed_path=str(tmp / "processed"),
            seed_length=4,
            continuation_length=4,
            min_piece_length=1,
            use_multi_dataset=True,
            dataset_paths={"maestro": str(tmp), "aria_midi": str(tmp)},
            dataset_weights={"maestro": 3.0, "aria_midi": 1.0},
            dataset_profiles={
                "maestro": {
                    "min_duration_seconds": 0.1,
                    "min_note_count": 1,
                    "min_distinct_pitches": 1,
                    "filter_velocity": False,
                },
                "aria_midi": {
                    "min_duration_seconds": 0.1,
                    "min_note_count": 1,
                    "min_distinct_pitches": 1,
                    "filter_velocity": False,
                },
            },
            min_duration_seconds=0.1,
            quality_filter_velocity=False,
            min_note_count=1,
            min_distinct_pitches=1,
            use_continuous_time=True,
        )
        proc = MultiDatasetPreprocessor(cfg)
        proc.preprocess()

        manifest = json.loads(
            (Path(cfg.processed_path) / "manifest.json").read_text(encoding="utf-8")
        )
        assert isinstance(manifest, list) and len(manifest) > 0
        item = manifest[0]
        onset_path = Path(str(item["onset_times_path"]))
        duration_path = Path(str(item["durations_path"]))
        assert onset_path.exists()
        assert duration_path.exists()

        onset = torch.from_numpy(np.load(onset_path))
        dur = torch.from_numpy(np.load(duration_path))
        assert onset.ndim == 1 and dur.ndim == 1
        assert onset.shape[0] == dur.shape[0]
        assert torch.all(onset[1:] >= onset[:-1])
        assert torch.all(dur > 0)
        _print_ok("Time feature extraction", f"tokens={int(onset.shape[0])}")

        train_cfg = TrainConfig(batch_size=2, device="cpu")
        train_loader, _, _ = create_dataloaders(cfg, train_cfg)
        assert train_loader.sampler is not None
        from torch.utils.data import WeightedRandomSampler

        assert isinstance(train_loader.sampler, WeightedRandomSampler)

        sample = next(iter(train_loader))
        assert isinstance(sample, dict)
        assert "token_ids" in sample
        assert "onset_times" in sample
        assert "durations" in sample
        assert "new_piece" in sample
        _print_ok("Weighted sampler + batch", "token/time fields present")


def _write_fast_ornaments_midi(path: Path) -> None:
    """Create a MIDI with many rapid ornamental notes."""

    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=132.0)
    piano = pretty_midi.Instrument(program=0)

    current_time = 0.0
    for phrase_idx in range(12):
        base_pitch = 72 + (phrase_idx % 3)
        for step in range(24):
            start = current_time + (0.032 * float(step))
            duration = 0.020 + (0.002 * float(step % 2))
            pitch = base_pitch + (step % 6)
            piano.notes.append(
                pretty_midi.Note(
                    velocity=86 - (step % 8),
                    pitch=int(pitch),
                    start=float(start),
                    end=float(start + duration),
                )
            )
        current_time += 0.82

    midi.instruments.append(piano)
    midi.write(str(path))


def _write_long_held_chords_midi(path: Path) -> None:
    """Create a MIDI with sustained chordal texture."""

    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=96.0)
    piano = pretty_midi.Instrument(program=0)

    chord_bank = [
        [48, 52, 55, 60],
        [50, 53, 57, 62],
        [45, 50, 53, 57],
        [47, 50, 55, 59],
    ]

    for block in range(16):
        chord = chord_bank[block % len(chord_bank)]
        start = 2.2 * float(block)
        end = start + 1.9
        for note_pitch in chord:
            piano.notes.append(
                pretty_midi.Note(
                    velocity=72 + (block % 6),
                    pitch=int(note_pitch),
                    start=float(start),
                    end=float(end),
                )
            )

    midi.instruments.append(piano)
    midi.write(str(path))


def _write_silence_gap_phrases_midi(path: Path) -> None:
    """Create a MIDI with explicit silence gaps between phrases."""

    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=112.0)
    piano = pretty_midi.Instrument(program=0)

    phrase_offsets = [0.0, 0.20, 0.45, 0.75, 1.05, 1.35]
    phrase_span = 1.65
    silence_gap = 1.10

    for phrase_idx in range(14):
        phrase_start = float(phrase_idx) * (phrase_span + silence_gap)
        transpose = phrase_idx % 4
        for note_idx, rel in enumerate(phrase_offsets):
            start = phrase_start + rel
            end = start + 0.16 + (0.01 * float(note_idx % 2))
            pitch = 60 + transpose + (2 * (note_idx % 3))
            piano.notes.append(
                pretty_midi.Note(
                    velocity=74 + (note_idx % 10),
                    pitch=int(pitch),
                    start=float(start),
                    end=float(end),
                )
            )

    midi.instruments.append(piano)
    midi.write(str(path))


def _assert_tokenizer_time_feature_invariants(
    tokenizer: PianoTokenizer,
    midi_path: Path,
    dataset_name: str,
    case_name: str,
) -> None:
    """Assert onset/duration invariants and BPE-group onset preservation."""

    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    raw = tokenizer._encode_raw(midi_path, encode_ids=False)
    seq = tokenizer._coerce_tok_sequence(raw)
    events = list(getattr(seq, "events", []) or [])
    event_onsets, _event_durations = tokenizer._extract_event_time_features(
        midi, events
    )

    token_ids, onset_times, durations = tokenizer.encode_with_time_features(midi_path)
    assert len(token_ids) > 0, f"No tokens for {dataset_name}/{case_name}."
    assert len(token_ids) == len(onset_times) == len(durations), (
        f"Length mismatch for {dataset_name}/{case_name}: "
        f"ids={len(token_ids)}, onsets={len(onset_times)}, durations={len(durations)}"
    )

    onset_arr = np.asarray(onset_times, dtype=np.float64)
    dur_arr = np.asarray(durations, dtype=np.float64)
    assert np.all(onset_arr[1:] >= onset_arr[:-1]), (
        f"Onsets are not monotonic for {dataset_name}/{case_name}."
    )
    assert np.all(dur_arr > 0.0), (
        f"Durations contain non-positive values for {dataset_name}/{case_name}."
    )

    group_lengths = tokenizer._bpe_group_lengths_from_ids(token_ids)
    assert group_lengths is not None, (
        f"BPE group inference unavailable for {dataset_name}/{case_name}."
    )
    assert len(group_lengths) == len(token_ids), (
        f"BPE group length mismatch for {dataset_name}/{case_name}."
    )

    compressed_groups = 0
    cursor = 0
    for idx, group_len in enumerate(group_lengths):
        safe_len = max(1, int(group_len))
        event_idx = min(max(0, cursor), max(0, len(event_onsets) - 1))
        expected = float(event_onsets[event_idx])
        observed = float(onset_times[idx])
        assert abs(observed - expected) <= 1e-6, (
            "BPE first-token onset mismatch for "
            f"{dataset_name}/{case_name} at group {idx}: "
            f"expected={expected:.6f}, observed={observed:.6f}"
        )
        if safe_len > 1:
            compressed_groups += 1
        cursor += safe_len

    assert compressed_groups > 0, (
        f"No BPE compression groups detected for {dataset_name}/{case_name}."
    )


def test_tokenizer_time_feature_invariants() -> None:
    """Validate tokenizer time invariants across edge cases and dataset labels."""

    print("Testing tokenizer time feature invariants across datasets...")

    dataset_names = ("maestro", "giant_midi", "aria_midi", "adl_piano")
    case_builders = {
        "fast_ornaments": _write_fast_ornaments_midi,
        "long_held_chords": _write_long_held_chords_midi,
        "silence_gaps": _write_silence_gap_phrases_midi,
    }

    checked = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        for dataset_name in dataset_names:
            dataset_dir = tmp / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            midi_paths: list[tuple[str, Path]] = []
            for case_name, build_case in case_builders.items():
                path = dataset_dir / f"{case_name}.mid"
                build_case(path)
                midi_paths.append((case_name, path))

            tokenizer = PianoTokenizer(strategy="remi")
            tokenizer.train(
                midi_paths=[path for _case, path in midi_paths],
                vocab_size=800,
            )

            for case_name, midi_path in midi_paths:
                _assert_tokenizer_time_feature_invariants(
                    tokenizer=tokenizer,
                    midi_path=midi_path,
                    dataset_name=dataset_name,
                    case_name=case_name,
                )
                checked += 1

    _print_ok(
        "Tokenizer invariants",
        f"datasets={len(dataset_names)}, edge_cases={len(case_builders)}, checks={checked}",
    )


if __name__ == "__main__":
    print("=" * 56)
    print("Model v3 smoke test")
    print("=" * 56)
    test_alibi_bias()
    test_attention_block()
    test_hybrid_forward()
    test_cfc_generation_state_threading()
    test_generation_health_random_weights()
    test_checkpoint_rotation_policy()
    test_low_disk_emergency_rotation_trigger()
    test_v2_model()
    test_model_factory_switch()
    test_tokenizer_time_feature_invariants()
    test_time_features_and_weighted_sampling()
    test_v1_checkpoint_generation_report()
    test_parameter_counts()
    print("=" * 56)
    print("All smoke tests completed.")
    print("=" * 56)
