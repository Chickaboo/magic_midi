from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file as safetensors_load_file

from config import DataConfig, ModelConfig
from data.tokenizer import PianoTokenizer
from data.tokenizer_custom import CustomDeltaTokenizer
from generation.generate import GenerationConfig, generate_continuation
from model.factory import build_model
from utils.midi_utils import compare_pianorolls, render_midi_audio
from utils.config_compat import normalize_model_config_payload


ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
MODELS_DIR = APP_DIR / "models"
TOKENIZER_DIR = APP_DIR / "tokenizer"
RUNTIME_DIR = APP_DIR / "runtime"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
STATE_PATH = MODELS_DIR / "latest_state.pt"
TOKENIZER_PATHS = [
    TOKENIZER_DIR / "custom_tokenizer.json",
    TOKENIZER_DIR / "tokenizer.json",
]

MAESTRO_ROOT = Path(r"C:\Users\Lucas\Downloads\maestro-v3.0.0-midi\maestro-v3.0.0")


def _env_int(name: str, default: int) -> int:
    """Read an integer override from the environment."""

    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    """Read a float override from the environment."""

    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _resolve_seed_path() -> Path:
    """Resolve a seed MIDI from env override or app runtime defaults."""

    seed_override = os.environ.get("IBP_SEED_PATH", "").strip()
    if seed_override:
        seed_path = Path(seed_override).expanduser()
        if not seed_path.exists():
            raise FileNotFoundError(f"IBP_SEED_PATH does not exist: {seed_path}")
        return seed_path

    candidates = sorted(RUNTIME_DIR.glob("seed_*.mid")) + sorted(
        RUNTIME_DIR.glob("seed_*.midi")
    )
    if not candidates:
        candidates = sorted(RUNTIME_DIR.glob("*.mid")) + sorted(
            RUNTIME_DIR.glob("*.midi")
        )

    if not candidates:
        raise FileNotFoundError(
            "No seed MIDI found in app/runtime. Add a .mid/.midi file there or set IBP_SEED_PATH."
        )
    return candidates[0]


def build_tokenizer_if_needed(data_cfg: DataConfig) -> PianoTokenizer | CustomDeltaTokenizer:
    """Load existing tokenizer from app/tokenizer, with optional fallback training."""

    for tokenizer_path in TOKENIZER_PATHS:
        if tokenizer_path.exists():
            print(f"Loading tokenizer from {tokenizer_path}")
            if str(getattr(data_cfg, "tokenization_strategy", "")).strip().lower() == "custom_delta":
                return CustomDeltaTokenizer.load(str(tokenizer_path))
            try:
                payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if str(payload.get("type", "")).strip() == "CustomDeltaTokenizer":
                return CustomDeltaTokenizer.load(str(tokenizer_path))
            return PianoTokenizer.load(str(tokenizer_path))

    midi_paths = sorted(
        list(MAESTRO_ROOT.rglob("*.midi")) + list(MAESTRO_ROOT.rglob("*.mid")),
        key=lambda p: str(p),
    )
    if not midi_paths:
        raise FileNotFoundError(f"No MIDI files found under {MAESTRO_ROOT}")

    print(f"Training tokenizer on {len(midi_paths)} MIDI files...")
    tokenizer = PianoTokenizer()
    tokenizer.train(midi_paths=midi_paths, vocab_size=2000)
    TOKENIZER_PATHS[0].parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_PATHS[-1]))
    print(f"Saved tokenizer to {TOKENIZER_PATHS[-1]}")
    return tokenizer


def main() -> None:
    """Run one local generation pass from app-stored checkpoint/tokenizer."""

    checkpoint_candidates = [
        MODELS_DIR / "latest.safetensors",
        MODELS_DIR / "latest_model.safetensors",
    ]
    checkpoint_path = next((p for p in checkpoint_candidates if p.exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "Checkpoint not found. Expected latest.safetensors or latest_model.safetensors"
        )
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"State file not found: {STATE_PATH}")

    seed_path = _resolve_seed_path()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{seed_path.stem}_continuation.mid"

    state = torch.load(STATE_PATH, map_location="cpu")
    model_cfg = ModelConfig(
        **normalize_model_config_payload(dict(state["model_config"]))
    )
    data_cfg = DataConfig(**state["data_config"])
    tokenizer = build_tokenizer_if_needed(data_cfg)
    model_cfg.vocab_size = tokenizer.vocab_size
    data_cfg.vocab_size = tokenizer.vocab_size
    data_cfg.tokenizer_path = str(next((p for p in TOKENIZER_PATHS if p.exists()), TOKENIZER_PATHS[-1]))
    data_cfg.maestro_path = str(MAESTRO_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(model_cfg)
    model_state = safetensors_load_file(str(checkpoint_path), device="cpu")
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()

    max_new_tokens = _env_int(
        "IBP_MAX_NEW_TOKENS",
        2048 if device.type == "cpu" else 8192,
    )
    max_new_tokens = max(1, max_new_tokens)
    num_samples = max(1, _env_int("IBP_NUM_SAMPLES", 1))
    temperature = max(0.1, _env_float("IBP_TEMPERATURE", 0.9))
    top_p = min(1.0, max(0.0, _env_float("IBP_TOP_P", 0.95)))
    top_k = max(3, _env_int("IBP_TOP_K", 50))
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.1,
        repetition_window=64,
        min_tokens_to_keep=3,
        num_samples=num_samples,
    )

    print(f"Generating from seed: {seed_path}")
    print(f"Output: {output_path}")
    print(f"Max new tokens: {max_new_tokens}")

    outputs = generate_continuation(
        model=model,
        tokenizer=tokenizer,
        seed_midi_path=seed_path,
        output_path=output_path,
        config=data_cfg,
        generation_config=gen_cfg,
    )

    seed_audio = output_path.with_name(f"{output_path.stem}_seed_preview.wav")
    output_audio = output_path.with_name(f"{output_path.stem}_output_preview.wav")
    comparison_png = output_path.with_name(f"{output_path.stem}_comparison.png")

    render_midi_audio(seed_path, seed_audio)
    render_midi_audio(outputs[0], output_audio)
    compare_pianorolls(seed_path, outputs[0], save_path=comparison_png)

    print(f"Done: {outputs[0]}")
    print(f"Seed audio: {seed_audio}")
    print(f"Output audio: {output_audio}")
    print(f"Comparison: {comparison_png}")


if __name__ == "__main__":
    main()
