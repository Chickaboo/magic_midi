from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file as safetensors_load_file

from config import DataConfig, ModelConfig
from data.tokenizer import PianoTokenizer
from generation.generate import GenerationConfig, generate_continuation
from model.hybrid import PianoHybridModel
from utils.config_compat import normalize_model_config_payload


ROOT = Path(r"C:\Users\Lucas\Downloads\midi_AI\piano_midi_model")
LOCAL_DRIVE = ROOT / "local_drive" / "piano_model"
CHECKPOINT = LOCAL_DRIVE / "checkpoints" / "latest_model.safetensors"
STATE_PATH = LOCAL_DRIVE / "checkpoints" / "latest_state.pt"
SEED_PATH = LOCAL_DRIVE / "generated" / "distinct_seed_10s.mid"
TOKENIZER_PATH = LOCAL_DRIVE / "tokenizer" / "tokenizer.json"
OUTPUT_PATH = LOCAL_DRIVE / "generated" / "distinct_seed_10s_continuation.mid"

MAESTRO_ROOT = Path(r"C:\Users\Lucas\Downloads\maestro-v3.0.0-midi\maestro-v3.0.0")


def build_tokenizer_if_needed() -> PianoTokenizer:
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TOKENIZER_PATH.exists():
        print(f"Loading tokenizer from {TOKENIZER_PATH}")
        return PianoTokenizer.load(str(TOKENIZER_PATH))

    midi_paths = sorted(
        list(MAESTRO_ROOT.rglob("*.midi")) + list(MAESTRO_ROOT.rglob("*.mid")),
        key=lambda p: str(p),
    )
    if not midi_paths:
        raise FileNotFoundError(f"No MIDI files found under {MAESTRO_ROOT}")

    print(f"Training tokenizer on {len(midi_paths)} MIDI files...")
    tokenizer = PianoTokenizer()
    tokenizer.train(midi_paths=midi_paths, vocab_size=2000)
    tokenizer.save(str(TOKENIZER_PATH))
    print(f"Saved tokenizer to {TOKENIZER_PATH}")
    return tokenizer


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"State file not found: {STATE_PATH}")
    if not SEED_PATH.exists():
        raise FileNotFoundError(f"Seed file not found: {SEED_PATH}")

    tokenizer = build_tokenizer_if_needed()

    state = torch.load(STATE_PATH, map_location="cpu")
    model_cfg = ModelConfig(
        **normalize_model_config_payload(dict(state["model_config"]))
    )
    data_cfg = DataConfig(**state["data_config"])
    model_cfg.vocab_size = tokenizer.vocab_size
    data_cfg.vocab_size = tokenizer.vocab_size
    data_cfg.tokenizer_path = str(TOKENIZER_PATH)
    data_cfg.maestro_path = str(MAESTRO_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = PianoHybridModel(model_cfg)
    model_state = safetensors_load_file(str(CHECKPOINT), device="cpu")
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()

    max_new_tokens = 2048 if device.type == "cpu" else 8192
    max_new_tokens = max(max_new_tokens, 2048)
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        repetition_window=64,
        min_tokens_to_keep=3,
        num_samples=1,
    )

    print(f"Generating from seed: {SEED_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Max new tokens: {max_new_tokens}")

    outputs = generate_continuation(
        model=model,
        tokenizer=tokenizer,
        seed_midi_path=SEED_PATH,
        output_path=OUTPUT_PATH,
        config=data_cfg,
        generation_config=gen_cfg,
    )

    print(f"Done: {outputs[0]}")


if __name__ == "__main__":
    main()
