from __future__ import annotations

import io
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file
from safetensors.torch import load_file as safetensors_load_file


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import ModelConfig
from data.tokenizer import PianoTokenizer
from model.hybrid import PianoHybridModel
from utils.config_compat import normalize_model_config_payload


app = Flask(__name__, template_folder=str(APP_DIR / "templates"))

MODELS_DIR = APP_DIR / "models"
TOKENIZER_PATH = APP_DIR / "tokenizer" / "tokenizer.json"
RUNTIME_DIR = APP_DIR / "runtime"
OUTPUT_DIR = RUNTIME_DIR / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_model_cache: Dict[str, Tuple[PianoHybridModel, Dict[str, Any]]] = {}
_tokenizer_cache: PianoTokenizer | None = None


def _strip_dataparallel_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module.") :]] = value
        else:
            stripped[key] = value
    return stripped


def _resolve_sidecar(model_path: Path) -> Path | None:
    if model_path.suffix == ".pt":
        return model_path if model_path.exists() else None
    if model_path.name.endswith("_model.safetensors"):
        sidecar = model_path.with_name(
            model_path.name.replace("_model.safetensors", "_state.pt")
        )
        if sidecar.exists():
            return sidecar
    latest = model_path.with_name("latest_state.pt")
    if latest.exists():
        return latest
    return None


def _load_model_config_from_sidecar(sidecar_path: Path | None) -> ModelConfig | None:
    if sidecar_path is None:
        return None
    if sidecar_path.suffix != ".pt":
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


def _infer_vocab_size_from_state(state_dict: Dict[str, torch.Tensor]) -> int:
    emb_key = "token_embedding.weight"
    if emb_key in state_dict:
        return int(state_dict[emb_key].shape[0])
    lm_key = "lm_head.weight"
    if lm_key in state_dict:
        return int(state_dict[lm_key].shape[0])
    return 2000


def _load_state_dict(model_path: Path) -> Dict[str, torch.Tensor]:
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


def _load_model(model_name: str) -> Tuple[PianoHybridModel, Dict[str, Any]]:
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found in app/models: {model_name}")

    sidecar = _resolve_sidecar(model_path)
    model_cfg = _load_model_config_from_sidecar(sidecar)

    state_dict = _load_state_dict(model_path)
    if model_cfg is None:
        vocab = _infer_vocab_size_from_state(state_dict)
        model_cfg = ModelConfig(vocab_size=vocab, use_cfc=False)
    model_cfg = ModelConfig(**normalize_model_config_payload(dict(model_cfg.__dict__)))

    model = PianoHybridModel(model_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(torch.device("cpu"))

    meta = {
        "missing_keys": int(len(missing)),
        "unexpected_keys": int(len(unexpected)),
        "checkpoint": str(model_path.name),
    }
    _model_cache[model_name] = (model, meta)
    return model, meta


def _load_tokenizer() -> PianoTokenizer:
    global _tokenizer_cache
    if _tokenizer_cache is not None:
        return _tokenizer_cache
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}. "
            "Place tokenizer/tokenizer.json under app/."
        )
    _tokenizer_cache = PianoTokenizer.load(str(TOKENIZER_PATH))
    return _tokenizer_cache


def _list_models() -> list[str]:
    allowed = {".safetensors", ".pt"}
    ignored_pt_names = {"latest_state.pt", "best_state.pt"}
    models = []
    for p in sorted(MODELS_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in allowed:
            continue
        if p.suffix.lower() == ".pt":
            if p.name in ignored_pt_names or p.name.endswith("_state.pt"):
                continue
        models.append(p.name)
    return models


def _compute_repetition_warning(tokens: list[int]) -> str | None:
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.decode(tokens, out_path)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def api_models():
    return jsonify({"models": _list_models()})


@app.post("/api/generate")
def api_generate():
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
def api_download(file_name: str):
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
