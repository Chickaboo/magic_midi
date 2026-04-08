from __future__ import annotations

import io
import mimetypes
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DataConfig
from generation.generate import GenerationConfig, generate_continuation
from utils import checkpoint_loading as ckpt_utils
from utils.midi_utils import compare_pianorolls, midi_duration, render_midi_audio


app = Flask(__name__, template_folder=str(APP_DIR / "templates"))

MODELS_DIR = APP_DIR / "models"
TOKENIZER_PATH = APP_DIR / "tokenizer" / "tokenizer.json"
TOKENIZER_CUSTOM_PATH = APP_DIR / "tokenizer" / "custom_tokenizer.json"
RUNTIME_DIR = APP_DIR / "runtime"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
MODEL_SEARCH_DIRS = (MODELS_DIR,)
TOKENIZER_SEARCH_PATHS = (TOKENIZER_CUSTOM_PATH, TOKENIZER_PATH)

CPU_THREADS = max(1, int(os.environ.get("IBP_CPU_THREADS", min(8, os.cpu_count() or 1))))
if not torch.cuda.is_available():
    try:
        torch.set_num_threads(CPU_THREADS)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_model_cache: Dict[str, Tuple[Tuple[str, int, int], Any, Dict[str, Any]]] = {}
_tokenizer_cache: Tuple[Tuple[str, int, int], Any, Dict[str, Any]] | None = None


def _path_signature(path: Path) -> Tuple[str, int, int]:
    """Build a lightweight cache key for a file path."""

    stat = path.stat()
    return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))


def _resolve_existing_path(paths: tuple[Path, ...]) -> Path | None:
    """Return the first existing path from a candidate list."""

    for candidate in paths:
        if candidate.exists():
            return candidate
    return None


def _load_model(model_name: str) -> Tuple[Any, Dict[str, Any]]:
    """Load and cache a model by checkpoint filename."""

    model_path = None
    for directory in MODEL_SEARCH_DIRS:
        candidate = directory / model_name
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError(f"Checkpoint not found in app/models: {model_name}")

    signature = _path_signature(model_path)
    cached = _model_cache.get(model_name)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2]

    sidecar = ckpt_utils.resolve_sidecar_path(model_path)
    bundle = ckpt_utils.load_model_from_checkpoint(
        model_path=model_path,
        sidecar_path=sidecar,
        device="cpu",
        strict=True,
    )
    model = bundle.model

    meta = {
        "missing_keys": int(bundle.missing_keys),
        "unexpected_keys": int(bundle.unexpected_keys),
        "checkpoint": str(model_path.name),
        "checkpoint_path": str(model_path),
        "sidecar_path": str(bundle.sidecar_path)
        if bundle.sidecar_path is not None
        else None,
        "model_class": str(bundle.model_class),
        "model_config": dict(bundle.model_config),
        "checkpoint_metadata": dict(bundle.checkpoint_metadata),
    }
    _model_cache[model_name] = (signature, model, meta)
    return model, meta


def _load_tokenizer_for_model(meta: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
    """Load tokenizer from known search paths with cache and checkpoint metadata."""

    global _tokenizer_cache
    checkpoint_metadata = ckpt_utils.coerce_mapping(meta.get("checkpoint_metadata"))
    data_config = ckpt_utils.extract_data_config(checkpoint_metadata)
    tokenizer_path = ckpt_utils.resolve_tokenizer_path(
        data_config=data_config,
        search_paths=TOKENIZER_SEARCH_PATHS,
    )
    signature = _path_signature(tokenizer_path)

    if _tokenizer_cache is not None and _tokenizer_cache[0] == signature:
        return _tokenizer_cache[1], _tokenizer_cache[2]

    tokenizer, tokenizer_meta = ckpt_utils.load_tokenizer(
        tokenizer_path=tokenizer_path,
        data_config=data_config,
    )
    tokenizer_meta.update(
        {
            "tokenizer_path": str(tokenizer_path),
            "data_config": data_config,
        }
    )
    _tokenizer_cache = (signature, tokenizer, tokenizer_meta)
    return tokenizer, tokenizer_meta


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


def _make_request_dir() -> Path:
    """Create a unique output directory for one generation request."""

    request_dir = OUTPUT_DIR / uuid.uuid4().hex
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def _render_artifacts(
    request_dir: Path,
    seed_path: Path,
    output_path: Path,
) -> Dict[str, str]:
    """Render audio previews and pianoroll comparison images for the UI."""

    seed_audio_path = request_dir / "seed_preview.wav"
    output_audio_path = request_dir / "output_preview.wav"
    comparison_path = request_dir / "pianoroll_comparison.png"

    render_midi_audio(seed_path, seed_audio_path)
    render_midi_audio(output_path, output_audio_path)
    compare_pianorolls(seed_path, output_path, save_path=comparison_path)

    return {
        "seed_audio_path": str(seed_audio_path.relative_to(OUTPUT_DIR)),
        "output_audio_path": str(output_audio_path.relative_to(OUTPUT_DIR)),
        "comparison_image_path": str(comparison_path.relative_to(OUTPUT_DIR)),
    }


def _guess_mimetype(file_path: Path) -> str:
    """Guess a reasonable mimetype for generated artifacts."""

    mimetype, _encoding = mimetypes.guess_type(str(file_path))
    if mimetype:
        return mimetype
    if file_path.suffix.lower() in {".mid", ".midi"}:
        return "audio/midi"
    if file_path.suffix.lower() == ".wav":
        return "audio/wav"
    if file_path.suffix.lower() == ".png":
        return "image/png"
    return "application/octet-stream"


def _serve_artifact(file_name: str, *, as_attachment: bool) -> Any:
    """Serve a generated artifact from the runtime output tree."""

    target = OUTPUT_DIR / file_name
    if not target.exists() or not target.is_file():
        return jsonify({"error": "File not found"}), 404
    with target.open("rb") as f:
        data = io.BytesIO(f.read())
    data.seek(0)
    return send_file(
        data,
        mimetype=_guess_mimetype(target),
        as_attachment=as_attachment,
        download_name=target.name,
    )


def _build_local_generation_status() -> Dict[str, Any]:
    """Summarize whether local CPU generation assets are ready."""

    tokenizer_path = _resolve_existing_path(TOKENIZER_SEARCH_PATHS)
    tokenizer_kind = None
    if tokenizer_path is not None:
        tokenizer_kind = ckpt_utils.detect_tokenizer_kind(tokenizer_path, {})

    models = _list_models()
    ready = bool(models and tokenizer_path is not None)
    if ready:
        guidance = (
            "Local checkpoint and tokenizer are present. Generation will run on CPU and the UI will render seed/output audio plus a pianoroll preview."
        )
    else:
        guidance = (
            "Copy the latest checkpoint into app/models and the tokenizer JSON into app/tokenizer, then refresh the page."
        )
    return {
        "ready": ready,
        "device": "cpu",
        "cpu_threads": int(CPU_THREADS),
        "app_dir": str(APP_DIR),
        "model_search_dirs": [str(p) for p in MODEL_SEARCH_DIRS],
        "tokenizer_search_paths": [str(p) for p in TOKENIZER_SEARCH_PATHS],
        "models": models,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "tokenizer_kind": tokenizer_kind,
        "audio_backend": "pretty_midi.synthesize",
        "pianoroll_backend": "matplotlib",
        "guidance": guidance,
    }


@app.get("/")
def index() -> str:
    """Serve web UI homepage."""

    return render_template("index.html")


@app.get("/api/models")
def api_models() -> Any:
    """Return list of available checkpoints for UI selection."""

    return jsonify({"models": _list_models()})


@app.get("/api/status")
def api_status() -> Any:
    """Return a quick assessment of the local CPU generation setup."""

    return jsonify(_build_local_generation_status())


@app.get("/api/artifact/<path:file_name>")
def api_artifact(file_name: str) -> Any:
    """Serve a generated artifact for preview in the browser."""

    return _serve_artifact(file_name, as_attachment=False)


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
        tokenizer, tokenizer_meta = _load_tokenizer_for_model(meta)

        request_dir = _make_request_dir()
        seed_tmp = request_dir / f"seed_{uuid.uuid4().hex}.mid"
        seed_file.save(seed_tmp)

        if hasattr(tokenizer, "encode_with_time_features"):
            try:
                seed_tokens, seed_onset_times, _durations = tokenizer.encode_with_time_features(
                    seed_tmp
                )
            except Exception:
                seed_tokens = tokenizer.encode(seed_tmp)
                seed_onset_times = None
        else:
            seed_tokens = tokenizer.encode(seed_tmp)
            seed_onset_times = None

        if not seed_tokens:
            return jsonify({"error": "Tokenizer produced no tokens for seed file"}), 400

        data_cfg_payload = ckpt_utils.extract_data_config(
            ckpt_utils.coerce_mapping(meta.get("checkpoint_metadata", {}))
        )
        data_cfg = DataConfig(**data_cfg_payload)

        supports_time = bool(getattr(getattr(model, "config", None), "use_v2_architecture", False))
        gen_cfg = GenerationConfig(
            max_new_tokens=length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,
            repetition_window=64,
            min_tokens_to_keep=3,
            num_samples=1,
        )

        generation_started = time.perf_counter()
        generated_paths = generate_continuation(
            model=model,
            tokenizer=tokenizer,
            seed_midi_path=seed_tmp,
            output_path=request_dir / f"continuation_{uuid.uuid4().hex}.mid",
            config=data_cfg,
            generation_config=gen_cfg,
        )
        generation_elapsed = time.perf_counter() - generation_started
        generated_path = generated_paths[0]

        new_tokens = []
        try:
            generated_tokens = tokenizer.encode(generated_path)
            new_tokens = generated_tokens[len(seed_tokens) :]
        except Exception:
            generated_tokens = []
        warning = _compute_repetition_warning(new_tokens)

        health_check = getattr(model, "generation_health_check", None)
        health_report = None
        if callable(health_check):
            if supports_time and seed_onset_times is not None:
                health_report = health_check(
                    seed_tokens=seed_tokens,
                    seed_onset_times=torch.tensor(seed_onset_times, dtype=torch.float32),
                    steps=min(20, max(1, int(length))),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=1.1,
                    repetition_window=64,
                    min_tokens_to_keep=3,
                    top1_threshold=0.95,
                    raise_on_failure=False,
                    step_seconds=float(
                        max(
                            1e-4,
                            float(
                                data_cfg_payload.get(
                                    "time_feature_fallback_step_seconds", 0.1
                                )
                            ),
                        )
                    ),
                )
            else:
                health_report = health_check(
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

        rendered_paths = _render_artifacts(request_dir, seed_tmp, generated_path)

        seed_duration = midi_duration(seed_tmp)
        output_duration = midi_duration(generated_path)

        payload = {
            "ok": True,
            "download_url": f"/api/artifact/{generated_path.relative_to(OUTPUT_DIR)}",
            "seed_audio_url": f"/api/artifact/{rendered_paths['seed_audio_path']}",
            "output_audio_url": f"/api/artifact/{rendered_paths['output_audio_path']}",
            "comparison_image_url": f"/api/artifact/{rendered_paths['comparison_image_path']}",
            "generated_tokens": int(len(new_tokens)),
            "generation_stats": dict(getattr(model, "last_generation_stats", {})),
            "health_report": health_report,
            "health_warning": warning,
            "checkpoint_meta": meta,
            "tokenizer_meta": tokenizer_meta,
            "local_generation_status": _build_local_generation_status(),
            "seed_duration": float(seed_duration),
            "output_duration": float(output_duration),
            "generation_elapsed_seconds": float(generation_elapsed),
        }
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/download/<path:file_name>")
def api_download(file_name: str) -> Any:
    """Download generated MIDI artifact by file name."""

    return _serve_artifact(file_name, as_attachment=True)


if __name__ == "__main__":
    host = os.environ.get("IBP_HOST", "127.0.0.1")
    port = int(os.environ.get("IBP_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
