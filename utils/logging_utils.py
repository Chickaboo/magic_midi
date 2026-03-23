from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


PROJECT_LOGGER_NAME = "itty_bitty_piano"


def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Create and configure a logger once.

    Args:
        name: Logger name.
        log_file: Optional file path for mirrored logs.

    Returns:
        Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_project_logger(log_file: str | None = None) -> logging.Logger:
    """Return the shared project logger."""

    return setup_logger(PROJECT_LOGGER_NAME, log_file=log_file)


def log_model_summary(model: Any, config: Any) -> None:
    """Log core model configuration and parameter counts."""

    logger = get_project_logger()
    model_type = model.__class__.__name__
    logger.info("Model type: %s", model_type)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total params: %s", f"{total_params:,}")
    logger.info("Trainable params: %s", f"{trainable_params:,}")

    if hasattr(model, "get_num_params"):
        model.get_num_params()

    mamba_info = "unknown"
    fallback_info = "unknown"
    if hasattr(model, "layers") and len(getattr(model, "layers", [])) > 0:
        first_layer = model.layers[0]
        mamba_module = None
        if hasattr(first_layer, "mamba"):
            mamba_module = getattr(first_layer, "mamba")
        elif isinstance(first_layer, dict) and "mamba" in first_layer:
            mamba_module = first_layer["mamba"]
        elif hasattr(first_layer, "__contains__") and hasattr(
            first_layer, "__getitem__"
        ):
            try:
                if "mamba" in first_layer:
                    mamba_module = first_layer["mamba"]
            except Exception:
                mamba_module = None

        if mamba_module is not None:
            fallback_info = str(getattr(mamba_module, "using_fallback", "unknown"))
            mamba_info = "GRU fallback" if fallback_info == "True" else "mamba-ssm"

    use_cfc = getattr(config, "use_cfc", getattr(model, "use_cfc", "unknown"))
    use_mamba = getattr(config, "use_mamba", getattr(model, "use_mamba", "unknown"))
    attention_bias = getattr(config, "attention_bias_type", "unknown")
    tied_emb = getattr(config, "tie_embeddings", "unknown")
    logit_scale = getattr(config, "output_logit_scale", "auto")

    logger.info("Mamba backend: %s", mamba_info)
    logger.info("Fallback active: %s", fallback_info)
    logger.info("CfC enabled: %s", use_cfc)
    logger.info("Mamba enabled: %s", use_mamba)
    logger.info("Attention bias: %s", attention_bias)
    logger.info("Tied embeddings: %s", tied_emb)
    logger.info("Output logit scale: %s", logit_scale)
