from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DataConfig:
    """Data pipeline configuration.

    Attributes:
        maestro_path: Default MAESTRO root for single-dataset training.
        tokenizer_path: Path to tokenizer JSON file.
        processed_path: Directory where tokenized arrays and manifest are saved.
        vocab_size: Target tokenizer vocabulary size.
        tokenization_strategy: Unified tokenization backend ("custom_delta").
        seed_length: Prefix length fed to the model.
        continuation_length: Number of target tokens after the seed.
        stride: Sliding-window stride used by preprocessing helpers.
        min_piece_length: Minimum token count to keep a piece.
        max_sequence_length: Maximum context length accepted by the model.
        max_pieces: Optional cap on number of pieces used from manifest.
        dataset_paths: Optional named dataset roots for multi-dataset runs.
        dataset_weights: Per-dataset sampling weights for training.
        dataset_profiles: Optional per-dataset filter overrides.
        use_multi_dataset: Enable multi-dataset preprocessing and sampling.
        quality_filter_velocity: Filter out low-variance velocity pieces.
        min_duration_seconds: Minimum piece duration in seconds.
        min_note_count: Minimum note count required for one piece.
        min_distinct_pitches: Minimum distinct pitch classes per piece.
        piano_dominance_threshold: Required piano-note fraction in mixed files.
        use_continuous_time: Load onset/duration arrays for v2 model.
        time_feature_fallback_step_seconds: Fallback delta for missing time arrays.
    """

    maestro_path: str = "maestro-v3.0.0"
    tokenizer_path: str = "tokenizer.json"
    processed_path: str = "processed/"
    vocab_size: int = 374
    tokenization_strategy: str = "custom_delta"
    seed_length: int = 256
    continuation_length: int = 768
    stride: int = 128
    min_piece_length: int = 1200
    max_sequence_length: int = 1024
    max_pieces: Optional[int] = None
    dataset_paths: Dict[str, str] = field(default_factory=dict)
    dataset_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "maestro": 1.5,
            "giant_midi": 1.2,
            "aria_midi": 1.0,
            "adl_piano": 1.3,
        }
    )
    dataset_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    use_multi_dataset: bool = False
    quality_filter_velocity: bool = True
    min_duration_seconds: float = 30.0
    min_note_count: int = 100
    min_distinct_pitches: int = 12
    piano_dominance_threshold: float = 0.70
    use_continuous_time: bool = False
    time_feature_fallback_step_seconds: float = 0.5


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        vocab_size: Token vocabulary size.
        d_model: Main hidden size.
        n_layers: Number of Mamba/CfC stacked layers.
        d_state: Mamba state size.
        d_conv: Mamba convolution kernel size.
        expand: Mamba expansion factor.
        cfc_units: CfC hidden units (v3 expects this to match d_model).
        cfc_backbone_units: CfC backbone hidden units.
        cfc_backbone_layers: CfC backbone depth.
        cfc_every_n_layers: Insert CfC every n Mamba layers.
        num_attention_heads: Sparse attention head count.
        attention_every_n_layers: Insert attention every n layers.
        attention_dropout: Sparse attention dropout.
        use_relative_attention: Enable relative position bias in attention.
        attention_bias_type: Relative bias type ("alibi" or "learned").
        max_relative_distance: Relative-bias distance cap for learned bias.
        use_absolute_positions: Add absolute position embeddings.
        dropout: Residual/dropout probability.
        use_cfc: Enable CfC block.
        use_mamba: Enable Mamba block.
        ffn_expansion: FFN expansion ratio for fallback mode.
        tie_embeddings: Tie output projection to token embeddings.
        embedding_init_std: Embedding initialization standard deviation.
        output_logit_scale: Optional manual output logit scale.
        max_sequence_length: Maximum sequence length used by embeddings.
        use_v2_architecture: Enable the v2 dual-stream architecture.
        use_continuous_time_encoding: Add continuous-time onset encoding.
        max_time_seconds: Maximum encoded absolute time in seconds.
        stream_dim: Optional per-stream width for dual-stream model.
        harmonic_ratio: Fraction of stream capacity allocated to harmonic stream.
        cross_stream_every_n_layers: Cross-stream attention cadence.
        cross_attention_every_n: Alias for cross-stream cadence.
        tokens_per_phrase: Phrase summarizer grouping window.
        phrase_dim: Optional phrase representation width.
        memory_size: Episodic memory capacity in phrase slots.
        theme_memory_heads: Number of heads for memory attention.
        debug: Enable extra runtime assertions.
    """

    vocab_size: int = 374
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    cfc_units: int = 256
    cfc_backbone_units: int = 128
    cfc_backbone_layers: int = 2
    cfc_every_n_layers: int = 1
    num_attention_heads: int = 4
    attention_every_n_layers: int = 2
    attention_dropout: float = 0.1
    use_relative_attention: bool = True
    attention_bias_type: str = "alibi"
    max_relative_distance: int = 128
    use_absolute_positions: bool = True
    dropout: float = 0.1
    use_cfc: bool = True
    use_mamba: bool = True
    ffn_expansion: int = 4
    tie_embeddings: bool = True
    embedding_init_std: float = 0.02
    output_logit_scale: Optional[float] = None
    max_sequence_length: int = 1024
    use_v2_architecture: bool = False
    use_continuous_time_encoding: bool = False
    max_time_seconds: float = 600.0
    stream_dim: Optional[int] = None
    harmonic_ratio: float = 0.5
    cross_stream_every_n_layers: int = 2
    cross_attention_every_n: Optional[int] = None
    tokens_per_phrase: int = 16
    phrase_dim: Optional[int] = None
    memory_size: int = 64
    theme_memory_heads: int = 4
    debug: bool = False


@dataclass
class TrainConfig:
    """Training hyperparameters and runtime behavior.

    Attributes:
        batch_size: Mini-batch size per optimization step.
        grad_accumulation_steps: Gradient accumulation factor.
        learning_rate: Initial optimizer learning rate.
        lr_schedule: Learning-rate schedule identifier.
        min_lr_ratio: Minimum LR ratio for cosine floor.
        weight_decay: AdamW weight decay.
        label_smoothing: Cross-entropy label smoothing coefficient.
        max_epochs: Total number of epochs.
        warmup_steps: LR warmup steps.
        max_grad_norm: Gradient clipping threshold.
        generation_health_steps: Validation generation health steps.
        generation_health_top1_threshold: Max allowed top-1 generation confidence.
        generation_health_temperature: Temperature for health-check sampling.
        generation_health_top_p: Top-p for health-check sampling.
        generation_health_top_k: Top-k for health-check sampling.
        generation_health_repetition_penalty: Repetition penalty for health checks.
        generation_health_min_tokens_to_keep: Minimum candidate set size.
        save_every_n_steps: Save latest checkpoint every N optimizer steps (0 disables).
        save_every_n_epochs: Tagged checkpoint cadence.
        keep_every_n_epochs: Milestone checkpoint cadence.
        max_checkpoints: Hard cap on `.safetensors` checkpoint files.
        theme_memory_reset_on_piece: Reset v2 memory at piece boundaries.
        use_amp: Enable mixed precision on CUDA.
        val_generation_check: Enable generation health checks in validation.
        val_generation_steps: Number of generation steps for health checks.
        checkpoint_dir: Output directory for checkpoints.
        use_wandb: Enable Weights & Biases logging.
        seed: Random seed.
        device: Device selector (`auto`, `cpu`, or `cuda`).
    """

    batch_size: int = 8
    grad_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    max_epochs: int = 50
    warmup_steps: int = 500
    max_grad_norm: float = 1.5
    generation_health_steps: int = 20
    generation_health_top1_threshold: float = 0.95
    generation_health_temperature: float = 0.9
    generation_health_top_p: float = 0.95
    generation_health_top_k: int = 50
    generation_health_repetition_penalty: float = 1.1
    generation_health_min_tokens_to_keep: int = 3
    save_every_n_steps: int = 0
    save_every_n_epochs: int = 10
    keep_every_n_epochs: int = 25
    max_checkpoints: int = 8
    theme_memory_reset_on_piece: bool = True
    use_amp: bool = True
    val_generation_check: bool = True
    val_generation_steps: int = 20
    checkpoint_dir: str = "checkpoints/"
    use_wandb: bool = False
    seed: int = 42
    device: str = "auto"
