from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    maestro_path: str = "maestro-v3.0.0"
    tokenizer_path: str = "tokenizer.json"
    processed_path: str = "processed/"
    vocab_size: int = 2000
    tokenization_strategy: str = "remi"
    seed_length: int = 256
    continuation_length: int = 768
    stride: int = 128
    min_piece_length: int = 1200
    max_sequence_length: int = 1024
    max_pieces: Optional[int] = None


@dataclass
class ModelConfig:
    vocab_size: int = 2000
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
    debug: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accumulation_steps: int = 4
    learning_rate: float = 3e-4
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
    save_every_n_epochs: int = 5
    keep_every_n_epochs: int = 10
    checkpoint_dir: str = "checkpoints/"
    use_wandb: bool = False
    seed: int = 42
    device: str = "auto"
