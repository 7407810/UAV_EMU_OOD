"""Single-source configuration for the multimodal set-prediction system.

The configuration is serialized in every checkpoint.  Consequently a saved
validation fold and the formal public-test entry point reconstruct precisely
the same IQ/STFT, Radar, EO, normalization and decoder contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DataConfig:
    num_models: int = 8
    num_queries: int = 3
    num_nodes: int = 4

    # RF captures are documented as 10 ms but use different node sampling
    # rates. Raw IQ is resampled to a common physical-time representation;
    # native-rate STFTs retain their own rate-derived axes.
    raw_iq_length: int = 8192
    iq_window_seconds: float = 0.010
    stft_freq_bins: int = 128
    stft_time_bins: int = 128
    stft_window_seconds: tuple[float, ...] = (0.00025, 0.0005, 0.001)
    stft_hop_ratio: float = 0.25

    # Radar is retained as raw [E, N, U, rel_time_s] points. The model sees no
    # handcrafted track, velocity, calibration or point-cloud statistic.
    max_radar_points: int = 256
    radar_point_dropout: float = 0.15

    eo_size: int = 224
    modality_dropout: float = 0.20


@dataclass
class ModelConfig:
    # ``base`` is deliberately Radar/decoder-heavy. ``auto`` resolves this at
    # runtime from available CUDA memory and the resolved values are checkpointed.
    scale: str = "base"
    dim: int = 256
    heads: int = 8
    dropout: float = 0.10

    rf_raw_width: int = 80
    rf_spec_width: int = 80
    rf_node_layers: int = 3
    radar_layers: int = 6
    decoder_layers: int = 4

    # EO is optional; when disabled the network has an explicit learned missing
    # modality token and never tries to download or instantiate DINO.
    use_eo: bool = True
    eo_backbone: str = "dinov3_vits16plus"
    eo_pretrained: bool = True
    eo_pretrained_path: str = ""
    dinov3_repo_dir: str = ""
    eo_train_last_blocks: int = 0

    # Small normalized-coordinate noise is a standard numerical augmentation,
    # not a direction, speed, range, or frame assumption.
    radar_normalized_noise_std: float = 0.01


@dataclass
class OptimConfig:
    epochs: int = 120
    batch_size: int = 8
    eval_batch_size: int = 8
    workers: int = 4
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-2
    warmup_ratio: float = 0.05
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    early_stopping_patience: int = 18
    bf16: bool = True
    fp16: bool = True
    accumulation_steps: int = 1


@dataclass
class LossConfig:
    # Hungarian assignment cost. It uses only a model classification term and
    # fold-normalized position error, exactly matching the supervised targets.
    matching_class_weight: float = 1.0
    matching_position_weight: float = 2.0

    objectness_weight: float = 1.0
    classification_weight: float = 1.0
    location_weight: float = 3.0
    objectness_focal_gamma: float = 2.0

    location_smooth_l1_weight: float = 1.0
    location_log_distance_weight: float = 0.10
    location_gaussian_nll_weight: float = 0.20

    # This is a generic confidence threshold for a set detector, selected once
    # from its semantic objectness/class probabilities. It is not a model,
    # position, signature, allowlist-combination, or trajectory rule.
    decode_confidence_threshold: float = 0.25

    # Higher is better only for early stopping; it is never reported as an
    # official/proxy leaderboard metric.
    selection_f1_weight: float = 0.50
    selection_distance_penalty: float = 0.0015


@dataclass
class CVConfig:
    folds: int = 5
    seed: int = 3407


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    cv: CVConfig = field(default_factory=CVConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ProjectConfig":
        return cls(
            data=DataConfig(**values.get("data", {})),
            model=ModelConfig(**values.get("model", {})),
            optim=OptimConfig(**values.get("optim", {})),
            loss=LossConfig(**values.get("loss", {})),
            cv=CVConfig(**values.get("cv", {})),
        )
