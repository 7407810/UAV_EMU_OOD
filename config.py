"""Configuration for the OOD-oriented UAV ENU system.

All preprocessing values live here so validation, parity checking and test
inference always reconstruct exactly the same data path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DataConfig:
    num_slots: int = 8
    num_nodes: int = 4
    raw_iq_length: int = 8192
    # Dataset contract: each valid IQ record is approximately a 10 ms capture,
    # while sampling rates vary by receiver.  All raw inputs are resampled onto
    # this common physical-time grid; native-rate STFTs retain their own Hz axis.
    iq_window_seconds: float = 0.010
    stft_freq_bins: int = 128
    stft_time_bins: int = 128
    stft_window_seconds: tuple[float, ...] = (0.00025, 0.0005, 0.001)
    stft_hop_ratio: float = 0.25
    max_radar_points: int = 256
    eo_size: int = 224
    train_random_radar_sample: bool = True
    modality_dropout: float = 0.20


@dataclass
class ModelConfig:
    num_slots: int = 8
    dim: int = 192
    rf_raw_width: int = 96
    rf_spec_width: int = 96
    rf_node_layers: int = 3
    radar_layers: int = 3
    fusion_layers: int = 2
    heads: int = 6
    dropout: float = 0.10
    radar_presence_scale: float = 0.15
    fusion_presence_scale: float = 0.25
    # EO stays strictly auxiliary. The backbone is a local, deterministic
    # DINOv3 ViT-S+/16 distilled load so validation and formal test inference cannot
    # silently fetch or substitute a different vision model.
    eo_backbone: str = "dinov3_vits16plus"
    eo_pretrained: bool = True
    eo_pretrained_path: str = ""
    dinov3_repo_dir: str = ""
    # EO has no object/track supervision and is deliberately auxiliary.  Keep
    # the large pretrained DINO backbone frozen by default; only its adapter is
    # optimized in the main graph.
    eo_train_last_blocks: int = 0


@dataclass
class OptimConfig:
    epochs: int = 120
    batch_size: int = 8
    eval_batch_size: int = 8
    workers: int = 4
    lr: float = 2.0e-4
    # Three global calibration latents need to settle early enough that EMA sees
    # a stable frame; all other network weights retain the base learning rate.
    calibration_lr_multiplier: float = 3.0
    weight_decay: float = 1.0e-2
    warmup_ratio: float = 0.05
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    early_stopping_patience: int = 18
    bf16: bool = True
    fp16: bool = True
    accumulation_steps: int = 1


@dataclass
class SessionConfig:
    folds: int = 5
    neighbor_sample_gap: int = 8
    neighbor_search_back: int = 10
    continuity_distance_m: float = 42.0
    radar_similarity_z: float = 1.6  # RMS robust-z distance across descriptor dimensions
    max_signature_delta: int = 2
    max_session_samples: int = 96
    max_session_id_span: int = 128
    min_sessions_per_fold: int = 4
    seed: int = 3407


@dataclass
class LossConfig:
    count_weight: float = 0.15
    anchor_weight: float = 0.10
    focal_gamma_neg: float = 3.0
    focal_gamma_pos: float = 0.0
    focal_clip: float = 0.05


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ProjectConfig":
        # Checkpoints contain only primitives, which makes restoring preprocessing
        # settings independent of argparse defaults on another machine.
        return cls(
            data=DataConfig(**values.get("data", {})),
            model=ModelConfig(**values.get("model", {})),
            optim=OptimConfig(**values.get("optim", {})),
            session=SessionConfig(**values.get("session", {})),
            loss=LossConfig(**values.get("loss", {})),
        )
