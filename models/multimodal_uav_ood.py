"""Fixed-slot multimodal OOD network for UAV recognition and ENU positioning."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from config import DataConfig, ModelConfig
from .eo_encoder import EOEncoder
from .radar_motion_transformer import RadarMotionTransformer
from .rf_encoder import RFEncoder


class MultimodalUAVOODNet(nn.Module):
    """Eight permanent model-id slots; no unordered queries or matching stage."""

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        calibration_init: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if data_cfg.num_slots != model_cfg.num_slots:
            raise ValueError("Data/model slot count must match")
        if model_cfg.eo_backbone != "dinov3_vits16plus":
            raise ValueError(f"Unsupported EO backbone contract: {model_cfg.eo_backbone}")
        self.data_cfg, self.model_cfg = data_cfg, model_cfg
        dim, slots = model_cfg.dim, model_cfg.num_slots
        self.rf = RFEncoder(dim, model_cfg.rf_raw_width, model_cfg.rf_spec_width, model_cfg.heads, model_cfg.rf_node_layers, model_cfg.dropout, slots)
        self.radar = RadarMotionTransformer(
            dim, model_cfg.heads, model_cfg.radar_layers, model_cfg.dropout, slots, calibration_init,
        )
        self.eo = EOEncoder(
            dim=dim,
            dropout_probability=data_cfg.modality_dropout,
            pretrained=model_cfg.eo_pretrained,
            pretrained_path=model_cfg.eo_pretrained_path,
            dinov3_repo_dir=model_cfg.dinov3_repo_dir,
            train_last_blocks=model_cfg.eo_train_last_blocks,
        )
        self.model_embedding = nn.Parameter(torch.randn(1, slots, dim) * 0.02)
        self.allowlist_embedding = nn.Embedding(2, dim)
        self.modality_position = nn.Parameter(torch.randn(1, 3 + slots, dim) * 0.02)
        fusion_layer = nn.TransformerEncoderLayer(dim, model_cfg.heads, dim_feedforward=dim * 4, dropout=model_cfg.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=model_cfg.fusion_layers)
        self.slot_norm = nn.LayerNorm(dim)
        self.fusion_presence = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.fusion_residual = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        self.global_head = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(model_cfg.dropout))
        self.count_head = nn.Linear(dim, 3)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rf = self.rf(batch["raw_iq"], batch["stft"], batch["rf_scalars"], batch["node_present"], batch["node_enu"])
        radar = self.radar(batch["radar_points"], batch["radar_mask"])
        eo_token = self.eo(batch["eo_image"], batch["has_eo"])
        batch_size = batch["raw_iq"].shape[0]
        allow_embedding = self.allowlist_embedding(batch["allowlist_mask"].long())
        slot_tokens = self.model_embedding.expand(batch_size, -1, -1)
        slot_tokens = slot_tokens + rf["rf_model_tokens"] + radar["radar_track_tokens"] + allow_embedding
        context = torch.stack([rf["rf_global"], radar["radar_global"], eo_token], dim=1)
        fused = self.fusion(torch.cat([context, slot_tokens], dim=1) + self.modality_position)
        fused_slots = self.slot_norm(fused[:, 3:])
        fusion_logits = self.fusion_presence(fused_slots).squeeze(-1)
        # Recognition remains RF-led by construction. Radar has only a bounded
        # weak contribution and is not allowed to become a location shortcut.
        presence_logits = (
            rf["rf_presence_logits"]
            + self.model_cfg.radar_presence_scale * radar["radar_presence_logits"]
            + self.model_cfg.fusion_presence_scale * fusion_logits
        )
        # The residual is learned directly from multimodal evidence.  No
        # undocumented metre cap is imposed on its correction.
        position_pred = radar["radar_position"] + self.fusion_residual(fused_slots)
        global_token = self.global_head(torch.cat([rf["rf_global"], radar["radar_global"], eo_token], dim=-1))
        return {
            "presence_logits": presence_logits,
            "position_pred": position_pred,
            "count_logits": self.count_head(global_token),
            "radar_anchor": radar["radar_anchor"],
            "radar_assignment": radar["radar_assignment"],
            "radar_sink_assignment": radar["radar_sink_assignment"],
            "rf_presence_logits": rf["rf_presence_logits"],
            "radar_presence_logits": radar["radar_presence_logits"],
            "calibrated_radar": radar["calibrated_radar"],
        }

    def calibration_state(self) -> dict[str, Any]:
        return self.radar.calibration.export()
