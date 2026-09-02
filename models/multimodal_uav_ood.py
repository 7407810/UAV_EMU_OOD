"""End-to-end multimodal unordered target-query network."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from config import DataConfig, ModelConfig
from .eo_encoder import EOEncoder
from .radar_motion_transformer import RadarPointTransformer
from .rf_encoder import RFEncoder


class TargetDecoderLayer(nn.Module):
    """Self-attention followed by Radar -> RF -> EO cross-attention."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.radar_norm = nn.LayerNorm(dim)
        self.rf_norm = nn.LayerNorm(dim)
        self.eo_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.radar_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.rf_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.eo_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _cross(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor,
        attention: nn.MultiheadAttention,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        update, _ = attention(norm(query), memory, memory, key_padding_mask=key_padding_mask, need_weights=False)
        return query + self.dropout(update)

    def forward(
        self,
        query: torch.Tensor,
        radar_tokens: torch.Tensor,
        radar_mask: torch.Tensor,
        rf_tokens: torch.Tensor,
        rf_mask: torch.Tensor,
        eo_tokens: torch.Tensor,
        eo_mask: torch.Tensor,
    ) -> torch.Tensor:
        update, _ = self.self_attention(self.self_norm(query), self.self_norm(query), self.self_norm(query), need_weights=False)
        query = query + self.dropout(update)
        query = self._cross(query, radar_tokens, radar_mask, self.radar_attention, self.radar_norm)
        query = self._cross(query, rf_tokens, rf_mask, self.rf_attention, self.rf_norm)
        query = self._cross(query, eo_tokens, eo_mask, self.eo_attention, self.eo_norm)
        return query + self.dropout(self.ffn(self.ffn_norm(query)))


class MultimodalUAVOODNet(nn.Module):
    """Three unordered target queries for model ID, ENU and uncertainty.

    Query ``i`` has no semantic model or position binding. It can represent any
    target on any sample; Hungarian assignment supplies permutation-invariant
    supervision. All Radar-to-current-position behavior is learned by the
    point encoder and cross-modal decoder rather than an explicit motion model.
    """

    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig, fold_stats: Mapping[str, object]) -> None:
        super().__init__()
        if data_cfg.num_queries != 3:
            raise ValueError("The documented task permits at most three targets; num_queries must be 3")
        if data_cfg.num_models != 8:
            raise ValueError("The documented model space has exactly 8 classes")
        if model_cfg.dim % model_cfg.heads:
            raise ValueError(f"model dim {model_cfg.dim} must be divisible by heads {model_cfg.heads}")
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        dim = model_cfg.dim
        self.rf = RFEncoder(
            dim=dim,
            raw_width=model_cfg.rf_raw_width,
            spec_width=model_cfg.rf_spec_width,
            heads=model_cfg.heads,
            layers=model_cfg.rf_node_layers,
            dropout=model_cfg.dropout,
            num_nodes=data_cfg.num_nodes,
        )
        self.radar = RadarPointTransformer(
            dim=dim,
            heads=model_cfg.heads,
            layers=model_cfg.radar_layers,
            dropout=model_cfg.dropout,
            fold_stats=fold_stats,
            normalized_noise_std=model_cfg.radar_normalized_noise_std,
        )
        self.eo = EOEncoder(
            dim=dim,
            dropout_probability=data_cfg.modality_dropout,
            enabled=model_cfg.use_eo,
            pretrained=model_cfg.eo_pretrained,
            pretrained_path=model_cfg.eo_pretrained_path,
            dinov3_repo_dir=model_cfg.dinov3_repo_dir,
            train_last_blocks=model_cfg.eo_train_last_blocks,
        )
        self.target_queries = nn.Parameter(torch.randn(1, data_cfg.num_queries, dim) * 0.02)
        self.decoder = nn.ModuleList([
            TargetDecoderLayer(dim, model_cfg.heads, model_cfg.dropout)
            for _ in range(model_cfg.decoder_layers)
        ])
        self.query_norm = nn.LayerNorm(dim)
        self.objectness_head = nn.Linear(dim, 1)
        self.model_head = nn.Linear(dim, data_cfg.num_models)
        self.position_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        self.log_sigma_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        mean = torch.as_tensor(fold_stats["enu_mean"], dtype=torch.float32).flatten()
        std = torch.as_tensor(fold_stats["enu_std"], dtype=torch.float32).flatten()
        if mean.numel() != 3 or std.numel() != 3:
            raise ValueError("fold_stats enu_mean/enu_std must each contain three values")
        self.register_buffer("enu_mean", mean)
        self.register_buffer("enu_std", std.clamp_min(1.0e-3))

    def normalize_position(self, position: torch.Tensor) -> torch.Tensor:
        return (position - self.enu_mean.view(1, 1, 3)) / self.enu_std.view(1, 1, 3)

    def denormalize_position(self, position: torch.Tensor) -> torch.Tensor:
        return position * self.enu_std.view(1, 1, 3) + self.enu_mean.view(1, 1, 3)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rf = self.rf(batch["raw_iq"], batch["stft"], batch["rf_scalars"], batch["node_present"])
        radar = self.radar(batch["radar_points"], batch["radar_mask"])
        eo = self.eo(batch["eo_image"], batch["has_eo"])
        query = self.target_queries.expand(batch["raw_iq"].shape[0], -1, -1)
        for layer in self.decoder:
            query = layer(
                query,
                radar["tokens"], radar["key_padding_mask"],
                rf["tokens"], rf["key_padding_mask"],
                eo["tokens"], eo["key_padding_mask"],
            )
        query = self.query_norm(query)
        position_mu_norm = self.position_head(query)
        # Bounding uncertainty log-scale avoids degenerate numerical variance;
        # it is not a bound on ENU coordinates or a model-specific range.
        log_sigma_norm = self.log_sigma_head(query).clamp(min=-6.0, max=5.0)
        position_mu = self.denormalize_position(position_mu_norm)
        position_log_sigma = log_sigma_norm + torch.log(self.enu_std).view(1, 1, 3)
        return {
            "objectness_logits": self.objectness_head(query).squeeze(-1),
            "model_logits": self.model_head(query),
            "position_mu_norm": position_mu_norm,
            "position_mu": position_mu,
            "position_log_sigma_norm": log_sigma_norm,
            "position_log_sigma": position_log_sigma,
        }
