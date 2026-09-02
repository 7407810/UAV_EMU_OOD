"""Raw Radar point Transformer with no hand calibration or motion equation."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


class RadarPointTransformer(nn.Module):
    """Encodes the supplied ``[E,N,U,rel_time_s]`` points directly.

    Coordinate standardization is fitted on the current training fold solely for
    numerical conditioning. The network receives the four normalized raw values
    through a learned embedding and is responsible for learning any systematic
    relationship to label ENU from end-to-end supervision. There is no yaw,
    translation, centering, velocity, track assignment, or physical anchor.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        layers: int,
        dropout: float,
        fold_stats: Mapping[str, object],
        normalized_noise_std: float = 0.01,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(fold_stats["radar_mean"], dtype=torch.float32).flatten()
        std = torch.as_tensor(fold_stats["radar_std"], dtype=torch.float32).flatten()
        if mean.numel() != 4 or std.numel() != 4:
            raise ValueError("fold_stats radar_mean/radar_std must each contain four values")
        self.register_buffer("radar_mean", mean)
        self.register_buffer("radar_std", std.clamp_min(1.0e-6))
        self.normalized_noise_std = float(normalized_noise_std)
        self.point_input = nn.Sequential(
            nn.Linear(4, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=dim * 4, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, points: torch.Tensor, point_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] != 4:
            raise ValueError(f"Radar points must be [B,N,4], got {tuple(points.shape)}")
        normalized = (points - self.radar_mean.view(1, 1, 4)) / self.radar_std.view(1, 1, 4)
        if self.training and self.normalized_noise_std > 0.0:
            normalized = normalized + torch.randn_like(normalized) * self.normalized_noise_std
        point_tokens = self.point_input(normalized)
        batch_size = point_tokens.shape[0]
        tokens = torch.cat([self.cls.expand(batch_size, -1, -1), point_tokens], dim=1)
        key_padding_mask = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=points.device),
            ~point_mask.bool(),
        ], dim=1)
        encoded = self.output_norm(self.encoder(tokens, src_key_padding_mask=key_padding_mask))
        return {"tokens": encoded, "key_padding_mask": key_padding_mask}
