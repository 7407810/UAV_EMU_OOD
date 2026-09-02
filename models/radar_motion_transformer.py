"""Calibrated Radar point association with learned temporal localization."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class RigidRadarCalibration(nn.Module):
    """No-prior learned global rigid calibration.

    There is intentionally no analytic Radar-to-GT fit here.  A full-circle yaw
    and unbounded translation begin at identity and are optimized through the
    joint RF/Radar/location losses.  ``translation_scale_m`` is Radar-input-only
    conditioning, not a position or yaw prior.
    """

    def __init__(self, initialization: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        initialization = dict(initialization or {})
        scale = torch.as_tensor(initialization.get("translation_scale_m", [100.0, 100.0, 100.0]), dtype=torch.float32).flatten()
        if scale.numel() != 3:
            raise ValueError(f"translation_scale_m must contain three values, received {scale.numel()}")
        self.register_buffer("translation_scale_m", scale.clamp_min(1.0))
        # 2*atan maps an unconstrained scalar smoothly to (-pi, pi), avoiding a
        # hand-selected yaw range.  Translation is linear and unbounded.  The
        # former sinh mapping was also unbounded, but its exponential derivative
        # could overflow after a single unstable update and contaminate every
        # fixed slot with NaN/Inf coordinates.
        self.yaw_latent = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.translation_latent = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def _rigid_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        yaw = 2.0 * torch.atan(self.yaw_latent)
        translation = self.translation_scale_m * self.translation_latent
        return yaw, translation

    def forward(self, raw_xyz: torch.Tensor) -> torch.Tensor:
        """Apply the documented global yaw plus translation correction."""
        e, n, u = raw_xyz.unbind(dim=-1)
        yaw, translation = self._rigid_parameters()
        c, s = torch.cos(yaw), torch.sin(yaw)
        ce = c * e - s * n + translation[0]
        cn = s * e + c * n + translation[1]
        cu = u + translation[2]
        return torch.stack([ce, cn, cu], dim=-1)

    def export(self) -> dict[str, Any]:
        yaw, translation = self._rigid_parameters()
        return {
            "mode": "learned_identity_no_prior",
            "yaw_rad": float(yaw.detach().cpu()),
            "tx": float(translation[0].detach().cpu()),
            "ty": float(translation[1].detach().cpu()),
            "tz": float(translation[2].detach().cpu()),
            "yaw_latent": float(self.yaw_latent.detach().cpu()),
            "translation_latent": self.translation_latent.detach().cpu().tolist(),
            "translation_scale_m": self.translation_scale_m.detach().cpu().tolist(),
        }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class RadarMotionTransformer(nn.Module):
    """Point association plus a learned, non-parametric temporal anchor head.

    ``rel_time_s`` is encoded as an observed input.  The model deliberately does
    not impose constant velocity, nearest-neighbour velocity, or a closed-form
    t=0 extrapolation because none is guaranteed by the dataset contract.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        layers: int,
        dropout: float,
        slots: int = 8,
        calibration_init: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.slots = slots
        calibration_init = dict(calibration_init or {})
        self.calibration = RigidRadarCalibration(calibration_init)
        point_scale = torch.as_tensor(
            calibration_init.get("point_scale_m", calibration_init.get("translation_scale_m", [100.0, 100.0, 100.0])),
            dtype=torch.float32,
        ).flatten()
        if point_scale.numel() != 3:
            raise ValueError(f"point_scale_m must contain three values, received {point_scale.numel()}")
        self.register_buffer("point_scale_m", point_scale.clamp_min(1.0))
        self.point_input = nn.Sequential(
            nn.Linear(3, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        encoder = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 4, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.point_transformer = nn.TransformerEncoder(encoder, num_layers=layers)
        self.query = nn.Parameter(torch.randn(1, slots + 1, dim) * 0.02)  # final query is clutter/allowlist sink
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.track_norm = nn.LayerNorm(dim)
        self.track_presence = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.temporal_statistics = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.temporal_anchor = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        self.global_norm = nn.LayerNorm(dim)

    def forward(self, radar_points: torch.Tensor, radar_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_xyz, rel_time = radar_points[..., :3], radar_points[..., 3]
        xyz = self.calibration(raw_xyz)
        center = _masked_mean(xyz, radar_mask)
        relative_xyz = xyz - center.unsqueeze(1)
        # Association consumes only translation-invariant point geometry and
        # observed time, so Radar absolute coordinates cannot become a presence
        # shortcut.  Absolute calibrated coordinates are used only after slot
        # attention to form a localization reference.
        encoded = self.point_input(relative_xyz / self.point_scale_m.view(1, 1, 3))
        encoded = encoded + self.time_embedding((rel_time / 3.0).unsqueeze(-1))
        encoded = self.point_transformer(encoded, src_key_padding_mask=~radar_mask)
        queries = self.query.expand(encoded.shape[0], -1, -1)
        attended, attention = self.cross_attention(
            queries, encoded, encoded, key_padding_mask=~radar_mask,
            need_weights=True, average_attn_weights=False,
        )
        track_tokens = self.track_norm(attended + queries)
        attention = attention.mean(dim=1)  # B, (8 + sink), P
        assignment = attention[:, : self.slots] * radar_mask.unsqueeze(1).to(attention.dtype)
        normalized_assignment = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        reference = torch.einsum("bsp,bpc->bsc", normalized_assignment, xyz)
        mean_time = (normalized_assignment * rel_time.unsqueeze(1)).sum(dim=-1)
        time_std = torch.sqrt(
            (normalized_assignment * (rel_time.unsqueeze(1) - mean_time.unsqueeze(-1)).square()).sum(dim=-1).clamp_min(0.0)
        )
        temporal_token = self.temporal_statistics(torch.stack([mean_time / 3.0, time_std / 3.0], dim=-1))
        # No velocity law and no hard metre cap: the neural temporal head learns
        # whatever t=0 correction is supported by the indexed training data.
        anchor = reference + self.temporal_anchor(track_tokens[:, : self.slots] + temporal_token)
        global_token = self.global_norm(_masked_mean(encoded, radar_mask))
        return {
            "radar_track_tokens": track_tokens[:, : self.slots],
            "radar_sink_token": track_tokens[:, self.slots],
            "radar_global": global_token,
            "radar_presence_logits": self.track_presence(track_tokens[:, : self.slots]).squeeze(-1),
            "radar_anchor": anchor,
            "radar_position": anchor,
            "radar_assignment": assignment,
            "radar_sink_assignment": attention[:, self.slots],
            "calibrated_radar": xyz,
        }
