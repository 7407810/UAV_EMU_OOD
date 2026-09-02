"""Documented-task losses for fixed-slot classification and ENU localization."""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from config import LossConfig


class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg: float = 3.0, gamma_pos: float = 0.0, clip: float = 0.05) -> None:
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip = gamma_neg, gamma_pos, clip

    def forward(self, logits: torch.Tensor, target: torch.Tensor, positive_weights: torch.Tensor | None = None) -> torch.Tensor:
        positive = torch.sigmoid(logits)
        negative = 1.0 - positive
        if self.clip > 0:
            negative = (negative + self.clip).clamp(max=1.0)
        loss = target * torch.log(positive.clamp_min(1e-8)) + (1.0 - target) * torch.log(negative.clamp_min(1e-8))
        gamma = self.gamma_pos * target + self.gamma_neg * (1.0 - target)
        weight = (1.0 - positive * target - negative * (1.0 - target)).pow(gamma)
        if positive_weights is not None:
            balance = target * positive_weights.view(1, -1) + (1.0 - target)
            return -(loss * weight * balance).sum() / balance.sum().clamp_min(1.0)
        return -(loss * weight).mean()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _location_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    presence: torch.Tensor,
    enu_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = presence > 0
    difference = prediction - target
    normalized = difference / enu_std.view(1, 1, 3)
    smooth = F.smooth_l1_loss(normalized, torch.zeros_like(normalized), reduction="none").mean(dim=-1)
    distance = torch.linalg.vector_norm(difference, dim=-1)
    # Physical log-distance keeps optimization sensitive to large OOD errors.
    value = smooth + 0.10 * torch.log1p(distance)
    return _masked_mean(value, mask), _masked_mean(distance, mask)


class UAVLoss(torch.nn.Module):
    def __init__(self, loss_cfg: LossConfig, fold_stats: Mapping[str, Any]) -> None:
        super().__init__()
        self.cfg = loss_cfg
        self.presence = AsymmetricLoss(loss_cfg.focal_gamma_neg, loss_cfg.focal_gamma_pos, loss_cfg.focal_clip)
        self.register_buffer("enu_std", torch.tensor(fold_stats["enu_std"], dtype=torch.float32))
        self.register_buffer("class_positive_weights", torch.tensor(fold_stats.get("class_positive_weights", [1.0] * 8), dtype=torch.float32))

    def forward(self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        presence_target = batch["presence_target"]
        presence_loss = self.presence(outputs["presence_logits"], presence_target, self.class_positive_weights)
        location_loss, mean_distance = _location_loss(
            outputs["position_pred"], batch["position_target"], presence_target, self.enu_std,
        )
        anchor_loss, _ = _location_loss(
            outputs["radar_anchor"], batch["position_target"], presence_target, self.enu_std,
        )
        count_loss = F.cross_entropy(outputs["count_logits"], batch["count_target"], ignore_index=-100)
        total = presence_loss + location_loss + self.cfg.count_weight * count_loss + self.cfg.anchor_weight * anchor_loss
        logs = {
            "loss": float(total.detach()), "presence": float(presence_loss.detach()), "location": float(location_loss.detach()),
            "count": float(count_loss.detach()), "anchor": float(anchor_loss.detach()),
            "train_3d_mean": float(mean_distance.detach()),
        }
        return total, logs
