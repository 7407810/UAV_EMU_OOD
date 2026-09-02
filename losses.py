"""Hungarian set-prediction objective for unordered UAV targets."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from config import LossConfig


def _hungarian_matches(
    model_logits: torch.Tensor,
    position_mu_norm: torch.Tensor,
    target_model_ids: torch.Tensor,
    target_position_norm: torch.Tensor,
    target_mask: torch.Tensor,
    class_weight: float,
    position_weight: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute detached minimum-cost query/target assignments per sample."""
    log_probability = torch.log_softmax(model_logits.detach().float(), dim=-1)
    matches: list[tuple[np.ndarray, np.ndarray]] = []
    for batch_index in range(model_logits.shape[0]):
        valid_targets = torch.nonzero(target_mask[batch_index], as_tuple=False).flatten()
        if not len(valid_targets):
            matches.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
            continue
        target_models = target_model_ids[batch_index, valid_targets]
        class_cost = -log_probability[batch_index, :, target_models]
        position_cost = torch.cdist(
            position_mu_norm[batch_index].detach().float(), target_position_norm[batch_index, valid_targets].detach().float(), p=1,
        )
        cost = class_weight * class_cost + position_weight * position_cost
        query_indices, target_columns = linear_sum_assignment(cost.cpu().numpy())
        matches.append((query_indices.astype(np.int64), valid_targets[target_columns].detach().cpu().numpy().astype(np.int64)))
    return matches


class SetPredictionLoss(torch.nn.Module):
    """Objectness + matched model CE + uncertainty-aware ENU localization."""

    def __init__(self, cfg: LossConfig, fold_stats: Mapping[str, object]) -> None:
        super().__init__()
        self.cfg = cfg
        mean = torch.as_tensor(fold_stats["enu_mean"], dtype=torch.float32)
        std = torch.as_tensor(fold_stats["enu_std"], dtype=torch.float32)
        self.register_buffer("enu_mean", mean)
        self.register_buffer("enu_std", std.clamp_min(1.0e-3))

    def _target_normalized(self, target_position: torch.Tensor) -> torch.Tensor:
        return (target_position - self.enu_mean.view(1, 1, 3)) / self.enu_std.view(1, 1, 3)

    def forward(self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        target_model_ids = batch["target_model_ids"].long()
        target_position = batch["target_positions"].float()
        target_mask = batch["target_mask"].bool()
        target_position_norm = self._target_normalized(target_position)
        matches = _hungarian_matches(
            outputs["model_logits"], outputs["position_mu_norm"], target_model_ids, target_position_norm, target_mask,
            self.cfg.matching_class_weight, self.cfg.matching_position_weight,
        )

        objectness_target = torch.zeros_like(outputs["objectness_logits"])
        batch_indices: list[torch.Tensor] = []
        query_indices: list[torch.Tensor] = []
        target_indices: list[torch.Tensor] = []
        for batch_index, (query_index, target_index) in enumerate(matches):
            if len(query_index):
                q = torch.as_tensor(query_index, dtype=torch.long, device=objectness_target.device)
                t = torch.as_tensor(target_index, dtype=torch.long, device=objectness_target.device)
                objectness_target[batch_index, q] = 1.0
                batch_indices.append(torch.full_like(q, batch_index))
                query_indices.append(q)
                target_indices.append(t)

        bce = F.binary_cross_entropy_with_logits(outputs["objectness_logits"], objectness_target, reduction="none")
        probability = torch.sigmoid(outputs["objectness_logits"])
        focal_probability = torch.where(objectness_target > 0, probability, 1.0 - probability)
        objectness_loss = (bce * (1.0 - focal_probability).pow(self.cfg.objectness_focal_gamma)).mean()

        if not batch_indices:
            raise RuntimeError("A training batch has no valid targets")
        batch_index = torch.cat(batch_indices)
        query_index = torch.cat(query_indices)
        target_index = torch.cat(target_indices)
        matched_model_logits = outputs["model_logits"][batch_index, query_index]
        matched_model_ids = target_model_ids[batch_index, target_index]
        classification_loss = F.cross_entropy(matched_model_logits, matched_model_ids)

        prediction_norm = outputs["position_mu_norm"][batch_index, query_index]
        target_norm = target_position_norm[batch_index, target_index]
        prediction = outputs["position_mu"][batch_index, query_index]
        target = target_position[batch_index, target_index]
        log_sigma_norm = outputs["position_log_sigma_norm"][batch_index, query_index]
        difference_norm = prediction_norm - target_norm
        smooth_l1 = F.smooth_l1_loss(prediction_norm, target_norm)
        physical_distance = torch.linalg.vector_norm(prediction - target, dim=-1)
        log_distance = torch.log1p(physical_distance).mean()
        inverse_variance = torch.exp(-2.0 * log_sigma_norm)
        gaussian_nll = 0.5 * (difference_norm.square() * inverse_variance + 2.0 * log_sigma_norm).mean()
        location_loss = (
            self.cfg.location_smooth_l1_weight * smooth_l1
            + self.cfg.location_log_distance_weight * log_distance
            + self.cfg.location_gaussian_nll_weight * gaussian_nll
        )
        total = (
            self.cfg.objectness_weight * objectness_loss
            + self.cfg.classification_weight * classification_loss
            + self.cfg.location_weight * location_loss
        )
        logs = {
            "loss": float(total.detach()),
            "objectness": float(objectness_loss.detach()),
            "classification": float(classification_loss.detach()),
            "location": float(location_loss.detach()),
            "location_smooth_l1": float(smooth_l1.detach()),
            "location_log_distance": float(log_distance.detach()),
            "location_gaussian_nll": float(gaussian_nll.detach()),
            "train_3d_mean": float(physical_distance.mean().detach()),
            "matched_targets": float(len(batch_index)),
        }
        return total, logs
