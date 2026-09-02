"""Interpretable CV metrics only; intentionally no fabricated LocMass proxy."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from decoder import decode_fixed_slots


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.logical_and(prediction, target).sum())
    fp = float(np.logical_and(prediction, ~target).sum())
    fn = float(np.logical_and(~prediction, target).sum())
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return _safe_div(2.0 * precision * recall, precision + recall), precision, recall


def _error_summary(errors: np.ndarray) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    if not len(errors):
        return {"count": 0, "e_mae": float("nan"), "n_mae": float("nan"), "u_mae": float("nan"), "3d_mean": float("nan"), "3d_median": float("nan"), "3d_p75": float("nan"), "3d_p90": float("nan"), "3d_p95": float("nan")}
    distance = np.linalg.norm(errors, axis=1)
    return {
        "count": int(len(errors)), "e_mae": float(np.mean(np.abs(errors[:, 0]))), "n_mae": float(np.mean(np.abs(errors[:, 1]))),
        "u_mae": float(np.mean(np.abs(errors[:, 2]))), "3d_mean": float(np.mean(distance)),
        "3d_median": float(np.median(distance)), "3d_p75": float(np.percentile(distance, 75)),
        "3d_p90": float(np.percentile(distance, 90)), "3d_p95": float(np.percentile(distance, 95)),
    }


def evaluate_predictions(
    presence_logits: np.ndarray,
    count_prob: np.ndarray,
    position_pred: np.ndarray,
    allowlist_mask: np.ndarray,
    target_presence: np.ndarray,
    target_position: np.ndarray,
    radar_nn_distance: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate fixed slots after the exact same deployment decoder used at test."""
    decoded_drones, decoded_mask, masked_prob = decode_fixed_slots(presence_logits, count_prob, position_pred, allowlist_mask)
    target = np.asarray(target_presence, dtype=bool)
    prediction = np.asarray(decoded_mask, dtype=bool)
    micro_f1, micro_precision, micro_recall = _f1(prediction, target)
    model_f1 = [_f1(prediction[:, model], target[:, model])[0] for model in range(target.shape[1])]
    target_count = target.sum(axis=1)
    pred_count = prediction.sum(axis=1)
    # Raw fixed-slot ENU errors for each target isolate localization quality from
    # a classifier miss; detected-only errors are supplied separately for review.
    all_errors, all_models, all_counts, all_samples = [], [], [], []
    matched_errors = []
    for sample in range(len(target)):
        for model in np.flatnonzero(target[sample]):
            error = position_pred[sample, model] - target_position[sample, model]
            all_errors.append(error)
            all_models.append(model)
            all_counts.append(target_count[sample])
            all_samples.append(sample)
            if prediction[sample, model]:
                matched_errors.append(error)
    error_array = np.asarray(all_errors, dtype=np.float64).reshape(-1, 3)
    report: dict[str, Any] = {
        "classification": {
            "micro_f1": micro_f1, "micro_precision": micro_precision, "micro_recall": micro_recall,
            "macro_f1": float(np.mean(model_f1)), "exact_set_accuracy": float(np.mean(np.all(prediction == target, axis=1))),
            "count_accuracy": float(np.mean(pred_count == target_count)), "f1_by_model": {str(model): float(value) for model, value in enumerate(model_f1)},
        },
        "position_all_gt_slots": _error_summary(error_array),
        "position_detected_gt_slots": _error_summary(np.asarray(matched_errors, dtype=np.float64).reshape(-1, 3)),
        "position_by_model": {}, "position_by_target_count": {}, "position_by_radar_nn": {},
    }
    models = np.asarray(all_models)
    counts = np.asarray(all_counts)
    sample_index = np.asarray(all_samples)
    for model in range(8):
        report["position_by_model"][str(model)] = _error_summary(error_array[models == model])
    for count in (1, 2, 3):
        report["position_by_target_count"][str(count)] = _error_summary(error_array[counts == count])
    if radar_nn_distance is not None:
        radar = np.asarray(radar_nn_distance, dtype=float)[sample_index]
        bins = [("le_0.3", -np.inf, 0.3), ("0.3_to_0.8", 0.3, 0.8), ("0.8_to_2", 0.8, 2.0), ("gt_2", 2.0, np.inf)]
        for name, low, high in bins:
            select = (radar <= high) if np.isneginf(low) else ((radar > low) & (radar <= high) if np.isfinite(high) else radar > low)
            report["position_by_radar_nn"][name] = _error_summary(error_array[select])
    extra = {
        "decoded_drones": decoded_drones, "decoded_mask": decoded_mask, "masked_presence_prob": masked_prob,
        "target_count": target_count, "pred_count": pred_count,
    }
    return report, extra


def robust_selection_score(metrics: Mapping[str, Any]) -> float:
    """Higher is better: strict F1, then median and P90 localization robustness."""
    f1 = float(metrics["classification"]["micro_f1"])
    position = metrics["position_all_gt_slots"]
    median, p90 = float(position["3d_median"]), float(position["3d_p90"])
    if not np.isfinite(median) or not np.isfinite(p90):
        return -np.inf
    return f1 - 0.001 * (median + 0.5 * p90)
