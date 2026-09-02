"""Interpretable set-detection and ENU metrics; no fabricated LocMass proxy."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import LossConfig
from decoder import decode_query_sets
from utils import MODEL_COUNT


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    true_positive = float(np.logical_and(prediction, target).sum())
    false_positive = float(np.logical_and(prediction, ~target).sum())
    false_negative = float(np.logical_and(~prediction, target).sum())
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    return _safe_div(2.0 * precision * recall, precision + recall), precision, recall


def _error_summary(errors: Sequence[np.ndarray] | np.ndarray, reference_count: int | None = None) -> dict[str, float | int]:
    array = np.asarray(errors, dtype=np.float64).reshape(-1, 3)
    coverage = float(len(array) / reference_count) if reference_count else (1.0 if len(array) else 0.0)
    if not len(array):
        return {
            "count": 0, "coverage": coverage, "e_mae": float("nan"), "n_mae": float("nan"), "u_mae": float("nan"),
            "3d_mean": float("nan"), "3d_median": float("nan"), "3d_p75": float("nan"), "3d_p90": float("nan"), "3d_p95": float("nan"),
        }
    distance = np.linalg.norm(array, axis=1)
    return {
        "count": int(len(array)), "coverage": coverage,
        "e_mae": float(np.mean(np.abs(array[:, 0]))), "n_mae": float(np.mean(np.abs(array[:, 1]))), "u_mae": float(np.mean(np.abs(array[:, 2]))),
        "3d_mean": float(np.mean(distance)), "3d_median": float(np.median(distance)),
        "3d_p75": float(np.percentile(distance, 75)), "3d_p90": float(np.percentile(distance, 90)), "3d_p95": float(np.percentile(distance, 95)),
    }


def _target_presence(target_model_ids: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    result = np.zeros((len(target_model_ids), MODEL_COUNT), dtype=bool)
    for row in range(len(result)):
        result[row, target_model_ids[row, target_mask[row]].astype(np.int64)] = True
    return result


def evaluate_predictions(
    prediction: Mapping[str, np.ndarray],
    confidence_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate with the exact allowlist-masked query decoder used at test."""
    required = {
        "objectness_logits", "model_logits", "position_mu", "position_log_sigma", "allowlist_mask",
        "target_model_ids", "target_positions", "target_mask",
    }
    missing = required - set(prediction)
    if missing:
        raise ValueError(f"Prediction archive misses metric fields: {sorted(missing)}")
    decoded, decoder_info = decode_query_sets(
        prediction["objectness_logits"], prediction["model_logits"], prediction["position_mu"], prediction["position_log_sigma"],
        prediction["allowlist_mask"], confidence_threshold,
    )
    target_model_ids = np.asarray(prediction["target_model_ids"], dtype=np.int64)
    target_positions = np.asarray(prediction["target_positions"], dtype=np.float64)
    target_mask = np.asarray(prediction["target_mask"], dtype=bool)
    target = _target_presence(target_model_ids, target_mask)
    decoded_mask = np.asarray(decoder_info["decoded_model_mask"], dtype=bool)
    micro_f1, micro_precision, micro_recall = _f1(decoded_mask, target)
    per_model_f1 = [_f1(decoded_mask[:, model_id], target[:, model_id])[0] for model_id in range(MODEL_COUNT)]
    target_count = target.sum(axis=1)
    prediction_count = decoded_mask.sum(axis=1)

    correct_errors: list[np.ndarray] = []
    correct_models: list[int] = []
    correct_counts: list[int] = []
    query_errors: list[np.ndarray] = []
    query_models: list[int] = []
    query_counts: list[int] = []
    correct_reference_by_model = np.zeros(MODEL_COUNT, dtype=np.int64)
    query_reference_by_model = np.zeros(MODEL_COUNT, dtype=np.int64)
    reference_by_count = {count: 0 for count in (1, 2, 3)}

    position_mu = np.asarray(prediction["position_mu"], dtype=np.float64)
    for row in range(len(target)):
        valid = np.flatnonzero(target_mask[row])
        gt_models = target_model_ids[row, valid]
        gt_positions = target_positions[row, valid]
        count = int(len(valid))
        reference_by_count[count] += count
        gt_by_model = {int(model_id): position for model_id, position in zip(gt_models, gt_positions)}
        pred_by_model = {int(candidate["model_id"]): np.asarray(candidate["position"], dtype=np.float64) for candidate in decoded[row]}
        for model_id, position in gt_by_model.items():
            correct_reference_by_model[model_id] += 1
            query_reference_by_model[model_id] += 1
            if model_id in pred_by_model:
                correct_errors.append(pred_by_model[model_id] - position)
                correct_models.append(model_id)
                correct_counts.append(count)

        # This all-target assignment ignores predicted model identity and shows
        # the genuine ENU capacity of the three localization queries. It never
        # excludes a class-miss silently; every GT target is represented.
        distance_matrix = np.linalg.norm(position_mu[row, :, None, :] - gt_positions[None, :, :], axis=-1)
        query_index, target_column = linear_sum_assignment(distance_matrix)
        for query, column in zip(query_index, target_column):
            model_id = int(gt_models[column])
            query_errors.append(position_mu[row, query] - gt_positions[column])
            query_models.append(model_id)
            query_counts.append(count)

    report: dict[str, Any] = {
        "classification": {
            "micro_f1": micro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "macro_f1": float(np.mean(per_model_f1)),
            "exact_set_accuracy": float(np.mean(np.all(decoded_mask == target, axis=1))),
            "count_accuracy": float(np.mean(prediction_count == target_count)),
            "f1_by_model": {str(model_id): float(value) for model_id, value in enumerate(per_model_f1)},
        },
        "position_query_hungarian": _error_summary(query_errors, int(target_mask.sum())),
        "position_correct_model": _error_summary(correct_errors, int(target_mask.sum())),
        "position_by_model": {},
        "position_by_target_count": {},
        "position_correct_model_by_model": {},
        "position_correct_model_by_target_count": {},
    }
    query_model_array = np.asarray(query_models, dtype=np.int64)
    query_count_array = np.asarray(query_counts, dtype=np.int64)
    correct_model_array = np.asarray(correct_models, dtype=np.int64)
    correct_count_array = np.asarray(correct_counts, dtype=np.int64)
    query_error_array = np.asarray(query_errors, dtype=np.float64).reshape(-1, 3)
    correct_error_array = np.asarray(correct_errors, dtype=np.float64).reshape(-1, 3)
    for model_id in range(MODEL_COUNT):
        report["position_by_model"][str(model_id)] = _error_summary(
            query_error_array[query_model_array == model_id], int(query_reference_by_model[model_id])
        )
        report["position_correct_model_by_model"][str(model_id)] = _error_summary(
            correct_error_array[correct_model_array == model_id], int(correct_reference_by_model[model_id])
        )
    for count in (1, 2, 3):
        report["position_by_target_count"][str(count)] = _error_summary(
            query_error_array[query_count_array == count], reference_by_count[count]
        )
        report["position_correct_model_by_target_count"][str(count)] = _error_summary(
            correct_error_array[correct_count_array == count], reference_by_count[count]
        )
    extra = {
        "decoded": decoded,
        "decoded_model_mask": decoded_mask,
        "target_count": target_count,
        "prediction_count": prediction_count,
        **decoder_info,
    }
    return report, extra


def robust_selection_score(metrics: Mapping[str, Any], loss_cfg: LossConfig) -> float:
    """Early-stop rank using F1 plus explicit query localization long-tail error."""
    localization = metrics["position_query_hungarian"]
    median = float(localization["3d_median"])
    p90 = float(localization["3d_p90"])
    if not np.isfinite(median) or not np.isfinite(p90):
        return -float("inf")
    return (
        loss_cfg.selection_f1_weight * float(metrics["classification"]["micro_f1"])
        - loss_cfg.selection_distance_penalty * (median + 0.5 * p90)
    )
