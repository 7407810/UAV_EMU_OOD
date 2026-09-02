"""Small deployment decoder for unordered target queries.

The only task-specific rule is mandatory allowlist masking of model logits
before a class is chosen. Query ranking/deduplication uses the detector's own
objectness and class probabilities; it contains no trajectory, position,
signature, model-combination or count heuristic.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils import MAX_TARGETS, MODEL_COUNT, parse_int_list


def hard_mask_model_logits(model_logits: np.ndarray, allowlist_mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(model_logits, dtype=np.float64)
    allowlist = np.asarray(allowlist_mask, dtype=bool)
    if logits.ndim != 3 or logits.shape[-1] != MODEL_COUNT:
        raise ValueError(f"model_logits must be [B,Q,{MODEL_COUNT}], got {logits.shape}")
    if allowlist.shape != (logits.shape[0], MODEL_COUNT):
        raise ValueError(f"allowlist shape {allowlist.shape} does not match logits {logits.shape}")
    invalid = ~np.isfinite(logits)
    if invalid.any():
        row, query, model = np.argwhere(invalid)[0]
        raise FloatingPointError(f"Non-finite model logit before allowlist mask at row={row}, query={query}, model={model}")
    result = logits.copy()
    result[np.broadcast_to(allowlist[:, None, :], result.shape)] = -np.inf
    if np.any(np.all(~np.isfinite(result), axis=-1)):
        row, query = np.argwhere(np.all(~np.isfinite(result), axis=-1))[0]
        raise ValueError(f"Every model is allowlisted at row={row}, query={query}; task requires a non-allowlisted target")
    return result


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    probability = np.exp(shifted)
    return probability / np.maximum(probability.sum(axis=-1, keepdims=True), 1.0e-12)


def query_probabilities(
    objectness_logits: np.ndarray,
    model_logits: np.ndarray,
    allowlist_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    objectness_logits = np.asarray(objectness_logits, dtype=np.float64)
    if objectness_logits.ndim != 2:
        raise ValueError(f"objectness_logits must be [B,Q], got {objectness_logits.shape}")
    if not np.isfinite(objectness_logits).all():
        row, query = np.argwhere(~np.isfinite(objectness_logits))[0]
        raise FloatingPointError(f"Non-finite objectness logit at row={row}, query={query}")
    masked_logits = hard_mask_model_logits(model_logits, allowlist_mask)
    objectness = 1.0 / (1.0 + np.exp(-objectness_logits))
    class_probability = _softmax(masked_logits)
    model_id = np.argmax(class_probability, axis=-1).astype(np.int64)
    confidence = objectness * np.take_along_axis(class_probability, model_id[..., None], axis=-1).squeeze(-1)
    return objectness.astype(np.float32), class_probability.astype(np.float32), model_id, confidence.astype(np.float32)


def _candidates_for_row(
    objectness: np.ndarray,
    class_probability: np.ndarray,
    model_id: np.ndarray,
    confidence: np.ndarray,
    position_mu: np.ndarray,
    position_log_sigma: np.ndarray,
    row: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for query in range(len(model_id[row])):
        position = np.asarray(position_mu[row, query], dtype=np.float64)
        log_sigma = np.asarray(position_log_sigma[row, query], dtype=np.float64)
        if not np.isfinite(position).all() or not np.isfinite(log_sigma).all():
            raise FloatingPointError(f"Non-finite query location/uncertainty at row={row}, query={query}")
        model = int(model_id[row, query])
        candidates.append({
            "query_index": int(query),
            "model_id": model,
            "objectness": float(objectness[row, query]),
            "class_probability": float(class_probability[row, query, model]),
            "confidence": float(confidence[row, query]),
            "position": position,
            "log_sigma": log_sigma,
        })
    return candidates


def _deduplicate_and_select(candidates: Sequence[Mapping[str, Any]], confidence_threshold: float) -> list[dict[str, Any]]:
    eligible = [dict(candidate) for candidate in candidates if float(candidate["confidence"]) >= confidence_threshold]
    if not eligible:
        eligible = [dict(max(candidates, key=lambda candidate: float(candidate["confidence"])))]
    best_by_model: dict[int, dict[str, Any]] = {}
    for candidate in eligible:
        model_id = int(candidate["model_id"])
        if model_id not in best_by_model or float(candidate["confidence"]) > float(best_by_model[model_id]["confidence"]):
            best_by_model[model_id] = candidate
    selected = sorted(best_by_model.values(), key=lambda candidate: float(candidate["confidence"]), reverse=True)[:MAX_TARGETS]
    return selected


def decode_query_sets(
    objectness_logits: np.ndarray,
    model_logits: np.ndarray,
    position_mu: np.ndarray,
    position_log_sigma: np.ndarray,
    allowlist_mask: np.ndarray,
    confidence_threshold: float,
) -> tuple[list[list[dict[str, Any]]], dict[str, np.ndarray]]:
    """Mask allowlist -> classify queries -> simple confidence set selection."""
    position_mu = np.asarray(position_mu, dtype=np.float32)
    position_log_sigma = np.asarray(position_log_sigma, dtype=np.float32)
    if position_mu.shape[:2] != np.asarray(objectness_logits).shape or position_mu.shape[-1] != 3:
        raise ValueError("position_mu shape must be [B,Q,3] aligned with objectness logits")
    if position_log_sigma.shape != position_mu.shape:
        raise ValueError("position_log_sigma shape must match position_mu")
    objectness, class_probability, model_id, confidence = query_probabilities(objectness_logits, model_logits, allowlist_mask)
    decoded: list[list[dict[str, Any]]] = []
    decoded_mask = np.zeros((len(objectness), MODEL_COUNT), dtype=bool)
    for row in range(len(objectness)):
        candidates = _candidates_for_row(objectness, class_probability, model_id, confidence, position_mu, position_log_sigma, row)
        chosen = _deduplicate_and_select(candidates, confidence_threshold)
        decoded_mask[row, [int(candidate["model_id"]) for candidate in chosen]] = True
        decoded.append(chosen)
    return decoded, {
        "query_objectness_prob": objectness,
        "query_model_prob": class_probability,
        "query_model_id": model_id,
        "query_confidence": confidence,
        "decoded_model_mask": decoded_mask,
    }


def ensemble_query_predictions(
    fold_predictions: Sequence[Mapping[str, np.ndarray]],
    allowlist_mask: np.ndarray,
    confidence_threshold: float,
) -> tuple[list[list[dict[str, Any]]], dict[str, np.ndarray]]:
    """Fuse fold candidates by model ID with objectness/uncertainty weights."""
    if not fold_predictions:
        raise ValueError("At least one fold prediction is required")
    reference_shape = np.asarray(fold_predictions[0]["objectness_logits"]).shape
    num_rows = reference_shape[0]
    per_fold: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for prediction in fold_predictions:
        if np.asarray(prediction["objectness_logits"]).shape != reference_shape:
            raise ValueError("Fold query shapes differ")
        per_fold.append(query_probabilities(
            prediction["objectness_logits"], prediction["model_logits"], allowlist_mask,
        ))

    model_score = np.zeros((num_rows, MODEL_COUNT), dtype=np.float32)
    position_by_model = np.zeros((num_rows, MODEL_COUNT, 3), dtype=np.float32)
    log_sigma_by_model = np.zeros((num_rows, MODEL_COUNT, 3), dtype=np.float32)
    decoded: list[list[dict[str, Any]]] = []
    decoded_mask = np.zeros((num_rows, MODEL_COUNT), dtype=bool)
    fold_count = len(fold_predictions)

    for row in range(num_rows):
        entries_by_model: dict[int, list[dict[str, Any]]] = {model_id: [] for model_id in range(MODEL_COUNT)}
        for fold_index, prediction in enumerate(fold_predictions):
            objectness, class_probability, model_id, confidence = per_fold[fold_index]
            candidates = _candidates_for_row(
                objectness, class_probability, model_id, confidence,
                np.asarray(prediction["position_mu"]), np.asarray(prediction["position_log_sigma"]), row,
            )
            # A fold contributes only its strongest query for a predicted model.
            for candidate in candidates:
                model = int(candidate["model_id"])
                previous = entries_by_model[model]
                if not previous or all(int(item.get("fold_index", -1)) != fold_index for item in previous):
                    candidate["fold_index"] = fold_index
                    entries_by_model[model].append(candidate)
                else:
                    existing_index = next(index for index, item in enumerate(previous) if int(item.get("fold_index", -1)) == fold_index)
                    if float(candidate["confidence"]) > float(previous[existing_index]["confidence"]):
                        candidate["fold_index"] = fold_index
                        previous[existing_index] = candidate

        fused_candidates: list[dict[str, Any]] = []
        for model_id, entries in entries_by_model.items():
            if not entries:
                continue
            confidence = np.asarray([float(entry["confidence"]) for entry in entries], dtype=np.float64)
            positions = np.stack([np.asarray(entry["position"], dtype=np.float64) for entry in entries])
            log_sigma = np.stack([np.asarray(entry["log_sigma"], dtype=np.float64) for entry in entries])
            score = float(confidence.sum() / fold_count)
            uncertainty_weight = confidence * np.exp(-np.clip(log_sigma.mean(axis=1), -12.0, 12.0))
            if not np.isfinite(uncertainty_weight).all() or uncertainty_weight.sum() <= 1.0e-12:
                uncertainty_weight = np.maximum(confidence, 1.0e-8)
            # Do not naively average ENU across folds. The uncertainty-aware
            # objectness weighting is the sole deployed fusion rule; a median
            # alternative is not silently selected from leaky cross-fold data.
            position = np.average(positions, axis=0, weights=uncertainty_weight)
            fused_log_sigma = np.average(log_sigma, axis=0, weights=uncertainty_weight)
            model_score[row, model_id] = score
            position_by_model[row, model_id] = position.astype(np.float32)
            log_sigma_by_model[row, model_id] = fused_log_sigma.astype(np.float32)
            fused_candidates.append({
                "model_id": model_id,
                "confidence": score,
                "position": position,
                "log_sigma": fused_log_sigma,
            })

        chosen = _deduplicate_and_select(fused_candidates, confidence_threshold)
        decoded_mask[row, [int(candidate["model_id"]) for candidate in chosen]] = True
        decoded.append(chosen)

    return decoded, {
        "model_score": model_score,
        "position_by_model": position_by_model,
        "log_sigma_by_model": log_sigma_by_model,
        "decoded_model_mask": decoded_mask,
        "position_ensemble_mode": np.asarray(["uncertainty_weighted_mean"]),
    }


def submission_drones(decoded: Sequence[Sequence[Mapping[str, Any]]]) -> list[list[dict[str, float]]]:
    result: list[list[dict[str, float]]] = []
    for candidates in decoded:
        drones = []
        for candidate in sorted(candidates, key=lambda item: int(item["model_id"])):
            e_m, n_m, u_m = [float(value) for value in np.asarray(candidate["position"]).tolist()]
            if not all(math.isfinite(value) for value in (e_m, n_m, u_m)):
                raise ValueError(f"Non-finite output position for model {candidate['model_id']}")
            drones.append({"model_id": int(candidate["model_id"]), "e_m": e_m, "n_m": n_m, "u_m": u_m})
        result.append(drones)
    return result


def write_submission(sample_ids: Sequence[int], drones: Sequence[Sequence[Mapping[str, Any]]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for sample_id, row_drones in zip(sample_ids, drones):
            handle.write(json.dumps({"sample_id": int(sample_id), "drones": list(row_drones)}, ensure_ascii=False, separators=(",", ":")) + "\n")


def audit_submission(path: str | Path, test_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target = Path(path)
    expected = {int(row["sample_id"]): row for row in test_rows}
    seen: set[int] = set()
    violations = 0
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            sample_id = int(payload["sample_id"])
            if sample_id not in expected or sample_id in seen:
                raise ValueError(f"Invalid/duplicate sample_id at line {line_number}: {sample_id}")
            seen.add(sample_id)
            drones = payload.get("drones")
            if not isinstance(drones, list) or not 1 <= len(drones) <= MAX_TARGETS:
                raise ValueError(f"Submission line {line_number} must contain 1..{MAX_TARGETS} drones")
            model_ids = [int(drone["model_id"]) for drone in drones]
            if model_ids != sorted(model_ids) or len(model_ids) != len(set(model_ids)) or any(model < 0 or model >= MODEL_COUNT for model in model_ids):
                raise ValueError(f"Invalid model IDs at submission line {line_number}")
            allowlist = set(parse_int_list(expected[sample_id].get("allowlist", "")))
            violations += sum(model in allowlist for model in model_ids)
            for drone in drones:
                coordinates = (float(drone["e_m"]), float(drone["n_m"]), float(drone["u_m"]))
                if not all(math.isfinite(value) for value in coordinates):
                    raise ValueError(f"Non-finite output coordinate at line {line_number}")
    if seen != set(expected):
        raise ValueError(f"Submission IDs do not exactly match test index; missing={sorted(set(expected) - seen)[:10]}")
    if violations:
        raise ValueError(f"allowlist violation={violations}; mandatory hard masking failed")
    return {"submission": str(target), "rows": len(seen), "allowlist_violation": 0}
