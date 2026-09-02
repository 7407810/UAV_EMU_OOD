"""Leakage-resistant trajectory/session grouping and CV credibility diagnostics."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from config import SessionConfig
from utils import (
    SLOT_COUNT,
    atomic_json_dump,
    causal_radar_points,
    indexed_path,
    pairwise_nearest_distances,
    percentile_summary,
    radar_fingerprint,
)


RADAR_FINGERPRINT_PREPROCESS_REVISION = 2


@dataclass
class SessionResult:
    session_ids: np.ndarray
    radar_features: np.ndarray
    chosen_config: dict[str, Any]
    candidate_report: list[dict[str, Any]]


def precompute_radar_fingerprints(
    data_root: Path,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    cache_path: Path | None = None,
) -> np.ndarray:
    """Read exactly the index-referenced radar files and cache compact descriptors."""
    wanted_ids = np.asarray([int(row["sample_id"]) for row in rows], dtype=np.int64)
    if cache_path is not None and cache_path.is_file():
        cached = np.load(cache_path)
        cached_revision = int(cached["preprocess_revision"][0]) if "preprocess_revision" in cached.files else -1
        if cached_revision == RADAR_FINGERPRINT_PREPROCESS_REVISION and np.array_equal(cached["sample_ids"], wanted_ids):
            return cached["features"].astype(np.float32, copy=False)
    features = []
    for row in rows:
        points = np.load(indexed_path(data_root, split, str(row["radar_npy_relpath"])), allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(f"Invalid radar array for indexed sample {row['sample_id']}: {points.shape}")
        features.append(radar_fingerprint(causal_radar_points(points)))
    out = np.stack(features).astype(np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            sample_ids=wanted_ids,
            features=out,
            preprocess_revision=np.asarray([RADAR_FINGERPRINT_PREPROCESS_REVISION], dtype=np.int16),
        )
    return out


def _robust_zscore(features: np.ndarray) -> np.ndarray:
    center = np.median(features, axis=0, keepdims=True)
    scale = np.median(np.abs(features - center), axis=0, keepdims=True) * 1.4826
    scale = np.maximum(scale, 1e-3)
    return (features - center) / scale


def _session_once(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    features_z: np.ndarray,
    cfg: SessionConfig,
) -> np.ndarray:
    """Greedy local linking, deliberately not a transitive spatial Union-Find.

    A sample may only attach to an *already observed nearby predecessor*.  A
    session is capped by both ID span and sample count, so a long continuous
    flight cannot collapse the dataset into a few giant connected components.
    """
    n = len(rows)
    order = np.argsort(np.asarray([int(row["sample_id"]) for row in rows]))
    ids = np.asarray([int(row["sample_id"]) for row in rows])
    session = np.full(n, -1, dtype=np.int64)
    first_id: list[int] = []
    size: list[int] = []
    next_session = 0

    for order_pos, index in enumerate(order):
        current_models = set(np.flatnonzero(labels[index]["presence"]).tolist())
        candidates: list[tuple[float, int]] = []
        start = max(0, order_pos - cfg.neighbor_search_back)
        for previous_index in order[start:order_pos][::-1]:
            gap = int(ids[index] - ids[previous_index])
            if gap <= 0 or gap > cfg.neighbor_sample_gap:
                continue
            previous_models = set(np.flatnonzero(labels[previous_index]["presence"]).tolist())
            shared = current_models & previous_models
            if not shared:
                continue
            if len(current_models ^ previous_models) > cfg.max_signature_delta:
                continue
            distances = [
                float(np.linalg.norm(labels[index]["positions"][model] - labels[previous_index]["positions"][model]))
                for model in shared
            ]
            if max(distances) > cfg.continuity_distance_m:
                continue
            feature_distance = float(np.sqrt(np.mean((features_z[index] - features_z[previous_index]) ** 2)))
            if feature_distance > cfg.radar_similarity_z:
                continue
            sid = int(session[previous_index])
            if size[sid] >= cfg.max_session_samples or int(ids[index] - first_id[sid]) > cfg.max_session_id_span:
                continue
            # Prefer trajectory continuity first, then radar similarity and sample proximity.
            score = max(distances) / cfg.continuity_distance_m + feature_distance / cfg.radar_similarity_z + 0.03 * gap
            candidates.append((score, sid))
        if candidates:
            _, sid = min(candidates, key=lambda value: value[0])
            session[index] = sid
            size[sid] += 1
        else:
            session[index] = next_session
            first_id.append(int(ids[index]))
            size.append(1)
            next_session += 1
    return session


def _build_strata(rows: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([
        f"n{int(np.sum(label['presence']))}:{str(row['label_signature']).strip()}"
        for row, label in zip(rows, labels)
    ])


def make_stratified_group_folds(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    session_ids: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use StratifiedGroupKFold, with an explicit deterministic group fallback."""
    y = _build_strata(rows, labels)
    indices = np.arange(len(rows))
    if len(np.unique(session_ids)) < n_splits:
        raise ValueError(f"Only {len(np.unique(session_ids))} sessions for {n_splits} folds")
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [(train.astype(np.int64), val.astype(np.int64)) for train, val in splitter.split(indices, y, groups=session_ids)]
    except Exception as exc:  # pragma: no cover - fallback protects minimal environments
        print(f"[session-cv] StratifiedGroupKFold unavailable/failed ({exc}); using deterministic grouped fallback")
        folds = _greedy_group_folds(y, session_ids, n_splits, seed)
    for fold, (train, val) in enumerate(folds):
        train_groups = set(session_ids[train].tolist())
        val_groups = set(session_ids[val].tolist())
        if train_groups & val_groups:
            raise AssertionError(f"session leakage in fold {fold}")
    return folds


def _greedy_group_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)
    classes, y_code = np.unique(y, return_inverse=True)
    group_hist = np.zeros((len(unique), len(classes)), dtype=np.float64)
    group_size = np.zeros(len(unique), dtype=np.float64)
    for row, group in enumerate(inverse):
        group_hist[group, y_code[row]] += 1.0
        group_size[group] += 1.0
    order = np.arange(len(unique))
    rng.shuffle(order)
    order = sorted(order, key=lambda item: (group_size[item], group_hist[item].max()), reverse=True)
    target = group_hist.sum(axis=0) / n_splits
    folds_hist = np.zeros((n_splits, len(classes)), dtype=np.float64)
    folds_size = np.zeros(n_splits, dtype=np.float64)
    group_fold = np.zeros(len(unique), dtype=np.int64)
    for group in order:
        costs = []
        for fold in range(n_splits):
            candidate = folds_hist.copy()
            candidate[fold] += group_hist[group]
            class_cost = np.mean(((candidate - target) / np.maximum(target, 1.0)) ** 2)
            size_cost = ((folds_size[fold] + group_size[group] - group_size.sum() / n_splits) / max(group_size.sum() / n_splits, 1.0)) ** 2
            costs.append(class_cost + 0.15 * size_cost)
        choice = int(np.argmin(costs))
        group_fold[group] = choice
        folds_hist[choice] += group_hist[group]
        folds_size[choice] += group_size[group]
    return [(np.flatnonzero(group_fold[inverse] != fold), np.flatnonzero(group_fold[inverse] == fold)) for fold in range(n_splits)]


def _fold_balance(rows: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]], folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    sizes = np.asarray([len(val) for _, val in folds], dtype=float)
    all_models = np.stack([label["presence"] for label in labels])
    global_rate = all_models.mean(axis=0)
    model_dev = []
    for _, val in folds:
        model_dev.append(np.abs(all_models[val].mean(axis=0) - global_rate).mean())
    return {
        "fold_size_min": float(sizes.min()), "fold_size_max": float(sizes.max()),
        "fold_size_cv": float(sizes.std() / max(sizes.mean(), 1.0)),
        "mean_model_rate_deviation": float(np.mean(model_dev)),
    }


def build_trajectory_sessions(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    radar_features: np.ndarray,
    cfg: SessionConfig,
) -> SessionResult:
    """Select a small, bounded local-link threshold family by fold balance."""
    z = _robust_zscore(radar_features)
    candidates = []
    parameter_grid = [
        (distance, radar_z, cap)
        for distance in (32.0, cfg.continuity_distance_m, 50.0)
        for radar_z in (1.1, cfg.radar_similarity_z, 2.2)
        for cap in (64, cfg.max_session_samples)
    ]
    seen: set[tuple[float, float, int]] = set()
    target_sessions = max(cfg.folds * 16, len(rows) / 36.0)
    for distance, radar_z, cap in parameter_grid:
        key = (distance, radar_z, cap)
        if key in seen:
            continue
        seen.add(key)
        trial = replace(cfg, continuity_distance_m=distance, radar_similarity_z=radar_z, max_session_samples=cap)
        session_ids = _session_once(rows, labels, z, trial)
        session_sizes = np.bincount(session_ids)
        n_sessions = len(session_sizes)
        report: dict[str, Any] = {
            "continuity_distance_m": distance, "radar_similarity_z": radar_z, "max_session_samples": cap,
            "sessions": n_sessions, "max_session_size": int(session_sizes.max()),
        }
        if n_sessions < cfg.folds * cfg.min_sessions_per_fold or session_sizes.max() > len(rows) / cfg.folds * 1.4:
            report["eligible"] = False
            report["score"] = float("inf")
            candidates.append((float("inf"), session_ids, trial, report))
            continue
        try:
            folds = make_stratified_group_folds(rows, labels, session_ids, cfg.folds, cfg.seed)
            balance = _fold_balance(rows, labels, folds)
            session_shape = abs(math.log(max(n_sessions, 1) / target_sessions))
            score = balance["fold_size_cv"] * 7.0 + balance["mean_model_rate_deviation"] * 30.0 + session_shape * 0.30
            report.update(balance)
            report["eligible"] = True
            report["score"] = float(score)
            candidates.append((score, session_ids, trial, report))
        except Exception as exc:
            report.update({"eligible": False, "score": float("inf"), "error": str(exc)})
            candidates.append((float("inf"), session_ids, trial, report))
    viable = [item for item in candidates if np.isfinite(item[0])]
    if not viable:
        details = [item[3] for item in candidates]
        raise RuntimeError(f"No viable bounded session partition: {details}")
    best = min(viable, key=lambda item: item[0])
    return SessionResult(
        session_ids=best[1], radar_features=radar_features,
        chosen_config=asdict(best[2]), candidate_report=[item[3] for item in candidates],
    )


def _same_model_gt(labels: Sequence[Mapping[str, Any]], indices: np.ndarray, model_id: int) -> np.ndarray:
    out = [labels[int(i)]["positions"][model_id] for i in indices if labels[int(i)]["presence"][model_id] > 0]
    return np.asarray(out, dtype=np.float32).reshape(-1, 3)


def cv_credibility_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    radar_features_train: np.ndarray,
    radar_features_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Quantify GT and radar-nearest-neighbour similarity for every CV fold."""
    report: dict[str, Any] = {"folds": {}}
    oof_radar_nn = np.full(len(rows), np.nan, dtype=np.float32)
    all_target_distances: list[np.ndarray] = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        per_model: dict[str, Any] = {}
        fold_distances = []
        for model_id in range(SLOT_COUNT):
            reference = _same_model_gt(labels, train_idx, model_id)
            query = _same_model_gt(labels, val_idx, model_id)
            distances = pairwise_nearest_distances(query, reference)
            per_model[str(model_id)] = percentile_summary(distances)
            fold_distances.append(distances)
        target_distances = np.concatenate(fold_distances) if fold_distances else np.empty(0)
        all_target_distances.append(target_distances)
        train_fp = radar_features_train[train_idx]
        center = np.median(train_fp, axis=0, keepdims=True)
        scale = np.maximum(np.median(np.abs(train_fp - center), axis=0, keepdims=True) * 1.4826, 1e-4)
        val_nn = pairwise_nearest_distances((radar_features_train[val_idx] - center) / scale, (train_fp - center) / scale)
        test_nn = pairwise_nearest_distances((radar_features_test - center) / scale, (train_fp - center) / scale)
        oof_radar_nn[val_idx] = val_nn
        report["folds"][str(fold)] = {
            "gt_same_model_nearest_3d": percentile_summary(target_distances),
            "gt_same_model_nearest_3d_by_model": per_model,
            "radar_nn_validation_to_train": percentile_summary(val_nn),
            "radar_nn_test_to_train": percentile_summary(test_nn),
        }
    report["all_folds_gt_same_model_nearest_3d"] = percentile_summary(np.concatenate(all_target_distances))
    return report, oof_radar_nn


def radar_only_calibration_metadata(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
) -> dict[str, Any]:
    """Return *input-only* numerical conditioning for learned calibration.

    This deliberately does not read labels and does not estimate yaw or a
    translation.  The old centroid-to-GT alignment was fragile because a Radar
    cloud includes clutter, allowlist tracks and three seconds of history.  We
    instead start the calibration exactly at identity and expose only a robust
    Radar-coordinate scale so the latent translation has well-conditioned
    gradients on datasets expressed in metres with different coordinate ranges.
    """
    per_sample_scale: list[np.ndarray] = []
    per_sample_point_scale: list[np.ndarray] = []
    for index in train_indices:
        points = np.load(indexed_path(data_root, "train", str(rows[int(index)]["radar_npy_relpath"])), allow_pickle=False)
        points = causal_radar_points(points)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            continue
        xyz = np.asarray(points[:, :3], dtype=np.float64)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz) == 0:
            continue
        # Absolute magnitude covers a possible frame-origin offset, while local
        # span covers centred coordinate systems.  Neither term uses GT labels.
        absolute_q90 = np.quantile(np.abs(xyz), 0.90, axis=0)
        span_q = np.quantile(xyz, 0.90, axis=0) - np.quantile(xyz, 0.10, axis=0)
        per_sample_scale.append(np.maximum(absolute_q90, span_q))
        centered = xyz - np.median(xyz, axis=0, keepdims=True)
        per_sample_point_scale.append(np.quantile(np.abs(centered), 0.90, axis=0))
    if per_sample_scale:
        scale = np.median(np.stack(per_sample_scale, axis=0), axis=0)
    else:
        scale = np.full(3, 100.0, dtype=np.float64)
    if per_sample_point_scale:
        point_scale = np.median(np.stack(per_sample_point_scale, axis=0), axis=0)
    else:
        point_scale = np.full(3, 1.0, dtype=np.float64)
    # This is a conditioning floor in metres, not a predicted displacement or
    # calibration prior.  The sinh parameterization in the network is unbounded.
    scale = np.maximum(scale, 100.0).astype(np.float32)
    return {
        "mode": "learned_identity_radar_only_scale",
        "yaw_rad": 0.0,
        "tx": 0.0,
        "ty": 0.0,
        "tz": 0.0,
        "translation_scale_m": scale.astype(float).tolist(),
        "point_scale_m": np.maximum(point_scale, 1.0).astype(float).tolist(),
    }


def save_session_artifacts(result: SessionResult, folds: Sequence[tuple[np.ndarray, np.ndarray]], output_dir: Path) -> None:
    payload = {
        "session_ids": result.session_ids.tolist(),
        "chosen_config": result.chosen_config,
        "candidate_report": result.candidate_report,
        "folds": [{"train_indices": train.tolist(), "val_indices": val.tolist()} for train, val in folds],
    }
    atomic_json_dump(payload, output_dir / "cv_sessions.json")
