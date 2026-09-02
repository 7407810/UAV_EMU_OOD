"""Reproducible five-fold label-stratified CV without trajectory assumptions."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from utils import atomic_json_dump, signature_from_models


def _strata(labels: Sequence[Mapping[str, Any]], folds: int) -> np.ndarray:
    """Balance target count and model-set prevalence without using it as input."""
    raw = []
    for label in labels:
        models = np.flatnonzero(np.asarray(label["presence"], dtype=bool)).tolist()
        raw.append(f"count={len(models)}|set={signature_from_models(models)}")
    frequency = Counter(raw)
    # StratifiedKFold cannot stratify a class occurring fewer than n_splits.
    # Such a rare label set is merged only into its target-count stratum; this
    # is a fold-balance fallback, not a model prediction prior.
    return np.asarray([value if frequency[value] >= folds else value.split("|")[0] for value in raw])


def make_stratified_folds(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    n_splits: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >=2")
    if len(rows) != len(labels):
        raise ValueError("rows and labels length mismatch")
    strata = _strata(labels, n_splits)
    count = Counter(strata.tolist())
    if min(count.values()) < n_splits:
        raise ValueError(f"Insufficient samples for stratified {n_splits}-fold split: {dict(count)}")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(rows), dtype=np.int64)
    folds = [(train.astype(np.int64), validation.astype(np.int64)) for train, validation in splitter.split(indices, strata)]
    coverage = np.zeros(len(rows), dtype=np.int64)
    for _, validation in folds:
        coverage[validation] += 1
    if not np.all(coverage == 1):
        raise AssertionError("CV fold validation coverage is not exactly once per indexed sample")

    presence = np.stack([np.asarray(label["presence"], dtype=np.float32) for label in labels])
    report: dict[str, Any] = {
        "method": "StratifiedKFold over GT-derived target-count/model-set strata; no trajectory/session inference assumptions",
        "n_splits": n_splits,
        "seed": seed,
        "stratum_count": {key: int(value) for key, value in sorted(count.items())},
        "folds": [],
    }
    for fold, (train, validation) in enumerate(folds):
        report["folds"].append({
            "fold": fold,
            "train_samples": int(len(train)),
            "validation_samples": int(len(validation)),
            "train_model_positive_rate": presence[train].mean(axis=0).astype(float).tolist(),
            "validation_model_positive_rate": presence[validation].mean(axis=0).astype(float).tolist(),
            "validation_target_count": {
                str(target_count): int(np.sum(presence[validation].sum(axis=1) == target_count))
                for target_count in (1, 2, 3)
            },
        })
    return folds, report


def save_fold_artifacts(
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    report: Mapping[str, Any],
    output_dir: Path,
) -> None:
    payload = dict(report)
    payload["fold_indices"] = [
        {"train_indices": train.tolist(), "validation_indices": validation.tolist()}
        for train, validation in folds
    ]
    atomic_json_dump(payload, output_dir / "cv_folds.json")
