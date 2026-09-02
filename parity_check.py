"""Mandatory validation-as-test parity gate for a saved query-detector fold."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inference import formal_inference_from_checkpoint, load_prediction_archive
from utils import parse_label, read_index


def _max_difference(name: str, expected: np.ndarray, actual: np.ndarray, sample_ids: np.ndarray) -> dict[str, Any]:
    if expected.shape != actual.shape:
        return {"name": name, "shape_expected": list(expected.shape), "shape_actual": list(actual.shape), "max_abs": float("inf")}
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    flat_index = int(np.nanargmax(difference)) if difference.size else 0
    index = np.unravel_index(flat_index, difference.shape) if difference.size else (0,)
    row = int(index[0]) if index else 0
    return {
        "name": name,
        "max_abs": float(difference[index]) if difference.size else 0.0,
        "argmax": [int(value) for value in index],
        "sample_id": int(sample_ids[row]) if len(sample_ids) else None,
        "expected": float(expected[index]) if difference.size else None,
        "actual": float(actual[index]) if difference.size else None,
    }


def run_parity_check(
    data_root: str | Path,
    checkpoint_path: str | Path,
    oof_path: str | Path,
    device: torch.device | None = None,
    probability_atol: float = 1.0e-4,
    position_atol: float = 1.0e-4,
) -> dict[str, Any]:
    """Treat one saved CV validation fold exactly like public test inference."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected = load_prediction_archive(oof_path)
    required = {
        "sample_id", "source_index", "query_objectness_prob", "query_model_prob", "position_mu", "position_log_sigma",
    }
    missing = required - set(expected)
    if missing:
        raise RuntimeError(f"OOF archive misses required parity fields: {sorted(missing)}")
    root = Path(data_root).resolve()
    rows = read_index(root, "train")
    validation_indices = expected["source_index"].astype(np.int64)
    if len(np.unique(validation_indices)) != len(validation_indices):
        raise RuntimeError("OOF archive has duplicate source indices")
    labels: list[Any] = [None] * len(rows)
    for index in validation_indices:
        labels[int(index)] = parse_label(root, "train", rows[int(index)])
    actual, loaded = formal_inference_from_checkpoint(
        checkpoint_path,
        root,
        "train",
        rows,
        validation_indices,
        device,
        labels=labels,
        progress_desc="parity validation-as-test inference",
    )
    if not np.array_equal(expected["sample_id"], actual["sample_id"]):
        raise RuntimeError("Parity source mismatch: sample ordering changed")
    if not np.array_equal(expected["source_index"], actual["source_index"]):
        raise RuntimeError("Parity source mismatch: validation indices changed")
    comparisons = [
        _max_difference("query_objectness_prob", expected["query_objectness_prob"], actual["query_objectness_prob"], expected["sample_id"]),
        _max_difference("query_model_prob", expected["query_model_prob"], actual["query_model_prob"], expected["sample_id"]),
        _max_difference("position_mu", expected["position_mu"], actual["position_mu"], expected["sample_id"]),
        _max_difference("position_log_sigma", expected["position_log_sigma"], actual["position_log_sigma"], expected["sample_id"]),
    ]
    failed = [
        item for item in comparisons
        if item["max_abs"] > (position_atol if item["name"].startswith("position_") else probability_atol)
    ]
    report = {
        "passed": not failed,
        "checkpoint": str(checkpoint_path),
        "oof": str(oof_path),
        "device": str(device),
        "probability_atol": probability_atol,
        "position_atol": position_atol,
        "differences": comparisons,
        "inference_contract": {
            "model_eval": not loaded.model.training,
            "checkpoint_is_best": bool(loaded.checkpoint.get("is_best", False)),
            "ema_loaded": bool(loaded.ema.shadow),
            "raw_iq_length": loaded.config.data.raw_iq_length,
            "iq_window_seconds": loaded.config.data.iq_window_seconds,
            "stft_shape": [loaded.config.data.stft_freq_bins, loaded.config.data.stft_time_bins],
            "max_radar_points": loaded.config.data.max_radar_points,
            "fold_enu_normalization": loaded.checkpoint["fold_stats"].get("enu_mean"),
            "fold_radar_normalization": loaded.checkpoint["fold_stats"].get("radar_mean"),
            "tta": "none",
        },
    }
    if failed:
        print("[PARITY FAILED] formal checkpoint inference differs from saved OOF")
        for item in failed:
            print(
                f"  {item['name']}: max_abs={item['max_abs']:.8g}, sample_id={item.get('sample_id')}, "
                f"index={item.get('argmax')}, expected={item.get('expected')}, actual={item.get('actual')}"
            )
        print("  Check model.eval(), EMA, fold statistics, Dataset seed/subsampling, STFT setup, checkpoint epoch and probability conversion.")
        raise RuntimeError("Inference parity failed; submission generation is forbidden.")
    print("[PARITY PASSED] " + "; ".join(f"{item['name']} max_abs={item['max_abs']:.3g}" for item in comparisons))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-as-test parity for a set-prediction checkpoint")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--probability-atol", type=float, default=1.0e-4)
    parser.add_argument("--position-atol", type=float, default=1.0e-4)
    args = parser.parse_args()
    run_parity_check(args.data_root, args.checkpoint, args.oof, probability_atol=args.probability_atol, position_atol=args.position_atol)


if __name__ == "__main__":
    main()
