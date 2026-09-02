"""Hard test-inference parity gate for a saved CV validation fold."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inference import formal_inference_from_checkpoint, load_prediction_archive
from utils import load_node_enu, parse_label, read_index


def _max_diff(name: str, expected: np.ndarray, actual: np.ndarray, sample_ids: np.ndarray) -> dict[str, Any]:
    if expected.shape != actual.shape:
        return {"name": name, "shape_expected": list(expected.shape), "shape_actual": list(actual.shape), "max_abs": float("inf")}
    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    flat = int(np.nanargmax(delta)) if delta.size else 0
    index = np.unravel_index(flat, delta.shape) if delta.size else (0,)
    row = int(index[0]) if index else 0
    return {
        "name": name, "max_abs": float(delta[index]) if delta.size else 0.0,
        "argmax": [int(value) for value in index], "sample_id": int(sample_ids[row]) if len(sample_ids) else None,
        "expected": float(expected[index]) if delta.size else None, "actual": float(actual[index]) if delta.size else None,
    }


def run_parity_check(
    data_root: str | Path,
    checkpoint_path: str | Path,
    oof_path: str | Path,
    device: torch.device | None = None,
    probability_atol: float = 1e-4,
    position_atol: float = 1e-4,
) -> dict[str, Any]:
    """Re-read an OOF fold through the formal checkpoint/test inference entry."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected = load_prediction_archive(oof_path)
    root = Path(data_root).resolve()
    rows = read_index(root, "train")
    required = {"sample_id", "source_index", "presence_prob", "count_prob", "position_pred"}
    missing = required - set(expected)
    if missing:
        raise RuntimeError(f"OOF archive missing parity fields: {sorted(missing)}")
    val_indices = expected["source_index"].astype(np.int64)
    if len(np.unique(val_indices)) != len(val_indices):
        raise RuntimeError("OOF archive has duplicate source_index values")
    labels: list[Any] = [None] * len(rows)
    for index in val_indices:
        labels[int(index)] = parse_label(root, "train", rows[int(index)])
    actual, loaded = formal_inference_from_checkpoint(
        checkpoint_path, root, "train", rows, val_indices, load_node_enu(root), device, labels=labels,
    )
    if not np.array_equal(expected["sample_id"], actual["sample_id"]):
        raise RuntimeError("Parity source mismatch: formal inference changed sample_id ordering")
    if not np.array_equal(expected["source_index"], actual["source_index"]):
        raise RuntimeError("Parity source mismatch: formal inference changed validation index ordering")
    comparisons = [
        _max_diff("presence_prob", expected["presence_prob"], actual["presence_prob"], expected["sample_id"]),
        _max_diff("count_prob", expected["count_prob"], actual["count_prob"], expected["sample_id"]),
        _max_diff("position_pred", expected["position_pred"], actual["position_pred"], expected["sample_id"]),
    ]
    failed = [
        item for item in comparisons
        if item["max_abs"] > (position_atol if item["name"] == "position_pred" else probability_atol)
    ]
    report = {
        "passed": not failed, "checkpoint": str(checkpoint_path), "oof": str(oof_path),
        "device": str(device), "probability_atol": probability_atol, "position_atol": position_atol,
        "differences": comparisons,
        "inference_contract": {
            "model_eval": not loaded.model.training,
            "checkpoint_is_best": bool(loaded.checkpoint.get("is_best", False)),
            "ema_loaded": bool(loaded.ema.shadow),
            "raw_iq_length": loaded.config.data.raw_iq_length,
            "iq_window_seconds": loaded.config.data.iq_window_seconds,
            "stft_shape": [loaded.config.data.stft_freq_bins, loaded.config.data.stft_time_bins],
            "max_radar_points": loaded.config.data.max_radar_points,
            "calibration_init": loaded.checkpoint.get("calibration_init"),
            "calibration_final": loaded.model.calibration_state(),
        },
    }
    if failed:
        print("[PARITY FAILED] formal inference differs from the saved training OOF.")
        for item in failed:
            print(f"  {item['name']}: max_abs={item['max_abs']:.8g}, sample_id={item.get('sample_id')}, index={item.get('argmax')}, expected={item.get('expected')}, actual={item.get('actual')}")
        print("  diagnostic contract:", report["inference_contract"])
        print("  Check eval mode, EMA state, fold preprocessing/STFT, calibration, TTA count, sigmoid/softmax use, slot order, and best-checkpoint selection.")
        raise RuntimeError("Inference parity gate failed; submission generation is forbidden.")
    print("[PARITY PASSED]", "; ".join(f"{item['name']} max_abs={item['max_abs']:.3g}" for item in comparisons))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal validation-as-test parity check")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--probability-atol", type=float, default=1e-4)
    parser.add_argument("--position-atol", type=float, default=1e-4)
    args = parser.parse_args()
    run_parity_check(args.data_root, args.checkpoint, args.oof, probability_atol=args.probability_atol, position_atol=args.position_atol)


if __name__ == "__main__":
    main()
