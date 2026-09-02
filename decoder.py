"""Minimal robust fixed-slot decoder with mandatory allowlist hard masking."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def hard_mask_presence_logits(presence_logits: np.ndarray, allowlist_mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(presence_logits, dtype=np.float64).copy()
    allow = np.asarray(allowlist_mask, dtype=bool)
    if logits.shape != allow.shape:
        raise ValueError(f"presence/allowlist shape mismatch: {logits.shape} vs {allow.shape}")
    logits[allow] = -np.inf
    return logits


def decode_fixed_slots(
    presence_logits: np.ndarray,
    count_prob: np.ndarray,
    position_pred: np.ndarray,
    allowlist_mask: np.ndarray,
) -> tuple[list[list[dict[str, float]]], np.ndarray, np.ndarray]:
    """Hard-mask allowlist -> predicted-count -> 8-way Top-K, in that order."""
    raw_logits = np.asarray(presence_logits, dtype=np.float64)
    allow = np.asarray(allowlist_mask, dtype=bool)
    if raw_logits.shape != allow.shape:
        raise ValueError(f"presence/allowlist shape mismatch: {raw_logits.shape} vs {allow.shape}")
    # ``np.isfinite(masked_logits)`` alone would conflate an all-allowlisted
    # sample with a numerical model failure.  Never silently turn NaN/Inf into
    # a decoder decision: report the first affected rows and slots explicitly.
    invalid = ~np.isfinite(raw_logits)
    if np.any(invalid):
        rows = np.flatnonzero(np.any(invalid, axis=1))[:5]
        detail = [
            {
                "row": int(row),
                "invalid_slots": np.flatnonzero(invalid[row]).astype(int).tolist(),
                "allowed_slots": np.flatnonzero(~allow[row]).astype(int).tolist(),
            }
            for row in rows
        ]
        raise FloatingPointError(f"Non-finite presence logits before allowlist masking: {detail}")
    masked_logits = hard_mask_presence_logits(presence_logits, allowlist_mask)
    masked_prob = 1.0 / (1.0 + np.exp(-masked_logits))
    decoded_mask = np.zeros_like(masked_prob, dtype=bool)
    output: list[list[dict[str, float]]] = []
    for row in range(masked_logits.shape[0]):
        requested = int(np.argmax(count_prob[row])) + 1
        available = np.flatnonzero(np.isfinite(masked_logits[row]))
        if len(available) < 1:
            raise ValueError(
                f"sample row {row} has every model_id=0..7 in allowlist; "
                "the task requires at least one non-whitelisted prediction slot"
            )
        count = min(max(requested, 1), 3, len(available))
        chosen = available[np.argsort(masked_logits[row, available])[-count:]]
        chosen = np.sort(chosen)
        decoded_mask[row, chosen] = True
        drones = []
        for model_id in chosen:
            e, n, u = [float(value) for value in position_pred[row, model_id]]
            if not all(math.isfinite(value) for value in (e, n, u)):
                raise ValueError(f"Non-finite coordinate for row={row}, model_id={model_id}")
            drones.append({"model_id": int(model_id), "e_m": e, "n_m": n, "u_m": u})
        output.append(drones)
    return output, decoded_mask, masked_prob.astype(np.float32)


def write_submission(sample_ids: Sequence[int], drones: Sequence[Sequence[Mapping[str, Any]]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample_id, row_drones in zip(sample_ids, drones):
            payload = {"sample_id": int(sample_id), "drones": list(row_drones)}
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def audit_submission(path: str | Path, test_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path = Path(path)
    expected = {int(row["sample_id"]): row for row in test_rows}
    seen: set[int] = set()
    violations = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            payload = json.loads(line)
            sample_id = int(payload["sample_id"])
            if sample_id not in expected or sample_id in seen:
                raise ValueError(f"Invalid/duplicate sample_id at submission line {line_number}: {sample_id}")
            seen.add(sample_id)
            drones = payload.get("drones")
            if not isinstance(drones, list) or not 1 <= len(drones) <= 3:
                raise ValueError(f"Submission line {line_number} must contain 1..3 drones")
            ids = [int(drone["model_id"]) for drone in drones]
            if ids != sorted(ids) or len(ids) != len(set(ids)) or any(model < 0 or model >= 8 for model in ids):
                raise ValueError(f"Submission line {line_number} has invalid model-id ordering/duplicates")
            allow = {int(value) for value in str(expected[sample_id].get("allowlist", "")).split() if str(value).strip()}
            violations += sum(model in allow for model in ids)
            for drone in drones:
                values = (float(drone["e_m"]), float(drone["n_m"]), float(drone["u_m"]))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite coordinate at submission line {line_number}")
    if seen != set(expected):
        missing = sorted(set(expected) - seen)[:10]
        extra = sorted(seen - set(expected))[:10]
        raise ValueError(f"Submission samples do not exactly match test index; missing={missing}, extra={extra}")
    if violations:
        raise ValueError(f"allowlist violation={violations}; decoder hard mask failed")
    return {"submission": str(path), "rows": len(seen), "allowlist_violation": 0}
