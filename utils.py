"""Shared safety, audit, serialization and numerical utilities."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


SLOT_COUNT = 8


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def canonical_relpath(value: str) -> PurePosixPath:
    """Normalise Windows/Linux index paths and reject paths leaving their split."""
    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe or empty indexed relative path: {value!r}")
    return path


def indexed_path(data_root: Path, split: str, relpath: str) -> Path:
    root = (Path(data_root) / split).resolve()
    out = (root / Path(canonical_relpath(relpath))).resolve()
    try:
        out.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Index path escaped split root: {relpath!r}") from exc
    return out


def parse_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return []
        items = text.replace("|", " ").replace(",", " ").split()
    result = sorted({int(x) for x in items})
    if any(x < 0 or x >= SLOT_COUNT for x in result):
        raise ValueError(f"model id outside [0, {SLOT_COUNT - 1}]: {result}")
    return result


def signature_from_models(models: Sequence[int]) -> str:
    return "|".join(str(x) for x in sorted(models))


def label_path_for_row(data_root: Path, split: str, row: Mapping[str, Any]) -> Path:
    radar_stem = canonical_relpath(str(row["radar_npy_relpath"])).stem
    stem = radar_stem[:-6] if radar_stem.endswith("_radar") else radar_stem
    return Path(data_root) / split / "label" / f"{stem}_label.json"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_index(data_root: Path, split: str) -> list[dict[str, str]]:
    path = Path(data_root) / split / "index.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing indexed split: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty index.csv: {path}")
    return rows


def _required_columns(split: str) -> set[str]:
    shared = {
        "sample_id", "has_eo", "eo_jpg_relpath", "iq_node0", "iq_node1",
        "iq_node2", "iq_node3", "iq_npz_relpath", "radar_npy_relpath", "allowlist",
    }
    return shared | ({"label_signature"} if split == "train" else set())


def find_dataset_description(data_root: Path) -> Path | None:
    # Prefer the actual UTF-8 filenames shipped with this dataset.  The legacy
    # fallback below remains only for older copies whose source filename was
    # already mojibake-encoded.
    preferred = [
        data_root / "数据集说明.md",
        data_root / "dataset_description" / "数据集说明.md",
        data_root / "dataset_description" / "dataset_description_en.md",
    ]
    found = next((path for path in preferred if path.is_file()), None)
    if found is not None:
        return found
    candidates = [
        data_root / "数据集说明.md",
        data_root / "dataset_description" / "数据集说明.md",
        data_root / "dataset_description" / "dataset_description_en.md",
    ]
    return next((p for p in candidates if p.is_file()), None)


def load_node_enu(data_root: Path) -> np.ndarray:
    path = Path(data_root) / "iq_node_enu.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing RF node geometry file: {path}")
    payload = read_json(path)
    coords = []
    for node in range(4):
        value = payload.get(f"node{node}")
        if not isinstance(value, Mapping):
            raise ValueError(f"iq_node_enu.json misses node{node}")
        coords.append([float(value["e_m"]), float(value["n_m"]), float(value["u_m"])])
    return np.asarray(coords, dtype=np.float32)


def parse_label(data_root: Path, split: str, row: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_json(label_path_for_row(data_root, split, row))
    allow = parse_int_list(payload.get("allowlist", row.get("allowlist", "")))
    drones = payload.get("drones", [])
    positions = np.full((SLOT_COUNT, 3), np.nan, dtype=np.float32)
    presence = np.zeros(SLOT_COUNT, dtype=np.float32)
    seen: set[int] = set()
    for drone in drones:
        model_id = int(drone["model_id"])
        if model_id in seen or model_id < 0 or model_id >= SLOT_COUNT:
            raise ValueError(f"Invalid duplicate/out-of-range label model_id in sample {row['sample_id']}")
        if model_id in allow:
            raise ValueError(f"GT target is also allowlisted in sample {row['sample_id']}")
        position = drone["position_enu"]
        xyz = [float(position["e_m"]), float(position["n_m"]), float(position["u_m"])]
        if not np.isfinite(xyz).all():
            raise ValueError(f"Non-finite GT coordinate in sample {row['sample_id']}")
        positions[model_id] = xyz
        presence[model_id] = 1.0
        seen.add(model_id)
    return {"allowlist": allow, "presence": presence, "positions": positions, "drones": drones}


def _binary_index_flag(row: Mapping[str, Any], field: str) -> int:
    try:
        value = int(str(row[field]).strip())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a 0/1 index field for sample {row.get('sample_id')}") from exc
    if value not in (0, 1):
        raise ValueError(f"{field} must be 0/1 for sample {row.get('sample_id')}, got {value}")
    return value


def _audit_iq_file(
    path: Path,
    row: Mapping[str, Any],
    sampling_rates: list[list[float]],
    durations_s: list[float],
) -> None:
    """Validate one official IQ NPZ against the documented contract."""
    required = {*(f"iq_node{i}" for i in range(4)), *(f"sr_node{i}" for i in range(4))}
    with np.load(path, allow_pickle=False) as npz:
        if not required.issubset(npz.files):
            raise ValueError(f"NPZ misses required IQ/SR fields: {path}; fields={npz.files}")
        for node in range(4):
            signal = npz[f"iq_node{node}"]
            rate_value = npz[f"sr_node{node}"]
            if signal.dtype != np.dtype(np.int16) or signal.ndim != 1 or signal.size % 2:
                raise ValueError(
                    f"Invalid iq_node{node} in {path}: dtype={signal.dtype}, shape={signal.shape}; "
                    "expected one-dimensional even-length int16 interleaved IQ"
                )
            if rate_value.dtype != np.dtype(np.float32) or rate_value.size != 1:
                raise ValueError(
                    f"Invalid sr_node{node} in {path}: dtype={rate_value.dtype}, shape={rate_value.shape}; "
                    "expected one float32 value"
                )
            rate = float(np.asarray(rate_value).reshape(-1)[0])
            if not np.isfinite(rate):
                raise ValueError(f"Non-finite sr_node{node} in {path}")
            present = _binary_index_flag(row, f"iq_node{node}")
            if present:
                if signal.size == 0 or rate <= 0.0:
                    raise ValueError(f"Present iq_node{node} has empty IQ or non-positive SR: {path}")
                sampling_rates[node].append(rate)
                durations_s.append((signal.size // 2) / rate)
            elif signal.size != 0 or rate != 0.0:
                raise ValueError(f"Missing iq_node{node} must have empty IQ and sr=0: {path}")


def causal_radar_points(points: np.ndarray) -> np.ndarray:
    """Remove positive-relative-time Radar rows under the supplied causal contract.

    The contract places the sample reference at ``t=0`` and defines the cloud
    as historical accumulation.  Positive rows are therefore excluded, rather
    than clipped, negated, or allowed to create a train-only future-time cue.
    This function is shared by dataset loading and every Radar-only CV/calibration
    statistic so the policy cannot leak through an auxiliary path.
    """
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Radar array must have shape (N,4), got {values.shape}")
    return values[values[:, 3] <= 0.0]


def _audit_radar_file(path: Path) -> dict[str, int | float | None]:
    """Validate one official Radar NPY and quantify excluded positive-time rows."""
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.dtype(np.float64) or values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Radar must be float64 (N,4), got dtype={values.dtype}, shape={values.shape}: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Radar contains non-finite values: {path}")
    if not len(values):
        return {
            "count": 0, "causal_count": 0, "time_min": None, "time_max": None,
            "negative": 0, "zero": 0, "positive": 0,
        }
    rel_time = values[:, 3]
    positive_rows = np.flatnonzero(rel_time > 0.0)
    return {
        "count": int(len(values)),
        "causal_count": int(len(values) - len(positive_rows)),
        "time_min": float(np.min(rel_time)),
        "time_max": float(np.max(rel_time)),
        "negative": int(np.count_nonzero(rel_time < 0.0)),
        "zero": int(np.count_nonzero(rel_time == 0.0)),
        "positive": int(np.count_nonzero(rel_time > 0.0)),
    }


def audit_dataset(data_root: str | Path, strict_counts: bool = True) -> dict[str, Any]:
    """Audit only files referenced by the two official index files.

    This deliberately never discovers IQ/Radar samples by directory enumeration,
    so unreferenced legacy files can never enter a training set.
    """
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    node_enu = load_node_enu(root)
    report: dict[str, Any] = {
        "data_root": str(root),
        "node_enu": node_enu.tolist(),
        "description": str(find_dataset_description(root) or ""),
        "splits": {},
    }
    expected = {"train": 7362, "test_public": 487}
    for split in ("train", "test_public"):
        rows = read_index(root, split)
        fields = set(rows[0])
        missing = _required_columns(split) - fields
        if missing:
            raise ValueError(f"{split}/index.csv missing fields: {sorted(missing)}")
        ids = [int(row["sample_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample_id in {split}/index.csv")
        if strict_counts and len(rows) != expected[split]:
            raise ValueError(f"{split} has {len(rows)} indexed samples; expected {expected[split]}")
        label_count = 0
        sampling_rates: list[list[float]] = [[] for _ in range(4)]
        durations_s: list[float] = []
        radar_points = 0
        radar_causal_points = 0
        radar_time_min, radar_time_max = float("inf"), -float("inf")
        radar_time_sign_counts = {"negative": 0, "zero": 0, "positive": 0}
        radar_files_with_positive_time = 0
        for row in rows:
            iq = indexed_path(root, split, row["iq_npz_relpath"])
            radar = indexed_path(root, split, row["radar_npy_relpath"])
            if not iq.is_file() or not radar.is_file():
                raise FileNotFoundError(f"Index references missing data: {iq} / {radar}")
            has_eo = _binary_index_flag(row, "has_eo")
            eo_relpath = str(row.get("eo_jpg_relpath", "")).strip()
            if has_eo:
                eo = indexed_path(root, split, row["eo_jpg_relpath"])
                if not eo.is_file():
                    raise FileNotFoundError(f"Index references missing EO image: {eo}")
            elif eo_relpath:
                raise ValueError(f"has_eo=0 but eo_jpg_relpath is non-empty for sample {row['sample_id']}")
            allow = parse_int_list(row.get("allowlist", ""))
            if len(allow) >= SLOT_COUNT:
                raise ValueError(
                    f"sample {row['sample_id']} allowlist disables all {SLOT_COUNT} model slots; "
                    "this contradicts the required 1..3 non-whitelisted targets"
                )
            _audit_iq_file(iq, row, sampling_rates, durations_s)
            radar_audit = _audit_radar_file(radar)
            count = int(radar_audit["count"])
            time_min = radar_audit["time_min"]
            time_max = radar_audit["time_max"]
            radar_points += count
            radar_causal_points += int(radar_audit["causal_count"])
            for sign in radar_time_sign_counts:
                radar_time_sign_counts[sign] += int(radar_audit[sign])
            radar_files_with_positive_time += int(int(radar_audit["positive"]) > 0)
            if time_min is not None and time_max is not None:
                radar_time_min = min(radar_time_min, time_min)
                radar_time_max = max(radar_time_max, time_max)
            if split == "train":
                label = parse_label(root, split, row)
                models = np.flatnonzero(label["presence"]).tolist()
                if not 1 <= len(models) <= 3:
                    raise ValueError(f"Training sample {row['sample_id']} has {len(models)} targets, expected 1..3")
                signature = signature_from_models(models)
                if signature != str(row["label_signature"]).strip():
                    raise ValueError(
                        f"label_signature is not GT-derived for sample {row['sample_id']}: "
                        f"{row['label_signature']!r} vs {signature!r}"
                    )
                if allow != label["allowlist"]:
                    raise ValueError(f"Index/label allowlist mismatch for sample {row['sample_id']}")
                if not np.any(label["presence"] * (1.0 - np.isin(np.arange(SLOT_COUNT), allow).astype(np.float32))):
                    raise ValueError(f"Training sample {row['sample_id']} has no non-allowlisted GT slot")
                label_count += 1
        iq_schema = {
            "all_indexed_files_checked": len(rows),
            "iq_dtype": "int16", "iq_shape": "(2*T,)", "sr_dtype": "float32",
            "sampling_rate_hz_by_node": [
                {"min": float(np.min(rate)), "median": float(np.median(rate)), "max": float(np.max(rate)), "count": len(rate)}
                if rate else {"count": 0}
                for rate in sampling_rates
            ],
            "capture_duration_s": (
                {"min": float(np.min(durations_s)), "median": float(np.median(durations_s)), "max": float(np.max(durations_s))}
                if durations_s else {"count": 0}
            ),
        }
        radar_schema = {
            "all_indexed_files_checked": len(rows), "dtype": "float64", "shape": "(N,4)",
            "total_points_raw": radar_points,
            "causal_points_retained": radar_causal_points,
            "positive_rel_time_points_excluded": radar_time_sign_counts["positive"],
            "files_with_positive_rel_time": radar_files_with_positive_time,
            "positive_rel_time_policy": "exclude rows where rel_time_s > 0; never clip, negate, or expose them to model/CV statistics",
            "rel_time_s": (
                {
                    "min": radar_time_min,
                    "max": radar_time_max,
                    "sign_counts": radar_time_sign_counts,
                    "policy": "observed and preserved; no assumed <=0 convention",
                }
                if radar_points else {"count": 0}
            ),
        }
        report["splits"][split] = {
            "indexed_samples": len(rows), "labels_checked": label_count,
            "iq_schema": iq_schema, "radar_schema": radar_schema,
        }
    return report


def ensure_runtime_paths(data_root: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(data_root).expanduser()
    out = Path(output_dir).expanduser()
    if not root.is_dir():
        raise PermissionError(f"Cannot read requested data root: {root}")
    out.mkdir(parents=True, exist_ok=True)
    probe = out / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(f"Cannot write requested output directory: {out}") from exc
    return root.resolve(), out.resolve()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        temp = Path(handle.name)
    os.replace(temp, path)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as handle:
        temp = Path(handle.name)
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def sha256_json(payload: Any) -> str:
    serial = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=json_default, separators=(",", ":"))
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()


@dataclass
class EMA:
    decay: float
    shadow: dict[str, torch.Tensor] = field(default_factory=dict)

    def initialize(self, model: torch.nn.Module) -> None:
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        if not self.shadow:
            self.initialize(model)
            return
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {key: value.clone() for key, value in state["shadow"].items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, value in self.shadow.items():
            state[key].copy_(value.to(device=state[key].device, dtype=state[key].dtype))
        model.load_state_dict(state, strict=True)


class EMAScope:
    """Temporarily evaluate a model with its EMA weights, then restore it."""

    def __init__(self, model: torch.nn.Module, ema: EMA | None):
        self.model, self.ema = model, ema
        self.original: dict[str, torch.Tensor] | None = None

    def __enter__(self) -> torch.nn.Module:
        if self.ema is not None:
            self.original = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
            self.ema.copy_to(self.model)
        self.model.eval()
        return self.model

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.original is not None:
            self.model.load_state_dict(self.original, strict=True)


def tensor_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def robust_location_stats(labels: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    all_pos = np.concatenate([item["positions"][item["presence"] > 0] for item in labels], axis=0)
    mean = np.median(all_pos, axis=0)
    mad = np.median(np.abs(all_pos - mean), axis=0)
    std = np.maximum(1.4826 * mad, 5.0)
    return {"enu_mean": mean.astype(float).tolist(), "enu_std": std.astype(float).tolist()}


def radar_fingerprint(points: np.ndarray) -> np.ndarray:
    """Translation-robust causal point-cloud descriptor for CV diagnostics."""
    points = causal_radar_points(points)
    if points.size == 0:
        # 1 log-count + 5 radius quantiles + 5 time quantiles + 3 spatial
        # spreads + 5 vertical-offset quantiles + 1 time span.
        return np.zeros(20, dtype=np.float32)
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    time = np.asarray(points[:, 3], dtype=np.float32)
    center = np.median(xyz, axis=0, keepdims=True)
    centered = xyz - center
    radius = np.linalg.norm(centered, axis=1)
    q = lambda x: np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9]).astype(np.float32)
    return np.concatenate([
        np.asarray([np.log1p(len(xyz))], dtype=np.float32),
        q(radius), q(time), np.std(centered, axis=0).astype(np.float32), q(np.abs(centered[:, 2])),
        np.asarray([float(np.ptp(time))], dtype=np.float32),
    ])


def pairwise_nearest_distances(query: np.ndarray, reference: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Exact Euclidean nearest distance without scipy, bounded in memory."""
    if len(reference) == 0:
        return np.full(len(query), np.inf, dtype=np.float32)
    result = np.empty(len(query), dtype=np.float32)
    ref2 = np.sum(reference * reference, axis=1)
    for start in range(0, len(query), chunk):
        block = query[start:start + chunk]
        dist2 = np.sum(block * block, axis=1, keepdims=True) + ref2[None, :] - 2.0 * block @ reference.T
        result[start:start + len(block)] = np.sqrt(np.maximum(dist2.min(axis=1), 0.0))
    return result


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {f"q{q:02d}": float("nan") for q in (1, 5, 10, 25, 50, 75, 90)}
    out = {f"q{q:02d}": float(np.percentile(values, q)) for q in (1, 5, 10, 25, 50, 75, 90)}
    out.update({f"le_{r:g}m": float(np.mean(values <= r)) for r in (1, 5, 10, 20, 30)})
    return out
