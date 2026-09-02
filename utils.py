"""Shared data-contract, serialization and numerical utilities.

Nothing in this module manufactures target tracks, model/position priors, or a
Radar-to-label transform.  It only validates the indexed official data and
computes fold-local normalization statistics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch


MODEL_COUNT = 8
MAX_TARGETS = 3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def canonical_relpath(value: str) -> PurePosixPath:
    """Normalize index paths and reject paths leaving the declared split."""
    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe or empty indexed relative path: {value!r}")
    return path


def indexed_path(data_root: Path, split: str, relpath: str) -> Path:
    split_root = (Path(data_root) / split).resolve()
    result = (split_root / Path(canonical_relpath(relpath))).resolve()
    try:
        result.relative_to(split_root)
    except ValueError as exc:
        raise ValueError(f"Index path escaped split root: {relpath!r}") from exc
    return result


def parse_int_list(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return []
        values = text.replace("|", " ").replace(",", " ").split()
    result = sorted({int(item) for item in values})
    if any(item < 0 or item >= MODEL_COUNT for item in result):
        raise ValueError(f"model_id outside [0,{MODEL_COUNT - 1}]: {result}")
    return result


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_index(data_root: str | Path, split: str) -> list[dict[str, str]]:
    path = Path(data_root) / split / "index.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing indexed split: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty index.csv: {path}")
    return rows


def label_path_for_row(data_root: Path, split: str, row: Mapping[str, Any]) -> Path:
    radar_stem = canonical_relpath(str(row["radar_npy_relpath"])).stem
    prefix = radar_stem[:-6] if radar_stem.endswith("_radar") else radar_stem
    return Path(data_root) / split / "label" / f"{prefix}_label.json"


def signature_from_models(models: Sequence[int]) -> str:
    return "|".join(str(model_id) for model_id in sorted(models))


def parse_label(data_root: str | Path, split: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Parse one training label into an unordered target set and audit views.

    The padded tensors are sorted only to make serialization deterministic.
    They have no query index meaning: Hungarian matching handles every target
    permutation during training.
    """
    payload = read_json(label_path_for_row(Path(data_root), split, row))
    allowlist = parse_int_list(payload.get("allowlist", row.get("allowlist", "")))
    raw_drones = payload.get("drones", [])
    if not isinstance(raw_drones, list):
        raise ValueError(f"label drones must be a list for sample {row.get('sample_id')}")

    seen: set[int] = set()
    drones: list[dict[str, Any]] = []
    presence = np.zeros(MODEL_COUNT, dtype=np.float32)
    positions = np.full((MODEL_COUNT, 3), np.nan, dtype=np.float32)
    for drone in raw_drones:
        if not isinstance(drone, Mapping):
            raise ValueError(f"Invalid drone object for sample {row.get('sample_id')}")
        model_id = int(drone["model_id"])
        if model_id < 0 or model_id >= MODEL_COUNT or model_id in seen:
            raise ValueError(f"Invalid duplicate/out-of-range model_id in sample {row.get('sample_id')}")
        if model_id in allowlist:
            raise ValueError(f"GT target is allowlisted in sample {row.get('sample_id')}")
        position = drone.get("position_enu")
        if not isinstance(position, Mapping):
            raise ValueError(f"Missing position_enu for sample {row.get('sample_id')}, model={model_id}")
        xyz = np.asarray([position["e_m"], position["n_m"], position["u_m"]], dtype=np.float32)
        if not np.isfinite(xyz).all():
            raise ValueError(f"Non-finite GT ENU for sample {row.get('sample_id')}, model={model_id}")
        drones.append({"model_id": model_id, "position": xyz})
        presence[model_id] = 1.0
        positions[model_id] = xyz
        seen.add(model_id)

    drones.sort(key=lambda item: int(item["model_id"]))
    if not 1 <= len(drones) <= MAX_TARGETS:
        raise ValueError(f"Sample {row.get('sample_id')} has {len(drones)} targets; expected 1..{MAX_TARGETS}")
    target_model_ids = np.full(MAX_TARGETS, -100, dtype=np.int64)
    target_positions = np.zeros((MAX_TARGETS, 3), dtype=np.float32)
    target_mask = np.zeros(MAX_TARGETS, dtype=np.bool_)
    for target_index, drone in enumerate(drones):
        target_model_ids[target_index] = int(drone["model_id"])
        target_positions[target_index] = np.asarray(drone["position"], dtype=np.float32)
        target_mask[target_index] = True
    return {
        "allowlist": allowlist,
        "drones": drones,
        "presence": presence,
        "positions": positions,
        "target_model_ids": target_model_ids,
        "target_positions": target_positions,
        "target_mask": target_mask,
    }


def causal_radar_points(points: np.ndarray) -> np.ndarray:
    """Keep only documented historical Radar points (``rel_time_s <= 0``).

    The official specification calls the cloud a three-second look-back window
    and defines its relative time as negative. Some training files contain a
    small number of positive-time rows while public test does not; retaining
    those would create a train-only future-time cue. Rows are excluded rather
    than modified, so no time value is invented.
    """
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Radar must have shape (N,4), got {values.shape}")
    return values[values[:, 3] <= 0.0]


def _binary_index_flag(row: Mapping[str, Any], field: str) -> int:
    try:
        value = int(str(row[field]).strip())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a 0/1 field for sample {row.get('sample_id')}") from exc
    if value not in (0, 1):
        raise ValueError(f"{field} must be 0/1 for sample {row.get('sample_id')}, got {value}")
    return value


def _read_npy_header(handle: Any) -> tuple[tuple[int, ...], bool, np.dtype]:
    """Read an NPY header from an NPZ member without decompressing IQ payload."""
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
    elif version == (3, 0):
        reader = getattr(np.lib.format, "read_array_header_3_0", None)
        if reader is None:
            # Numpy releases before the v3 helper can still parse the same
            # header representation through the v2 routine.
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            shape, fortran, dtype = reader(handle)
    else:
        raise ValueError(f"Unsupported NPY version {version}")
    return tuple(int(value) for value in shape), bool(fortran), np.dtype(dtype)


def _shape_size(shape: tuple[int, ...]) -> int:
    return int(math.prod(shape)) if shape else 1


def _audit_iq_file(
    path: Path,
    row: Mapping[str, Any],
    sampling_rates: list[list[float]],
    durations_s: list[float],
) -> None:
    """Audit IQ schema from NPY headers, loading only four scalar SR values."""
    required = {*(f"iq_node{node}" for node in range(4)), *(f"sr_node{node}" for node in range(4))}
    with np.load(path, allow_pickle=False) as npz, zipfile.ZipFile(path, "r") as archive:
        if not required.issubset(set(npz.files)):
            raise ValueError(f"NPZ misses required IQ/SR fields: {path}; fields={npz.files}")
        archive_names = set(archive.namelist())
        for node in range(4):
            member = f"iq_node{node}.npy"
            if member not in archive_names:
                raise ValueError(f"NPZ misses {member}: {path}")
            with archive.open(member) as handle:
                shape, fortran, dtype = _read_npy_header(handle)
            if dtype != np.dtype(np.int16) or fortran or len(shape) != 1 or shape[0] % 2:
                raise ValueError(
                    f"Invalid iq_node{node} header in {path}: dtype={dtype}, shape={shape}, fortran={fortran}; "
                    "expected a C-order even-length int16 vector"
                )
            rate_value = np.asarray(npz[f"sr_node{node}"])
            if rate_value.dtype != np.dtype(np.float32) or rate_value.size != 1:
                raise ValueError(
                    f"Invalid sr_node{node} in {path}: dtype={rate_value.dtype}, shape={rate_value.shape}; "
                    "expected one float32 value"
                )
            rate = float(rate_value.reshape(-1)[0])
            if not np.isfinite(rate):
                raise ValueError(f"Non-finite sr_node{node} in {path}")
            present = _binary_index_flag(row, f"iq_node{node}")
            signal_size = _shape_size(shape)
            if present:
                if signal_size == 0 or rate <= 0.0:
                    raise ValueError(f"Present iq_node{node} has empty IQ or non-positive SR: {path}")
                sampling_rates[node].append(rate)
                durations_s.append((signal_size // 2) / rate)
            elif signal_size != 0 or rate != 0.0:
                raise ValueError(f"Missing iq_node{node} must have empty IQ and sr=0: {path}")


def _audit_radar_file(path: Path) -> dict[str, int | float | None]:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.dtype(np.float64) or values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Radar must be float64 (N,4), got dtype={values.dtype}, shape={values.shape}: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Radar contains non-finite values: {path}")
    if len(values) == 0:
        return {"count": 0, "causal_count": 0, "time_min": None, "time_max": None, "positive": 0}
    rel_time = values[:, 3]
    positive = int(np.count_nonzero(rel_time > 0.0))
    return {
        "count": int(len(values)),
        "causal_count": int(len(values) - positive),
        "time_min": float(np.min(rel_time)),
        "time_max": float(np.max(rel_time)),
        "positive": positive,
    }


def _required_columns(split: str) -> set[str]:
    required = {
        "sample_id", "has_eo", "eo_jpg_relpath", "iq_node0", "iq_node1", "iq_node2", "iq_node3",
        "iq_npz_relpath", "radar_npy_relpath", "allowlist",
    }
    return required | ({"label_signature"} if split == "train" else set())


def find_dataset_description(data_root: str | Path) -> Path | None:
    root = Path(data_root)
    candidates = (
        root / "数据集说明.md",
        root / "dataset_description" / "数据集说明.md",
        root / "dataset_description" / "dataset_description_en.md",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def audit_dataset(data_root: str | Path, strict_counts: bool = True) -> dict[str, Any]:
    """Audit exactly and only samples referenced from official split indexes."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    expected = {"train": 7362, "test_public": 487}
    report: dict[str, Any] = {
        "data_root": str(root),
        "description": str(find_dataset_description(root) or ""),
        "splits": {},
        "audit_mode": "indexed_only_iq_header_plus_scalar_sr_and_full_radar_schema",
    }
    for split in ("train", "test_public"):
        rows = read_index(root, split)
        missing = _required_columns(split) - set(rows[0])
        if missing:
            raise ValueError(f"{split}/index.csv missing columns: {sorted(missing)}")
        sample_ids = [int(row["sample_id"]) for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Duplicate sample_id in {split}/index.csv")
        if strict_counts and len(rows) != expected[split]:
            raise ValueError(f"{split} has {len(rows)} index rows; expected {expected[split]}")

        sampling_rates: list[list[float]] = [[] for _ in range(4)]
        durations_s: list[float] = []
        radar_points = radar_causal_points = positive_time_points = 0
        positive_time_files = 0
        radar_time_min, radar_time_max = float("inf"), -float("inf")
        label_count = 0

        for row_index, row in enumerate(rows, start=1):
            iq_path = indexed_path(root, split, str(row["iq_npz_relpath"]))
            radar_path = indexed_path(root, split, str(row["radar_npy_relpath"]))
            if not iq_path.is_file() or not radar_path.is_file():
                raise FileNotFoundError(f"Index references missing file: {iq_path} / {radar_path}")
            has_eo = _binary_index_flag(row, "has_eo")
            eo_relpath = str(row.get("eo_jpg_relpath", "")).strip()
            if has_eo:
                eo_path = indexed_path(root, split, eo_relpath)
                if not eo_path.is_file():
                    raise FileNotFoundError(f"Index references missing EO file: {eo_path}")
            elif eo_relpath:
                raise ValueError(f"has_eo=0 but eo_jpg_relpath is non-empty for sample {row['sample_id']}")
            allowlist = parse_int_list(row.get("allowlist", ""))
            if len(allowlist) >= MODEL_COUNT:
                raise ValueError(f"sample {row['sample_id']} allowlists all model IDs")

            _audit_iq_file(iq_path, row, sampling_rates, durations_s)
            radar = _audit_radar_file(radar_path)
            radar_points += int(radar["count"])
            radar_causal_points += int(radar["causal_count"])
            positive_time_points += int(radar["positive"])
            positive_time_files += int(int(radar["positive"]) > 0)
            if radar["time_min"] is not None:
                radar_time_min = min(radar_time_min, float(radar["time_min"]))
                radar_time_max = max(radar_time_max, float(radar["time_max"]))

            if split == "train":
                label = parse_label(root, split, row)
                expected_signature = signature_from_models(np.flatnonzero(label["presence"]).tolist())
                if expected_signature != str(row["label_signature"]).strip():
                    raise ValueError(
                        f"label_signature mismatch for sample {row['sample_id']}: "
                        f"{row['label_signature']!r} vs GT-derived {expected_signature!r}"
                    )
                if allowlist != label["allowlist"]:
                    raise ValueError(f"index/label allowlist mismatch for sample {row['sample_id']}")
                label_count += 1
            if row_index % 128 == 0 or row_index == len(rows):
                print(f"[audit {split}] checked {row_index}/{len(rows)} indexed samples", flush=True)

        rate_summary = [
            {
                "count": len(rates),
                **({"min": float(np.min(rates)), "median": float(np.median(rates)), "max": float(np.max(rates))} if rates else {}),
            }
            for rates in sampling_rates
        ]
        duration_summary: dict[str, Any] = {"count": len(durations_s)}
        if durations_s:
            duration_summary.update({"min": float(np.min(durations_s)), "median": float(np.median(durations_s)), "max": float(np.max(durations_s))})
        report["splits"][split] = {
            "indexed_samples": len(rows),
            "labels_checked": label_count,
            "iq": {
                "schema": "iq_node0..3: int16 interleaved I,Q; sr_node0..3: one float32",
                "sampling_rate_hz_by_node": rate_summary,
                "capture_duration_s": duration_summary,
            },
            "radar": {
                "schema": "float64 (N,4) [E,N,U,rel_time_s]",
                "total_points_raw": radar_points,
                "causal_points_retained": radar_causal_points,
                "positive_rel_time_points_excluded": positive_time_points,
                "files_with_positive_rel_time": positive_time_files,
                "observed_rel_time_min": None if radar_points == 0 else radar_time_min,
                "observed_rel_time_max": None if radar_points == 0 else radar_time_max,
                "positive_time_policy": "exclude rel_time_s > 0 under documented historical-window contract",
            },
        }
    return report


def compute_fold_statistics(
    data_root: str | Path,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
) -> dict[str, Any]:
    """Compute only train-fold ENU/Radar mean and std for normalization."""
    target_positions = np.concatenate(
        [labels[int(index)]["target_positions"][labels[int(index)]["target_mask"]] for index in train_indices], axis=0
    ).astype(np.float64)
    if not len(target_positions):
        raise ValueError("Fold has no localization targets")
    enu_mean = target_positions.mean(axis=0)
    enu_std = np.maximum(target_positions.std(axis=0), 1.0e-3)

    radar_sum = np.zeros(4, dtype=np.float64)
    radar_square_sum = np.zeros(4, dtype=np.float64)
    radar_count = 0
    root = Path(data_root)
    for index in train_indices:
        row = rows[int(index)]
        values = np.load(indexed_path(root, "train", str(row["radar_npy_relpath"])), allow_pickle=False)
        values = causal_radar_points(values)
        if not len(values):
            continue
        values = np.asarray(values, dtype=np.float64)
        radar_sum += values.sum(axis=0)
        radar_square_sum += np.square(values).sum(axis=0)
        radar_count += len(values)
    if radar_count:
        radar_mean = radar_sum / radar_count
        radar_var = np.maximum(radar_square_sum / radar_count - np.square(radar_mean), 1.0e-12)
        radar_std = np.sqrt(radar_var)
    else:
        radar_mean = np.zeros(4, dtype=np.float64)
        radar_std = np.ones(4, dtype=np.float64)
    return {
        "enu_mean": enu_mean.astype(float).tolist(),
        "enu_std": enu_std.astype(float).tolist(),
        "radar_mean": radar_mean.astype(float).tolist(),
        "radar_std": np.maximum(radar_std, 1.0e-6).astype(float).tolist(),
        "radar_point_count": int(radar_count),
    }


def ensure_runtime_paths(data_root: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(data_root).expanduser()
    output = Path(output_dir).expanduser()
    if not root.is_dir():
        raise PermissionError(f"Cannot read requested data root: {root}")
    output.mkdir(parents=True, exist_ok=True)
    probe = output / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(f"Cannot write requested output directory: {output}") from exc
    return root.resolve(), output.resolve()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=target.parent, suffix=".tmp") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        temp = Path(handle.name)
    os.replace(temp, target)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".tmp") as handle:
        temp = Path(handle.name)
    try:
        torch.save(payload, temp)
        os.replace(temp, target)
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
        return {"decay": self.decay, "shadow": {key: value.cpu() for key, value in self.shadow.items()}}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {key: value.clone() for key, value in state["shadow"].items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, value in self.shadow.items():
            state[key].copy_(value.to(device=state[key].device, dtype=state[key].dtype))
        model.load_state_dict(state, strict=True)


class EMAScope:
    """Temporarily evaluate a model under EMA weights and restore raw weights."""

    def __init__(self, model: torch.nn.Module, ema: EMA | None) -> None:
        self.model = model
        self.ema = ema
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
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
