"""The sole inference path for validation, parity checking and public test."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from config import ProjectConfig
from dataset import IndexedUAVDataset, make_dataloader
from models import MultimodalUAVOODNet
from utils import EMA, EMAScope, tensor_to_device


SET_PREDICTION_FORMAT_VERSION = 20


@dataclass
class LoadedFold:
    model: MultimodalUAVOODNet
    ema: EMA
    config: ProjectConfig
    checkpoint: dict[str, Any]


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_fold_checkpoint(path: str | Path, device: torch.device) -> LoadedFold:
    checkpoint = _torch_load(Path(path), device)
    if int(checkpoint.get("format_version", 0)) != SET_PREDICTION_FORMAT_VERSION:
        raise RuntimeError(
            "Checkpoint is not the current unordered 3-query set-prediction format. "
            "Old fixed-slot/calibration checkpoints must not be reused."
        )
    if not checkpoint.get("is_best", False):
        raise RuntimeError(f"Checkpoint is not marked as the best validation epoch: {path}")
    config = ProjectConfig.from_dict(checkpoint["project_config"])
    model = MultimodalUAVOODNet(config.data, config.model, checkpoint["fold_stats"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    ema = EMA(config.optim.ema_decay)
    ema.load_state_dict(checkpoint["ema_state"])
    return LoadedFold(model=model, ema=ema, config=config, checkpoint=checkpoint)


def _autocast_context(device: torch.device, config: ProjectConfig) -> contextlib.AbstractContextManager[Any]:
    if device.type != "cuda":
        return contextlib.nullcontext()
    probe = getattr(torch.cuda, "is_bf16_supported", None)
    use_bf16 = bool(config.optim.bf16 and callable(probe) and probe())
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if config.optim.fp16 else None)
    if dtype is None:
        return contextlib.nullcontext()
    modern = getattr(torch, "autocast", None)
    return modern(device_type="cuda", dtype=dtype) if modern is not None else torch.cuda.amp.autocast(dtype=dtype)


@torch.no_grad()
def run_inference(
    model: MultimodalUAVOODNet,
    ema: EMA | None,
    dataset: IndexedUAVDataset,
    config: ProjectConfig,
    device: torch.device,
    progress_desc: str = "inference",
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Run deterministic EMA/model.eval inference using the common Dataset."""
    output_parts: dict[str, list[np.ndarray]] = {
        "objectness_logits": [], "model_logits": [], "position_mu": [], "position_log_sigma": [],
    }
    metadata: dict[str, list[np.ndarray]] = {
        "sample_id": [], "source_index": [], "allowlist_mask": [],
        "target_model_ids": [], "target_positions": [], "target_mask": [],
    }
    with EMAScope(model, ema):
        loader = make_dataloader(dataset, config.optim, training=False)
        iterator = tqdm(
            loader,
            desc=progress_desc,
            total=len(loader),
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for batch in iterator:
            for key in metadata:
                metadata[key].append(batch[key].cpu().numpy())
            device_batch = tensor_to_device(batch, device)
            with _autocast_context(device, config):
                output = model(device_batch)
            for key in output_parts:
                value = output[key].float()
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Non-finite inference output: {key}")
                output_parts[key].append(value.cpu().numpy())
    result = {key: np.concatenate(values, axis=0).astype(np.float32) for key, values in output_parts.items()}
    result["query_objectness_prob"] = (1.0 / (1.0 + np.exp(-result["objectness_logits"]))).astype(np.float32)
    shifted = result["model_logits"] - result["model_logits"].max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    result["query_model_prob"] = (probability / np.maximum(probability.sum(axis=-1, keepdims=True), 1.0e-12)).astype(np.float32)
    for key, values in metadata.items():
        value = np.concatenate(values, axis=0)
        result[key] = value.astype(np.float32) if value.dtype.kind == "f" else value
    return result


def formal_inference_from_checkpoint(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: torch.device,
    labels: Sequence[Mapping[str, Any]] | None = None,
    progress_desc: str | None = None,
) -> tuple[dict[str, np.ndarray], LoadedFold]:
    """Formal entry point shared without forked validation/test code."""
    loaded = load_fold_checkpoint(checkpoint_path, device)
    dataset = IndexedUAVDataset(
        data_root=data_root,
        split=split,
        rows=rows,
        indices=indices,
        data_cfg=loaded.config.data,
        labels=labels,
        training=False,
        seed=int(loaded.checkpoint["fold_seed"]),
    )
    description = progress_desc or f"formal {split} inference fold {loaded.checkpoint.get('fold', '?')}"
    return run_inference(loaded.model, loaded.ema, dataset, loaded.config, device, progress_desc=description), loaded


def save_prediction_archive(prediction: Mapping[str, np.ndarray], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{key: value for key, value in prediction.items() if isinstance(value, np.ndarray)})


def load_prediction_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
