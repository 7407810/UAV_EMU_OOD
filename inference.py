"""The only inference path used for validation, parity checks and public test."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from config import ProjectConfig
from dataset import IndexedUAVDataset, make_dataloader
from models import MultimodalUAVOODNet
from utils import EMA, EMAScope, tensor_to_device


@dataclass
class LoadedFold:
    model: MultimodalUAVOODNet
    ema: EMA
    config: ProjectConfig
    checkpoint: dict[str, Any]


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Older PyTorch
        return torch.load(path, map_location=device)


def load_fold_checkpoint(path: str | Path, device: torch.device) -> LoadedFold:
    checkpoint = _torch_load(Path(path), device)
    if int(checkpoint.get("format_version", 0)) != 6:
        raise RuntimeError(
            "Checkpoint predates causal Radar filtering and the current numerically stable EO/calibration contract. "
            "It cannot be reused; retrain this fold from scratch."
        )
    config = ProjectConfig.from_dict(checkpoint["project_config"])
    model = MultimodalUAVOODNet(config.data, config.model, checkpoint["calibration_init"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    ema = EMA(config.optim.ema_decay)
    ema.load_state_dict(checkpoint["ema_state"])
    if not checkpoint.get("is_best", False):
        raise RuntimeError(f"Checkpoint is not marked as best epoch: {path}")
    return LoadedFold(model=model, ema=ema, config=config, checkpoint=checkpoint)


def _autocast_context(device: torch.device, config: ProjectConfig) -> contextlib.AbstractContextManager[Any]:
    if device.type != "cuda":
        return contextlib.nullcontext()
    bf16_probe = getattr(torch.cuda, "is_bf16_supported", None)
    use_bf16 = bool(config.optim.bf16 and callable(bf16_probe) and bf16_probe())
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if config.optim.fp16 else None)
    if dtype is not None:
        modern = getattr(torch, "autocast", None)
        if modern is not None:
            return modern(device_type="cuda", dtype=dtype)
        return torch.cuda.amp.autocast(dtype=dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def run_inference(
    model: MultimodalUAVOODNet,
    ema: EMA | None,
    dataset: IndexedUAVDataset,
    config: ProjectConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """The single deterministic inference path for validation and public test."""
    current: dict[str, list[np.ndarray]] = {"presence_logits": [], "count_prob": [], "position_pred": []}
    ids, source, allow = [], [], []
    target_parts: dict[str, list[np.ndarray]] = {"presence_target": [], "position_target": [], "count_target": []}
    with EMAScope(model, ema):
        loader = make_dataloader(dataset, config.optim, training=False)
        for batch in loader:
            ids.append(batch["sample_id"].numpy())
            source.append(batch["source_index"].numpy())
            allow.append(batch["allowlist_mask"].numpy())
            for key in target_parts:
                target_parts[key].append(batch[key].numpy())
            batch = tensor_to_device(batch, device)
            with _autocast_context(device, config):
                output = model(batch)
            current["presence_logits"].append(output["presence_logits"].float().cpu().numpy())
            current["count_prob"].append(torch.softmax(output["count_logits"].float(), dim=-1).cpu().numpy())
            current["position_pred"].append(output["position_pred"].float().cpu().numpy())
    result = {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in current.items()}
    result["presence_prob"] = (1.0 / (1.0 + np.exp(-result["presence_logits"]))).astype(np.float32)
    result.update({
        "sample_id": np.concatenate(ids), "source_index": np.concatenate(source),
        "allowlist_mask": np.concatenate(allow).astype(np.float32),
    })
    result.update({
        key: np.concatenate(value, axis=0).astype(np.float32) if np.concatenate(value, axis=0).dtype.kind == "f" else np.concatenate(value, axis=0)
        for key, value in target_parts.items()
    })
    return result


def formal_inference_from_checkpoint(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    node_enu: np.ndarray,
    device: torch.device,
    labels: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, np.ndarray], LoadedFold]:
    """Formal checkpoint entry point shared by parity validation and test inference."""
    loaded = load_fold_checkpoint(checkpoint_path, device)
    checkpoint_node_enu = np.asarray(loaded.checkpoint["node_enu"], dtype=np.float32)
    if not np.allclose(checkpoint_node_enu, node_enu, rtol=0.0, atol=1e-6):
        raise RuntimeError("RF node ENU differs from checkpoint training geometry")
    dataset = IndexedUAVDataset(
        data_root=data_root, split=split, rows=rows, indices=indices, node_enu=node_enu,
        data_cfg=loaded.config.data, labels=labels,
        training=False, seed=int(loaded.checkpoint["fold_seed"]),
    )
    return run_inference(loaded.model, loaded.ema, dataset, loaded.config, device), loaded


def save_prediction_archive(prediction: Mapping[str, np.ndarray], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{key: value for key, value in prediction.items() if isinstance(value, np.ndarray)})


def load_prediction_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
