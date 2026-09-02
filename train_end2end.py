"""One-command 3-query multimodal UAV training, parity and submission pipeline."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from config import ProjectConfig
from cv_folds import make_stratified_folds, save_fold_artifacts
from dataset import IndexedUAVDataset, make_dataloader
from decoder import audit_submission, ensemble_query_predictions, submission_drones, write_submission
from inference import SET_PREDICTION_FORMAT_VERSION, formal_inference_from_checkpoint, load_prediction_archive, run_inference, save_prediction_archive
from losses import SetPredictionLoss
from metrics import evaluate_predictions, robust_selection_score
from models import MultimodalUAVOODNet
from parity_check import run_parity_check
from utils import (
    EMA,
    atomic_json_dump,
    atomic_torch_save,
    audit_dataset,
    compute_fold_statistics,
    ensure_runtime_paths,
    parse_label,
    read_index,
    seed_everything,
    sha256_json,
    tensor_to_device,
)


DEFAULT_DINOV3_REPO_DIR = "/data1/whd/AI_wireless/dinov3-main"
DEFAULT_DINOV3_WEIGHT_PATH = (
    "/data1/whd/AI_wireless/dinov3-main/weights/"
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)


def _validate_dinov3_runtime(enabled: bool) -> None:
    if not enabled:
        return
    if sys.version_info < (3, 10):
        current = ".".join(map(str, sys.version_info[:3]))
        raise RuntimeError(f"DINOv3 ViT-S+/16 requires Python >=3.10; current interpreter is {current}")
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        raise RuntimeError("DINOv3 ViT-S+/16 requires PyTorch >=2.0 (scaled_dot_product_attention is unavailable)")


def _device_from_arg(value: str) -> torch.device:
    return torch.device(value) if value != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cuda_bf16_available() -> bool:
    probe = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(torch.cuda.is_available() and callable(probe) and probe())


def _autocast(device: torch.device, config: ProjectConfig):
    if device.type != "cuda":
        from contextlib import nullcontext
        return nullcontext()
    dtype = torch.bfloat16 if config.optim.bf16 and _cuda_bf16_available() else (torch.float16 if config.optim.fp16 else None)
    if dtype is None:
        from contextlib import nullcontext
        return nullcontext()
    modern = getattr(torch, "autocast", None)
    return modern("cuda", dtype=dtype) if modern is not None else torch.cuda.amp.autocast(dtype=dtype)


def _make_grad_scaler(enabled: bool):
    modern_amp = getattr(torch, "amp", None)
    modern_scaler = getattr(modern_amp, "GradScaler", None)
    if modern_scaler is not None:
        try:
            return modern_scaler("cuda", enabled=enabled)
        except TypeError:
            return modern_scaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _resolve_capacity(config: ProjectConfig, requested: str, device: torch.device) -> str:
    """Choose an attention capacity that fits actual CUDA memory, then freeze it."""
    if requested not in {"auto", "compact", "base", "large"}:
        raise ValueError(f"Unsupported --model-scale {requested}")
    resolved = requested
    if requested == "auto":
        if device.type != "cuda":
            resolved = "compact"
        else:
            gib = torch.cuda.get_device_properties(device).total_memory / float(1024 ** 3)
            resolved = "large" if gib >= 32.0 else ("base" if gib >= 16.0 else "compact")
    presets = {
        "compact": {"dim": 192, "heads": 6, "rf_raw_width": 64, "rf_spec_width": 64, "rf_node_layers": 2, "radar_layers": 4, "decoder_layers": 3},
        "base": {"dim": 256, "heads": 8, "rf_raw_width": 80, "rf_spec_width": 80, "rf_node_layers": 3, "radar_layers": 6, "decoder_layers": 4},
        "large": {"dim": 320, "heads": 8, "rf_raw_width": 96, "rf_spec_width": 96, "rf_node_layers": 3, "radar_layers": 8, "decoder_layers": 6},
    }
    for key, value in presets[resolved].items():
        setattr(config.model, key, value)
    config.model.scale = resolved
    return resolved


class OverallProgress:
    """Visible coarse pipeline progress, independent of nested batch bars."""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = int(total_steps)
        self.bar = tqdm(
            total=self.total_steps,
            desc="overall 0/0",
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )

    def begin(self, name: str) -> None:
        current = self.bar.n + 1
        self.bar.set_description(f"overall {current}/{self.total_steps}")
        self.bar.set_postfix_str(name, refresh=True)

    def complete(self) -> None:
        self.bar.update(1)

    def close(self) -> None:
        if self.bar.n < self.total_steps:
            self.bar.set_postfix_str(f"stopped at {self.bar.n}/{self.total_steps}", refresh=True)
        else:
            self.bar.set_postfix_str("complete", refresh=True)
        self.bar.close()


def _raise_on_nonfinite_loss(loss: torch.Tensor, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> None:
    if bool(torch.isfinite(loss).item()):
        return
    watched = ("objectness_logits", "model_logits", "position_mu", "position_log_sigma")
    summary = {
        key: {"shape": list(outputs[key].shape), "nonfinite": int((~torch.isfinite(outputs[key])).sum().detach().cpu())}
        for key in watched if key in outputs
    }
    raise FloatingPointError(
        "Non-finite training loss; optimizer step blocked. "
        f"sample_ids={batch['sample_id'].detach().cpu().tolist()}, outputs={summary}"
    )


def _raise_on_nonfinite_gradients(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> None:
    invalid = [name for name, parameter in model.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
    if invalid:
        raise FloatingPointError(
            "Non-finite gradients; optimizer step blocked. "
            f"sample_ids={batch['sample_id'].detach().cpu().tolist()}, parameters={invalid[:16]}"
        )


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _checkpoint_payload(
    model: MultimodalUAVOODNet,
    ema: EMA,
    config: ProjectConfig,
    fold: int,
    fold_seed: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    fold_stats: Mapping[str, Any],
    kind: str,
) -> dict[str, Any]:
    return {
        "format_version": SET_PREDICTION_FORMAT_VERSION,
        "architecture": "MultimodalUAVSetPredictionNet/3-unordered-target-queries",
        "kind": kind,
        "is_best": True,
        "fold": int(fold),
        "fold_seed": int(fold_seed),
        "train_indices": np.asarray(train_indices, dtype=np.int64).tolist(),
        "validation_indices": np.asarray(validation_indices, dtype=np.int64).tolist(),
        "project_config": config.to_dict(),
        "fold_stats": dict(fold_stats),
        "model_state": _cpu_state(model),
        "ema_state": ema.state_dict(),
    }


def _make_fold_objects(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: ProjectConfig,
    fold: int,
    training: bool,
) -> tuple[dict[str, Any], IndexedUAVDataset, IndexedUAVDataset]:
    fold_stats = compute_fold_statistics(root, rows, labels, train_indices)
    fold_seed = config.cv.seed + 10_000 * fold
    train_dataset = IndexedUAVDataset(
        root, "train", rows, train_indices, config.data, labels=labels, training=training, seed=fold_seed,
    )
    validation_dataset = IndexedUAVDataset(
        root, "train", rows, validation_indices, config.data, labels=labels, training=False, seed=fold_seed,
    )
    return fold_stats, train_dataset, validation_dataset


def _run_pretraining_parity_probe(
    root: Path,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: ProjectConfig,
    device: torch.device,
    probe_fold: int,
) -> None:
    """Checkpoint round-trip gate before the first optimizer update."""
    probe_dir = output_dir / "pretraining_parity_probe" / f"fold_{probe_fold}"
    checkpoint_path = probe_dir / "probe_best.pt"
    oof_path = probe_dir / "probe_oof.npz"
    report_path = probe_dir / "report.json"
    train_indices, validation_indices = folds[probe_fold]
    fingerprint = sha256_json({
        "format": SET_PREDICTION_FORMAT_VERSION,
        "probe_fold": probe_fold,
        "validation_indices": validation_indices.tolist(),
        "config": config.to_dict(),
    })
    if checkpoint_path.is_file() and oof_path.is_file() and report_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if previous.get("fingerprint") == fingerprint and previous.get("passed"):
                print("[preflight] re-running saved validation-as-test parity probe")
                run_parity_check(root, checkpoint_path, oof_path, device)
                return
        except Exception as exc:
            print(f"[preflight] saved probe is incompatible; rebuilding ({exc})")

    print("[preflight] running validation-as-test parity probe before optimizer training")
    fold_stats, _, validation_dataset = _make_fold_objects(
        root, rows, labels, train_indices, validation_indices, config, probe_fold, training=False,
    )
    fold_seed = config.cv.seed + 10_000 * probe_fold
    seed_everything(fold_seed)
    model = MultimodalUAVOODNet(config.data, config.model, fold_stats).to(device)
    ema = EMA(config.optim.ema_decay)
    ema.initialize(model)
    prediction = run_inference(
        model,
        ema,
        validation_dataset,
        config,
        device,
        progress_desc=f"preflight OOF inference fold {probe_fold}",
    )
    probe_dir.mkdir(parents=True, exist_ok=True)
    save_prediction_archive(prediction, oof_path)
    atomic_torch_save(
        _checkpoint_payload(model, ema, config, probe_fold, fold_seed, train_indices, validation_indices, fold_stats, "pretraining_parity_probe"),
        checkpoint_path,
    )
    report = run_parity_check(root, checkpoint_path, oof_path, device)
    report["fingerprint"] = fingerprint
    atomic_json_dump(report, report_path)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _train_one_fold(
    root: Path,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: ProjectConfig,
    fold: int,
    device: torch.device,
) -> tuple[Path, Path]:
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_dir / "best.pt"
    oof_path = fold_dir / "oof_best.npz"
    fold_seed = config.cv.seed + 10_000 * fold
    seed_everything(fold_seed)
    fold_stats, train_dataset, validation_dataset = _make_fold_objects(
        root, rows, labels, train_indices, validation_indices, config, fold, training=True,
    )
    atomic_json_dump({
        "fold": fold,
        "fold_seed": fold_seed,
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "fold_stats": fold_stats,
    }, fold_dir / "fold_artifacts.json")

    model = MultimodalUAVOODNet(config.data, config.model, fold_stats).to(device)
    criterion = SetPredictionLoss(config.loss, fold_stats).to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
    train_loader = make_dataloader(train_dataset, config.optim, training=True)
    updates_per_epoch = math.ceil(len(train_loader) / max(config.optim.accumulation_steps, 1))
    total_updates = max(1, updates_per_epoch * config.optim.epochs)
    warmup_updates = max(1, int(total_updates * config.optim.warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup_updates:
            return float(step + 1) / warmup_updates
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = LambdaLR(optimizer, schedule)
    use_scaler = device.type == "cuda" and config.optim.fp16 and not (config.optim.bf16 and _cuda_bf16_available())
    scaler = _make_grad_scaler(use_scaler)
    ema = EMA(config.optim.ema_decay)
    ema.initialize(model)
    best_score = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(config.optim.epochs):
        model.train()
        train_dataset.set_epoch(epoch)
        epoch_logs: list[dict[str, float]] = []
        progress = tqdm(train_loader, desc=f"fold {fold} epoch {epoch + 1}/{config.optim.epochs}", leave=False)
        for step, batch in enumerate(progress):
            batch = tensor_to_device(batch, device)
            with _autocast(device, config):
                outputs = model(batch)
                loss, logs = criterion(outputs, batch)
            _raise_on_nonfinite_loss(loss, outputs, batch)
            scaler.scale(loss / max(config.optim.accumulation_steps, 1)).backward()
            should_step = (step + 1) % config.optim.accumulation_steps == 0 or step + 1 == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                _raise_on_nonfinite_gradients(model, batch)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
            epoch_logs.append(logs)
            progress.set_postfix(loss=f"{logs['loss']:.3f}", dist=f"{logs['train_3d_mean']:.1f}")

        validation = run_inference(
            model,
            ema,
            validation_dataset,
            config,
            device,
            progress_desc=f"fold {fold} validation epoch {epoch + 1}",
        )
        metrics, _ = evaluate_predictions(validation, config.loss.decode_confidence_threshold)
        score = robust_selection_score(metrics, config.loss)
        train_mean = {key: float(np.mean([item[key] for item in epoch_logs])) for key in epoch_logs[0]} if epoch_logs else {}
        record = {
            "epoch": epoch + 1,
            "selection_score": score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_mean,
            "validation": metrics,
        }
        history.append(record)
        atomic_json_dump(history, fold_dir / "history.json")
        localization = metrics["position_query_hungarian"]
        print(
            f"[fold {fold}] epoch={epoch + 1} selection={score:.5f} "
            f"F1={metrics['classification']['micro_f1']:.4f} macroF1={metrics['classification']['macro_f1']:.4f} "
            f"3Dmean={localization['3d_mean']:.2f} median={localization['3d_median']:.2f} "
            f"P90={localization['3d_p90']:.2f} P95={localization['3d_p95']:.2f}"
        )
        # The complete required metrics are printed and saved every epoch, not
        # reduced to an opaque score or an unavailable official formula.
        print("[fold-metrics] " + json.dumps(metrics, ensure_ascii=False, allow_nan=True))

        if score > best_score:
            best_score, stale = score, 0
            payload = _checkpoint_payload(
                model, ema, config, fold, fold_seed, train_indices, validation_indices, fold_stats, "trained_best",
            )
            payload.update({"best_epoch": epoch + 1, "selection_score": score})
            atomic_torch_save(payload, checkpoint_path)
            save_prediction_archive(validation, oof_path)
            atomic_json_dump(metrics, fold_dir / "best_metrics.json")
        else:
            stale += 1
            if stale >= config.optim.early_stopping_patience:
                print(f"[fold {fold}] early stopping at epoch {epoch + 1}")
                break

    if not checkpoint_path.is_file() or not oof_path.is_file():
        raise RuntimeError(f"Fold {fold} did not save a best checkpoint/OOF archive")
    parity = run_parity_check(root, checkpoint_path, oof_path, device)
    atomic_json_dump(parity, fold_dir / "parity.json")
    del model, criterion, optimizer, train_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_path, oof_path


def _assemble_oof(oof_paths: Sequence[Path], rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not oof_paths:
        raise ValueError("No OOF paths")
    first = load_prediction_archive(oof_paths[0])
    indexed_keys = [key for key, value in first.items() if isinstance(value, np.ndarray) and key not in {"sample_id", "source_index"}]
    total = len(rows)
    result = {
        key: np.zeros((total, *np.asarray(first[key]).shape[1:]), dtype=np.asarray(first[key]).dtype)
        for key in indexed_keys
    }
    filled = np.zeros(total, dtype=bool)
    for path in oof_paths:
        chunk = load_prediction_archive(path)
        indices = chunk["source_index"].astype(np.int64)
        if filled[indices].any():
            raise RuntimeError(f"Overlapping OOF validation indices in {path}")
        if set(indexed_keys) - set(chunk):
            raise RuntimeError(f"OOF archive fields differ: {path}")
        for key in indexed_keys:
            result[key][indices] = chunk[key]
        filled[indices] = True
    if not filled.all():
        raise RuntimeError(f"OOF misses {int((~filled).sum())} indexed train samples")
    result["sample_id"] = np.asarray([int(row["sample_id"]) for row in rows], dtype=np.int64)
    result["source_index"] = np.arange(total, dtype=np.int64)
    return result


def _run_test_ensemble(
    root: Path,
    output_dir: Path,
    checkpoint_paths: Sequence[Path],
    test_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    indices = np.arange(len(test_rows), dtype=np.int64)
    predictions: list[dict[str, np.ndarray]] = []
    loaded_configs: list[ProjectConfig] = []
    for fold_index, checkpoint_path in enumerate(checkpoint_paths):
        prediction, loaded = formal_inference_from_checkpoint(
            checkpoint_path,
            root,
            "test_public",
            test_rows,
            indices,
            device,
            labels=None,
            progress_desc=f"test inference fold {fold_index + 1}/{len(checkpoint_paths)}",
        )
        predictions.append(prediction)
        loaded_configs.append(loaded.config)
    reference = predictions[0]
    for current in predictions[1:]:
        for key in ("sample_id", "source_index", "allowlist_mask"):
            if not np.array_equal(reference[key], current[key]):
                raise RuntimeError(f"Fold test inference mismatch in {key}")
    threshold = loaded_configs[0].loss.decode_confidence_threshold
    if any(config.loss.decode_confidence_threshold != threshold for config in loaded_configs[1:]):
        raise RuntimeError("Fold checkpoints have inconsistent decode confidence thresholds")
    decoded, ensemble = ensemble_query_predictions(
        predictions, reference["allowlist_mask"], threshold,
    )
    drones = submission_drones(decoded)
    submission_path = output_dir / "submission.jsonl"
    write_submission(reference["sample_id"], drones, submission_path)
    audit = audit_submission(submission_path, test_rows)
    ensemble.update({
        "sample_id": reference["sample_id"],
        "source_index": reference["source_index"],
        "allowlist_mask": reference["allowlist_mask"],
    })
    save_prediction_archive(ensemble, output_dir / "test_ensemble_predictions.npz")
    atomic_json_dump({
        "fold_checkpoints": [str(path) for path in checkpoint_paths],
        "decode_confidence_threshold": threshold,
        "position_ensemble_mode": "uncertainty_weighted_mean",
        **audit,
    }, output_dir / "submission_audit.json")
    print(f"[submission] {audit}")
    return {"submission": str(submission_path), "audit": audit}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3-query multimodal UAV set prediction and generate a parity-gated submission")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-scale", choices=("auto", "compact", "base", "large"), default="auto")
    parser.add_argument("--only-fold", type=int, default=None, help="Train one fold, preserving its mandatory parity gate")
    parser.add_argument("--single-fold-submission", action="store_true", help="With --only-fold, create a one-checkpoint test submission after parity")
    parser.add_argument("--disable-eo", action="store_true", help="Ablate EO cleanly; no DINO repository/weight is loaded")
    parser.add_argument("--eo-pretrained-path", default=DEFAULT_DINOV3_WEIGHT_PATH)
    parser.add_argument("--dinov3-repo-dir", default=DEFAULT_DINOV3_REPO_DIR)
    parser.add_argument("--eo-train-last-blocks", type=int, default=0)
    parser.add_argument("--decode-confidence-threshold", type=float, default=None)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--reuse-folds", action="store_true", help="Reuse only checkpoints that pass the current formal parity gate")
    parser.add_argument("--no-strict-counts", action="store_true", help="Diagnostic-only override for non-official data copies")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be >=2 to provide a held-out validation fold")
    if args.single_fold_submission and args.only_fold is None:
        raise ValueError("--single-fold-submission requires --only-fold K")
    if args.decode_confidence_threshold is not None and not 0.0 <= args.decode_confidence_threshold <= 1.0:
        raise ValueError("--decode-confidence-threshold must be in [0,1]")

    root, output_dir = ensure_runtime_paths(args.data_root, args.output_dir)
    device = _device_from_arg(args.device)
    config = ProjectConfig()
    config.cv.folds = args.folds
    config.cv.seed = args.seed
    config.optim.epochs = args.epochs
    config.optim.batch_size = args.batch_size
    config.optim.eval_batch_size = args.eval_batch_size
    config.optim.workers = args.workers
    config.model.use_eo = not args.disable_eo
    config.model.eo_pretrained_path = str(Path(args.eo_pretrained_path).expanduser().resolve()) if args.eo_pretrained_path else ""
    config.model.dinov3_repo_dir = str(Path(args.dinov3_repo_dir).expanduser().resolve()) if args.dinov3_repo_dir else ""
    config.model.eo_train_last_blocks = args.eo_train_last_blocks
    if args.decode_confidence_threshold is not None:
        config.loss.decode_confidence_threshold = args.decode_confidence_threshold
    resolved_scale = _resolve_capacity(config, args.model_scale, device)
    print(f"[runtime] device={device}, output={output_dir}, model_scale={resolved_scale}, EO={'on' if config.model.use_eo else 'off'}")

    selected_fold_count = 1 if args.only_fold is not None else args.folds
    if args.parity_only:
        total_steps = 3  # audit, fold construction, parity
    elif args.only_fold is not None:
        total_steps = 4 + (2 if args.single_fold_submission else 0)
    else:
        total_steps = selected_fold_count + 6  # audit, CV, preflight, folds, OOF, final parity, submission
    overall = OverallProgress(total_steps)
    try:
        overall.begin("indexed data audit")
        audit = audit_dataset(root, strict_counts=not args.no_strict_counts)
        atomic_json_dump(audit, output_dir / "data_audit.json")
        print("[audit] " + json.dumps({
            "train": audit["splits"]["train"]["indexed_samples"],
            "test_public": audit["splits"]["test_public"]["indexed_samples"],
            "description": audit["description"],
        }, ensure_ascii=False))
        _validate_dinov3_runtime(config.model.use_eo)
        overall.complete()

        overall.begin("constructing CV folds")
        rows = read_index(root, "train")
        test_rows = read_index(root, "test_public")
        labels = [parse_label(root, "train", row) for row in rows]
        folds, fold_report = make_stratified_folds(rows, labels, config.cv.folds, config.cv.seed)
        save_fold_artifacts(folds, fold_report, output_dir)
        print("[cv] " + json.dumps({"method": fold_report["method"], "folds": len(folds)}, ensure_ascii=False))
        if args.only_fold is not None and not 0 <= args.only_fold < len(folds):
            raise ValueError(f"--only-fold must be in [0,{len(folds) - 1}]")
        overall.complete()

        if args.parity_only:
            parity_fold = 0 if args.only_fold is None else args.only_fold
            overall.begin(f"parity-only fold {parity_fold}")
            checkpoint_path = output_dir / f"fold_{parity_fold}" / "best.pt"
            oof_path = output_dir / f"fold_{parity_fold}" / "oof_best.npz"
            if not checkpoint_path.is_file() or not oof_path.is_file():
                raise FileNotFoundError("--parity-only needs fold_K/best.pt and fold_K/oof_best.npz in output-dir")
            report = run_parity_check(root, checkpoint_path, oof_path, device)
            atomic_json_dump(report, output_dir / "parity_only_report.json")
            overall.complete()
            return

        probe_fold = args.only_fold if args.only_fold is not None else 0
        overall.begin(f"preflight parity probe fold {probe_fold}")
        _run_pretraining_parity_probe(root, output_dir, rows, labels, folds, config, device, probe_fold)
        overall.complete()
        checkpoints: list[Path] = []
        oof_paths: list[Path] = []
        fold_iterator = [(args.only_fold, folds[args.only_fold])] if args.only_fold is not None else list(enumerate(folds))
        for fold, (train_indices, validation_indices) in fold_iterator:
            overall.begin(f"fold {fold}: train, validate, OOF and parity")
            checkpoint_path = output_dir / f"fold_{fold}" / "best.pt"
            oof_path = output_dir / f"fold_{fold}" / "oof_best.npz"
            if args.reuse_folds and checkpoint_path.is_file() and oof_path.is_file():
                print(f"[fold {fold}] reusing only after formal parity passes")
                parity = run_parity_check(root, checkpoint_path, oof_path, device)
                atomic_json_dump(parity, checkpoint_path.parent / "parity_reuse.json")
            else:
                checkpoint_path, oof_path = _train_one_fold(
                    root, output_dir, rows, labels, train_indices, validation_indices, config, fold, device,
                )
            checkpoints.append(checkpoint_path)
            oof_paths.append(oof_path)
            overall.complete()

        if args.only_fold is not None:
            report: dict[str, Any] = {
                "fold": args.only_fold,
                "checkpoint": str(checkpoints[0]),
                "oof": str(oof_paths[0]),
                "status": "single_fold_complete_no_submission",
            }
            if not args.single_fold_submission:
                atomic_json_dump(report, output_dir / f"fold_{args.only_fold}" / "single_fold_run.json")
                print(f"[single-fold] fold {args.only_fold} complete; parity passed; no incomplete-OOF aggregation/submission was created")
                return
            overall.begin("single-fold final parity")
            final_parity = run_parity_check(root, checkpoints[0], oof_paths[0], device)
            atomic_json_dump(final_parity, output_dir / "single_fold_pretest_parity.json")
            overall.complete()
            overall.begin("single-fold test inference and submission")
            report["status"] = "single_fold_submission_complete"
            report["submission"] = _run_test_ensemble(root, output_dir, checkpoints, test_rows, device)
            atomic_json_dump(report, output_dir / f"fold_{args.only_fold}" / "single_fold_run.json")
            overall.complete()
            return

        overall.begin("assemble and evaluate complete OOF")
        oof = _assemble_oof(oof_paths, rows)
        oof_metrics, _ = evaluate_predictions(oof, config.loss.decode_confidence_threshold)
        save_prediction_archive(oof, output_dir / "oof_complete.npz")
        atomic_json_dump(oof_metrics, output_dir / "oof_metrics.json")
        print("[OOF] " + json.dumps(oof_metrics, ensure_ascii=False, allow_nan=True))
        overall.complete()
        overall.begin("final pre-test parity")
        final_parity = run_parity_check(root, checkpoints[0], oof_paths[0], device)
        atomic_json_dump(final_parity, output_dir / "final_pretest_parity.json")
        overall.complete()
        overall.begin("five-fold test inference and submission")
        _run_test_ensemble(root, output_dir, checkpoints, test_rows, device)
        overall.complete()
    finally:
        overall.close()


if __name__ == "__main__":
    main()
