"""One-command end-to-end OOD CV training and submission pipeline.

The pipeline intentionally refuses to create a submission if the saved OOF and
formal checkpoint inference are not numerically identical within parity limits.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from config import ProjectConfig
from cv_session import (
    build_trajectory_sessions,
    cv_credibility_diagnostics,
    make_stratified_group_folds,
    precompute_radar_fingerprints,
    radar_only_calibration_metadata,
    save_session_artifacts,
)
from dataset import IndexedUAVDataset, make_dataloader
from decoder import audit_submission, decode_fixed_slots, write_submission
from inference import formal_inference_from_checkpoint, run_inference, save_prediction_archive
from losses import UAVLoss
from metrics import evaluate_predictions, robust_selection_score
from models import MultimodalUAVOODNet
from parity_check import run_parity_check
from utils import (
    EMA,
    atomic_json_dump,
    atomic_torch_save,
    audit_dataset,
    ensure_runtime_paths,
    load_node_enu,
    parse_label,
    read_index,
    robust_location_stats,
    seed_everything,
    sha256_json,
    tensor_to_device,
)


DEFAULT_DINOV3_REPO_DIR = "/data1/whd/AI_wireless/dinov3-main"
DEFAULT_DINOV3_WEIGHT_PATH = (
    "/data1/whd/AI_wireless/dinov3-main/weights/"
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)


def _validate_dinov3_runtime() -> None:
    """Fail before data audit when the external DINOv3 code cannot import."""
    if sys.version_info < (3, 10):
        current = ".".join(map(str, sys.version_info[:3]))
        raise RuntimeError(
            "DINOv3 ViT-S+/16 distilled requires Python >= 3.10 because the "
            f"official repository uses PEP-604 type annotations. Current Python is {current}. "
            "Create/use a Python 3.10+ environment, then rerun the identical command."
        )
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        raise RuntimeError(
            "DINOv3 ViT-S+/16 distilled requires PyTorch >= 2.0 "
            "(torch.nn.functional.scaled_dot_product_attention is unavailable)."
        )


def _device_from_arg(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cuda_bf16_available() -> bool:
    probe = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(torch.cuda.is_available() and callable(probe) and probe())


def _autocast(device: torch.device, config: ProjectConfig):
    if device.type != "cuda":
        from contextlib import nullcontext
        return nullcontext()
    dtype = torch.bfloat16 if config.optim.bf16 and _cuda_bf16_available() else (torch.float16 if config.optim.fp16 else None)
    if dtype is not None:
        modern = getattr(torch, "autocast", None)
        if modern is not None:
            return modern("cuda", dtype=dtype)
        return torch.cuda.amp.autocast(dtype=dtype)
    from contextlib import nullcontext
    return nullcontext()


def _make_grad_scaler(enabled: bool):
    """Support both torch>=2 AMP and legacy torch.cuda.amp on competition servers."""
    modern_amp = getattr(torch, "amp", None)
    modern_scaler = getattr(modern_amp, "GradScaler", None)
    if modern_scaler is not None:
        try:
            return modern_scaler("cuda", enabled=enabled)
        except TypeError:
            return modern_scaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _raise_on_nonfinite_training_loss(
    loss: torch.Tensor,
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    model: MultimodalUAVOODNet,
) -> None:
    """Fail at the originating batch instead of masking numerical failure in decoding."""
    if bool(torch.isfinite(loss).item()):
        return
    watched = ("presence_logits", "position_pred", "count_logits", "radar_anchor")
    summary = {
        key: {
            "shape": list(outputs[key].shape),
            "nonfinite": int((~torch.isfinite(outputs[key])).sum().detach().cpu()),
        }
        for key in watched
        if key in outputs
    }
    sample_ids = batch["sample_id"].detach().cpu().tolist()
    raise FloatingPointError(
        "Non-finite training loss; optimizer step was blocked. "
        f"sample_ids={sample_ids}, outputs={summary}, calibration={model.calibration_state()}"
    )


def _raise_on_nonfinite_gradients(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> None:
    bad = [name for name, parameter in model.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
    if bad:
        sample_ids = batch["sample_id"].detach().cpu().tolist()
        raise FloatingPointError(
            "Non-finite gradients; optimizer step was blocked. "
            f"sample_ids={sample_ids}, parameters={bad[:12]}"
        )


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _fold_checkpoint_payload(
    model: MultimodalUAVOODNet,
    ema: EMA,
    config: ProjectConfig,
    fold: int,
    fold_seed: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    node_enu: np.ndarray,
    fold_stats: Mapping[str, Any],
    calibration_init: Mapping[str, Any],
    kind: str,
) -> dict[str, Any]:
    return {
        # v7 makes sparse Radar temporal statistics differentiable at zero
        # variance and uses deterministic physical association weights.
        "format_version": 7,
        "kind": kind,
        "is_best": True,
        "fold": int(fold),
        "fold_seed": int(fold_seed),
        "train_indices": train_indices.astype(np.int64).tolist(),
        "val_indices": val_indices.astype(np.int64).tolist(),
        "project_config": config.to_dict(),
        "node_enu": np.asarray(node_enu, dtype=np.float32).tolist(),
        "fold_stats": dict(fold_stats),
        "calibration_init": dict(calibration_init),
        "calibration_model_state": model.calibration_state(),
        "model_state": _cpu_state(model),
        "ema_state": ema.state_dict(),
    }


def _make_fold_objects(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    node_enu: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    config: ProjectConfig,
    fold: int,
    training: bool,
) -> tuple[dict[str, Any], dict[str, Any], IndexedUAVDataset, IndexedUAVDataset]:
    fold_stats = robust_location_stats([labels[int(index)] for index in train_indices])
    train_presence = np.stack([labels[int(index)]["presence"] for index in train_indices])
    positive_rate = train_presence.mean(axis=0)
    fold_stats["class_positive_rate"] = positive_rate.astype(float).tolist()
    fold_stats["class_positive_weights"] = np.clip(np.sqrt((1.0 - positive_rate) / np.maximum(positive_rate, 1e-4)), 1.0, 4.0).astype(float).tolist()
    # Never derive Radar calibration from GT centroids.  This contains only a
    # fold-local, Radar-input scale; yaw/translation start at exact identity and
    # learn inside the common end-to-end graph.
    calibration_init = radar_only_calibration_metadata(root, rows, train_indices)
    fold_seed = config.session.seed + 10_000 * fold
    train_dataset = IndexedUAVDataset(
        root, "train", rows, train_indices, node_enu, config.data, labels,
        training=training, seed=fold_seed,
    )
    validation_dataset = IndexedUAVDataset(
        root, "train", rows, val_indices, node_enu, config.data, labels,
        training=False, seed=fold_seed,
    )
    return fold_stats, calibration_init, train_dataset, validation_dataset


def _run_pretraining_parity_probe(
    root: Path,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    node_enu: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: ProjectConfig,
    device: torch.device,
    probe_fold: int = 0,
) -> None:
    """A real checkpoint round-trip before any optimizer step on a fresh run."""
    probe_dir = output_dir / "pretraining_parity_probe" / f"fold_{probe_fold}"
    checkpoint_path, oof_path = probe_dir / "probe_best.pt", probe_dir / "probe_oof.npz"
    session_fingerprint = sha256_json({
        "probe_fold": probe_fold,
        "validation_indices": folds[probe_fold][1].tolist(),
        "config": config.to_dict(),
        "radar_preprocess_revision": 7,
    })
    report_path = probe_dir / "report.json"
    if checkpoint_path.is_file() and oof_path.is_file() and report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("session_fingerprint") == session_fingerprint and previous.get("passed"):
            print("[preflight] re-running saved pre-training formal parity probe")
            run_parity_check(root, checkpoint_path, oof_path, device)
            return
    print("[preflight] running validation-as-test parity probe before optimizer training")
    train_idx, val_idx = folds[probe_fold]
    stats, calibration_init, _, val_dataset = _make_fold_objects(
        root, rows, labels, node_enu, train_idx, val_idx, config, probe_fold, training=False,
    )
    probe_seed = config.session.seed + 10_000 * probe_fold
    seed_everything(probe_seed)
    model = MultimodalUAVOODNet(config.data, config.model, calibration_init).to(device)
    ema = EMA(config.optim.ema_decay)
    ema.initialize(model)
    prediction = run_inference(model, ema, val_dataset, config, device)
    probe_dir.mkdir(parents=True, exist_ok=True)
    save_prediction_archive(prediction, oof_path)
    payload = _fold_checkpoint_payload(
        model, ema, config, probe_fold, probe_seed, train_idx, val_idx, node_enu, stats,
        calibration_init, "pretraining_parity_probe",
    )
    atomic_torch_save(payload, checkpoint_path)
    result = run_parity_check(root, checkpoint_path, oof_path, device)
    result["session_fingerprint"] = session_fingerprint
    atomic_json_dump(result, report_path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _train_one_fold(
    root: Path,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    node_enu: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    oof_radar_nn: np.ndarray,
    config: ProjectConfig,
    fold: int,
    device: torch.device,
) -> tuple[Path, Path]:
    fold_dir = output_dir / f"fold_{fold}"
    checkpoint_path, oof_path = fold_dir / "best.pt", fold_dir / "oof_best.npz"
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_seed = config.session.seed + 10_000 * fold
    seed_everything(fold_seed)
    stats, calibration_init, train_dataset, val_dataset = _make_fold_objects(
        root, rows, labels, node_enu, train_indices, val_indices, config, fold, training=True,
    )
    atomic_json_dump({
        "fold": fold, "fold_seed": fold_seed, "fold_stats": stats,
        "calibration_init": calibration_init, "train_count": len(train_indices), "validation_count": len(val_indices),
    }, fold_dir / "fold_artifacts.json")
    model = MultimodalUAVOODNet(config.data, config.model, calibration_init).to(device)
    criterion = UAVLoss(config.loss, stats).to(device)
    calibration_parameters = list(model.radar.calibration.parameters())
    calibration_ids = {id(parameter) for parameter in calibration_parameters}
    network_parameters = [parameter for parameter in model.parameters() if id(parameter) not in calibration_ids]
    # The learned global frame must converge before EMA becomes the evaluation
    # path.  This is an optimization multiplier only; it supplies no yaw/ENU
    # value and calibration remains fully data-driven and differentiable.
    optimizer = AdamW([
        {"params": network_parameters, "lr": config.optim.lr, "weight_decay": config.optim.weight_decay},
        {
            "params": calibration_parameters,
            "lr": config.optim.lr * config.optim.calibration_lr_multiplier,
            "weight_decay": 0.0,
        },
    ])
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
    best_score, stale, history = -float("inf"), 0, []
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
            _raise_on_nonfinite_training_loss(loss, outputs, batch, model)
            loss = loss / max(config.optim.accumulation_steps, 1)
            scaler.scale(loss).backward()
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
        validation = run_inference(model, ema, val_dataset, config, device)
        metrics, _ = evaluate_predictions(
            validation["presence_logits"], validation["count_prob"], validation["position_pred"], validation["allowlist_mask"],
            validation["presence_target"], validation["position_target"], oof_radar_nn[validation["source_index"]],
        )
        score = robust_selection_score(metrics)
        logs_mean = {key: float(np.mean([entry[key] for entry in epoch_logs])) for key in epoch_logs[0]} if epoch_logs else {}
        record = {
            "epoch": epoch + 1, "selection_score": score, "train": logs_mean,
            "validation": metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
            # Raw training weights are recorded for calibration diagnostics;
            # validation/inference itself still uses EMA through run_inference.
            "calibration_live": model.calibration_state(),
        }
        history.append(record)
        atomic_json_dump(history, fold_dir / "history.json")
        print(
            f"[fold {fold}] epoch={epoch + 1} score={score:.5f} "
            f"F1={metrics['classification']['micro_f1']:.4f} "
            f"3Dmedian={metrics['position_all_gt_slots']['3d_median']:.2f} "
            f"P90={metrics['position_all_gt_slots']['3d_p90']:.2f} "
            f"calib_yaw={record['calibration_live']['yaw_rad']:.3f} "
            f"calib_t=({record['calibration_live']['tx']:.1f},"
            f"{record['calibration_live']['ty']:.1f},{record['calibration_live']['tz']:.1f})"
        )
        if score > best_score:
            best_score, stale = score, 0
            payload = _fold_checkpoint_payload(
                model, ema, config, fold, fold_seed, train_indices, val_indices, node_enu, stats,
                calibration_init, "trained_best",
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
        raise RuntimeError(f"Fold {fold} did not produce a best checkpoint/OOF")
    # This is intentionally immediate: no subsequent fold is trained before the
    # complete saved-checkpoint validation-as-test pathway has passed.
    parity = run_parity_check(root, checkpoint_path, oof_path, device)
    atomic_json_dump(parity, fold_dir / "parity.json")
    del model, criterion, optimizer, train_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_path, oof_path


def _assemble_oof(
    oof_paths: Sequence[Path],
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    n = len(rows)
    result = {
        "presence_logits": np.zeros((n, 8), dtype=np.float32), "count_prob": np.zeros((n, 3), dtype=np.float32),
        "position_pred": np.zeros((n, 8, 3), dtype=np.float32), "allowlist_mask": np.zeros((n, 8), dtype=np.float32),
    }
    filled = np.zeros(n, dtype=bool)
    for path in oof_paths:
        with np.load(path, allow_pickle=False) as chunk:
            indices = chunk["source_index"].astype(np.int64)
            if filled[indices].any():
                raise RuntimeError(f"Overlapping OOF validation indices in {path}")
            for key in result:
                result[key][indices] = chunk[key]
            filled[indices] = True
    if not filled.all():
        raise RuntimeError(f"OOF missing {int((~filled).sum())} indexed training samples")
    result["sample_id"] = np.asarray([int(row["sample_id"]) for row in rows], dtype=np.int64)
    result["source_index"] = np.arange(n, dtype=np.int64)
    result["presence_target"] = np.stack([label["presence"] for label in labels]).astype(np.float32)
    result["position_target"] = np.stack([np.nan_to_num(label["positions"], nan=0.0) for label in labels]).astype(np.float32)
    result["count_target"] = result["presence_target"].sum(axis=1).astype(np.int64) - 1
    result["presence_prob"] = 1.0 / (1.0 + np.exp(-result["presence_logits"]))
    return result


def _run_test_ensemble(
    root: Path,
    output_dir: Path,
    checkpoint_paths: Sequence[Path],
    test_rows: Sequence[Mapping[str, Any]],
    node_enu: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    indices = np.arange(len(test_rows), dtype=np.int64)
    predictions = []
    for checkpoint in checkpoint_paths:
        prediction, _ = formal_inference_from_checkpoint(checkpoint, root, "test_public", test_rows, indices, node_enu, device, labels=None)
        predictions.append(prediction)
    reference = predictions[0]
    for current in predictions[1:]:
        for key in ("sample_id", "source_index", "allowlist_mask"):
            if not np.array_equal(reference[key], current[key]):
                raise RuntimeError(f"Fold ensemble inference mismatch in {key}")
    ensemble = {
        "sample_id": reference["sample_id"], "source_index": reference["source_index"], "allowlist_mask": reference["allowlist_mask"],
        "presence_logits": np.mean(np.stack([item["presence_logits"] for item in predictions]), axis=0).astype(np.float32),
        "count_prob": np.mean(np.stack([item["count_prob"] for item in predictions]), axis=0).astype(np.float32),
        "position_pred": np.mean(np.stack([item["position_pred"] for item in predictions]), axis=0).astype(np.float32),
    }
    ensemble["presence_prob"] = 1.0 / (1.0 + np.exp(-ensemble["presence_logits"]))
    drones, decoded_mask, masked_prob = decode_fixed_slots(
        ensemble["presence_logits"], ensemble["count_prob"], ensemble["position_pred"], ensemble["allowlist_mask"],
    )
    ensemble["masked_presence_prob"] = masked_prob
    ensemble["decoded_mask"] = decoded_mask
    submission_path = output_dir / "submission.jsonl"
    write_submission(ensemble["sample_id"], drones, submission_path)
    audit = audit_submission(submission_path, test_rows)
    save_prediction_archive(ensemble, output_dir / "test_ensemble_predictions.npz")
    atomic_json_dump(audit, output_dir / "submission_audit.json")
    print(f"[submission] {audit}")
    return {"submission": str(submission_path), "audit": audit}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOD-oriented RF/Radar/EO UAV fixed-slot training")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5, help="Number of trajectory/session CV folds (>=2)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--only-fold", type=int, default=None, help="Train and parity-check one session-CV fold, then stop before test inference")
    parser.add_argument(
        "--single-fold-submission",
        action="store_true",
        help="With --only-fold, run mandatory parity then create a one-checkpoint public-test submission",
    )
    parser.add_argument(
        "--eo-pretrained-path",
        default=DEFAULT_DINOV3_WEIGHT_PATH,
        help="Local DINOv3 ViT-S+/16 distilled .pth (server default is under dinov3-main/weights)",
    )
    parser.add_argument(
        "--dinov3-repo-dir",
        default=DEFAULT_DINOV3_REPO_DIR,
        help="Local clone of facebookresearch/dinov3; required for the DINOv3 ViT-S+/16 EO encoder",
    )
    parser.add_argument("--parity-only", action="store_true", help="Run an existing formal parity gate and exit")
    parser.add_argument("--reuse-folds", action="store_true", help="Reuse only checkpoints that pass formal parity")
    parser.add_argument("--no-strict-counts", action="store_true", help="Diagnostic-only override for non-official copies")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be >=2 so the selected fold has a held-out validation set for parity checking")
    if args.single_fold_submission and args.only_fold is None:
        raise ValueError("--single-fold-submission requires --only-fold K")
    root, output_dir = ensure_runtime_paths(args.data_root, args.output_dir)
    config = ProjectConfig()
    config.session.folds = args.folds
    config.session.seed = args.seed
    config.optim.epochs = args.epochs
    config.optim.batch_size = args.batch_size
    config.optim.eval_batch_size = args.eval_batch_size
    config.optim.workers = args.workers
    config.model.eo_pretrained_path = str(Path(args.eo_pretrained_path).expanduser().resolve()) if args.eo_pretrained_path else ""
    config.model.dinov3_repo_dir = str(Path(args.dinov3_repo_dir).expanduser().resolve()) if args.dinov3_repo_dir else ""
    device = _device_from_arg(args.device)
    print(f"[runtime] device={device}, output={output_dir}")

    audit = audit_dataset(root, strict_counts=not args.no_strict_counts)
    atomic_json_dump(audit, output_dir / "data_audit.json")
    print("[audit]", json.dumps({"train": audit["splits"]["train"]["indexed_samples"], "test_public": audit["splits"]["test_public"]["indexed_samples"], "description": audit["description"]}, ensure_ascii=False))
    # Runtime compatibility is checked only after the full indexed data contract
    # has passed, so malformed data never remains hidden behind an EO import.
    _validate_dinov3_runtime()
    rows, test_rows = read_index(root, "train"), read_index(root, "test_public")
    labels = [parse_label(root, "train", row) for row in rows]
    node_enu = load_node_enu(root)

    cache_dir = output_dir / "cache"
    train_fp = precompute_radar_fingerprints(root, "train", rows, cache_dir / "train_radar_fingerprint.npz")
    test_fp = precompute_radar_fingerprints(root, "test_public", test_rows, cache_dir / "test_radar_fingerprint.npz")
    session_result = build_trajectory_sessions(rows, labels, train_fp, config.session)
    folds = make_stratified_group_folds(rows, labels, session_result.session_ids, config.session.folds, config.session.seed)
    if args.only_fold is not None and not 0 <= args.only_fold < len(folds):
        raise ValueError(f"--only-fold must be in [0, {len(folds) - 1}]")
    save_session_artifacts(session_result, folds, output_dir)
    diagnostic, oof_radar_nn = cv_credibility_diagnostics(rows, labels, folds, train_fp, test_fp)
    atomic_json_dump(diagnostic, output_dir / "cv_credibility_diagnostics.json")
    np.save(output_dir / "oof_radar_nn_distance.npy", oof_radar_nn)
    print("[session-cv]", json.dumps({"chosen": session_result.chosen_config, "all_fold_gt_nn": diagnostic["all_folds_gt_same_model_nearest_3d"]}, ensure_ascii=False))
    for fold_name, fold_report in diagnostic["folds"].items():
        print(
            f"[credibility fold={fold_name}] GT→train={fold_report['gt_same_model_nearest_3d']} "
            f"radar val→train={fold_report['radar_nn_validation_to_train']} "
            f"radar test→train={fold_report['radar_nn_test_to_train']}"
        )

    if args.parity_only:
        parity_fold = 0 if args.only_fold is None else args.only_fold
        checkpoint = output_dir / f"fold_{parity_fold}" / "best.pt"
        oof = output_dir / f"fold_{parity_fold}" / "oof_best.npz"
        if not checkpoint.is_file() or not oof.is_file():
            raise FileNotFoundError(f"--parity-only requires outputs/fold_{parity_fold}/best.pt and oof_best.npz")
        result = run_parity_check(root, checkpoint, oof, device)
        atomic_json_dump(result, output_dir / "parity_only_report.json")
        return

    # Must happen before the first optimizer update.
    probe_fold = 0 if args.only_fold is None else args.only_fold
    _run_pretraining_parity_probe(root, output_dir, rows, labels, node_enu, folds, config, device, probe_fold)
    checkpoint_paths: list[Path] = []
    oof_paths: list[Path] = []
    fold_iterator = [(args.only_fold, folds[args.only_fold])] if args.only_fold is not None else list(enumerate(folds))
    for fold, (train_indices, val_indices) in fold_iterator:
        checkpoint = output_dir / f"fold_{fold}" / "best.pt"
        oof = output_dir / f"fold_{fold}" / "oof_best.npz"
        if args.reuse_folds and checkpoint.is_file() and oof.is_file():
            print(f"[fold {fold}] reusing only after formal parity passes")
            parity = run_parity_check(root, checkpoint, oof, device)
            atomic_json_dump(parity, checkpoint.parent / "parity_reuse.json")
        else:
            checkpoint, oof = _train_one_fold(
                root, output_dir, rows, labels, node_enu, train_indices, val_indices,
                oof_radar_nn, config, fold, device,
            )
        checkpoint_paths.append(checkpoint)
        oof_paths.append(oof)

    if args.only_fold is not None:
        single_fold_report = {
            "fold": args.only_fold, "checkpoint": str(checkpoint_paths[0]), "oof": str(oof_paths[0]),
            "status": "single_fold_complete_no_submission",
        }
        if not args.single_fold_submission:
            atomic_json_dump(single_fold_report, output_dir / f"fold_{args.only_fold}" / "single_fold_run.json")
            print(f"[single-fold] fold {args.only_fold} complete; parity passed; intentionally skipping incomplete-OOF aggregation and test submission")
            return
        # The single checkpoint still must pass the exact saved-OOF versus
        # formal-inference gate immediately before public-test inference.
        final_parity = run_parity_check(root, checkpoint_paths[0], oof_paths[0], device)
        atomic_json_dump(final_parity, output_dir / "single_fold_pretest_parity.json")
        submission = _run_test_ensemble(root, output_dir, checkpoint_paths, test_rows, node_enu, device)
        single_fold_report.update({"status": "single_fold_submission_complete", "submission": submission})
        atomic_json_dump(single_fold_report, output_dir / f"fold_{args.only_fold}" / "single_fold_run.json")
        print(f"[single-fold] fold {args.only_fold} submission complete after formal parity")
        return

    oof = _assemble_oof(oof_paths, rows, labels)
    oof_metrics, _ = evaluate_predictions(
        oof["presence_logits"], oof["count_prob"], oof["position_pred"], oof["allowlist_mask"],
        oof["presence_target"], oof["position_target"], oof_radar_nn,
    )
    save_prediction_archive(oof, output_dir / "oof_complete.npz")
    atomic_json_dump(oof_metrics, output_dir / "oof_metrics.json")
    print("[OOF]", json.dumps({"classification": oof_metrics["classification"], "position_all_gt_slots": oof_metrics["position_all_gt_slots"]}, ensure_ascii=False))

    # Final mandatory gate immediately before public test formal inference.
    final_parity = run_parity_check(root, checkpoint_paths[0], oof_paths[0], device)
    atomic_json_dump(final_parity, output_dir / "final_pretest_parity.json")
    _run_test_ensemble(root, output_dir, checkpoint_paths, test_rows, node_enu, device)


if __name__ == "__main__":
    main()
