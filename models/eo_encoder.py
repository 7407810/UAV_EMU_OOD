"""DINOv3 ViT-S+/16 distilled EO encoder used only as an auxiliary modality.

The DINOv3 repository and checkpoint are deliberately required as local paths.
That prevents an implicit network download or a backbone substitution between
training, parity checking, and the formal test path.
"""
from __future__ import annotations

import hashlib
import importlib
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


_DINOV3_MODEL_NAME = "dinov3_vits16plus"
_DINOV3_EMBED_DIM = 384
_DINOV3_VITSPLUS_SHA256_PREFIX = "4057cbaa"


def _require_local_directory(value: str, option: str, required_file: str) -> Path:
    if not value:
        raise RuntimeError(
            f"{option} is required: use a local clone of facebookresearch/dinov3 "
            "and pass its directory explicitly."
        )
    path = Path(value).expanduser().resolve()
    if not path.is_dir() or not (path / required_file).is_file():
        raise RuntimeError(f"{option} must contain {required_file}: {path}")
    return path


def _require_local_file(value: str, option: str) -> Path:
    if not value:
        raise RuntimeError(
            f"{option} is required for DINOv3 ViT-S+/16 distilled. Download the approved "
            "official checkpoint once, then pass its local .pth path."
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{option} must point to an existing local checkpoint: {path}")
    return path


def _load_backbone_from_local_repo(repo_dir: Path) -> nn.Module:
    """Import only DINOv3's backbone module, never the heavy hubconf surface.

    Official ``hubconf.py`` imports segmentation and detector factories too.
    Those optional paths can require a newer AMP API than the ViT backbone
    itself, so loading through ``torch.hub.load`` needlessly breaks a pure
    feature-extraction deployment.
    """
    expected_module = (repo_dir / "dinov3" / "hub" / "backbones.py").resolve()
    if not expected_module.is_file():
        raise RuntimeError(f"DINOv3 repository is missing its backbone module: {expected_module}")
    existing = sys.modules.get("dinov3")
    if existing is not None:
        source = getattr(existing, "__file__", None)
        if source is None or repo_dir not in Path(source).resolve().parents:
            raise RuntimeError(
                "A different 'dinov3' package is already imported. Restart the "
                "process without that package or use the requested local repository."
            )
    repo_text = str(repo_dir)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    importlib.invalidate_caches()
    module = importlib.import_module("dinov3.hub.backbones")
    module_file = Path(getattr(module, "__file__", "")).resolve()
    if module_file != expected_module:
        raise RuntimeError(
            f"DINOv3 backbone resolved to {module_file}, expected the requested local file {expected_module}"
        )
    factory = getattr(module, _DINOV3_MODEL_NAME, None)
    if not callable(factory):
        raise RuntimeError(f"Requested DINOv3 factory {_DINOV3_MODEL_NAME!r} is absent from {module_file}")
    return factory(pretrained=False)


def _verify_checkpoint(path: Path) -> None:
    """Reject a truncated/wrong checkpoint before model construction."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if not actual.startswith(_DINOV3_VITSPLUS_SHA256_PREFIX):
        raise RuntimeError(
            "Unexpected DINOv3 ViT-S+/16 checkpoint SHA256. Expected prefix "
            f"{_DINOV3_VITSPLUS_SHA256_PREFIX}, received {actual[:8]}. "
            "Re-download the official LVD-1689M ViT-S+/16 distilled backbone."
        )


def _safe_load_local_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older PyTorch on competition servers.
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise RuntimeError("DINOv3 checkpoint is not a PyTorch state_dict")
    return {key.removeprefix("module."): value for key, value in state.items()}


def _extract_feature(output: Any) -> torch.Tensor:
    """Accept the official DINOv3 CLS output and fail loudly on API drift."""
    if isinstance(output, torch.Tensor):
        token = output
    elif isinstance(output, dict):
        token = output.get("x_norm_clstoken")
        if token is None:
            tensors = [value for value in output.values() if isinstance(value, torch.Tensor) and value.ndim >= 2]
            if not tensors:
                raise RuntimeError("DINOv3 output contains no usable feature tensor")
            token = tensors[0]
    elif isinstance(output, (tuple, list)) and output:
        return _extract_feature(output[0])
    else:
        raise RuntimeError(f"Unexpected DINOv3 output type: {type(output)!r}")
    if token.ndim == 3:
        token = token[:, 0]
    if token.ndim != 2 or token.shape[-1] != _DINOV3_EMBED_DIM:
        raise RuntimeError(
            "DINOv3 ViT-S+/16 EO token must have shape [B, 384], "
            f"but received {tuple(token.shape)}. Check the local DINOv3 checkout and weight file."
        )
    return token


class EOEncoder(nn.Module):
    """Pretrained DINOv3 ViT-S+/16 CLS token plus a small trainable adapter."""

    def __init__(
        self,
        dim: int,
        dropout_probability: float,
        pretrained: bool = True,
        pretrained_path: str = "",
        dinov3_repo_dir: str = "",
        train_last_blocks: int = 0,
    ) -> None:
        super().__init__()
        if not pretrained:
            raise RuntimeError("DINOv3 ViT-S+/16 must use its pretrained official checkpoint; random EO initialization is disabled.")
        if train_last_blocks < 0:
            raise ValueError("train_last_blocks must be non-negative")
        self.dropout_probability = float(dropout_probability)
        self.missing = nn.Parameter(torch.zeros(1, dim))
        repo_dir = _require_local_directory(dinov3_repo_dir, "--dinov3-repo-dir", "dinov3/hub/backbones.py")
        weights_path = _require_local_file(pretrained_path, "--eo-pretrained-path")
        _verify_checkpoint(weights_path)
        try:
            # Deliberately bypass hubconf.py: it imports optional segmentors and
            # detectors which are irrelevant to EO token extraction.
            self.backbone = _load_backbone_from_local_repo(repo_dir)
            self.backbone.load_state_dict(_safe_load_local_state_dict(weights_path), strict=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the local DINOv3 ViT-S+/16 backbone. Verify the "
                "official repository, checkpoint hash, and PyTorch SDPA support. "
                "The loader intentionally skips hubconf optional segmentation dependencies."
            ) from exc

        backbone_dim = int(getattr(self.backbone, "embed_dim", _DINOV3_EMBED_DIM))
        if backbone_dim != _DINOV3_EMBED_DIM:
            raise RuntimeError(f"Expected DINOv3 ViT-S+/16 embed_dim=384, got {backbone_dim}")
        self._configure_trainable_blocks(train_last_blocks)
        self.backbone_trainable = any(parameter.requires_grad for parameter in self.backbone.parameters())
        self.project = nn.Sequential(
            nn.LayerNorm(_DINOV3_EMBED_DIM),
            nn.Linear(_DINOV3_EMBED_DIM, dim),
            nn.LayerNorm(dim),
        )

    def _configure_trainable_blocks(self, train_last_blocks: int) -> None:
        # EO is intentionally not the recognition main path. Freezing early
        # DINOv3 layers reduces sample-level shortcut fitting; the final blocks
        # and adapter can still adapt to this EO domain end-to-end.
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        blocks = list(getattr(self.backbone, "blocks", []))
        if not blocks and train_last_blocks:
            raise RuntimeError("Official DINOv3 backbone does not expose transformer blocks for EO fine-tuning")
        for block in blocks[-train_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        if train_last_blocks:
            for name in ("norm", "cls_norm", "head"):
                module = getattr(self.backbone, name, None)
                if isinstance(module, nn.Module):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "EOEncoder":
        super().train(mode)
        # With the default frozen backbone, preserve deterministic pretrained
        # features even while the adapter/fusion network is training.
        if not getattr(self, "backbone_trainable", False):
            self.backbone.eval()
        return self

    def _backbone_token(self, image: torch.Tensor) -> torch.Tensor:
        # DINO is a pretrained auxiliary encoder.  Evaluate it in fp32 under a
        # surrounding bf16/fp16 training context so its attention path cannot
        # introduce mixed-precision NaN into the RF/Radar main graph.
        autocast = torch.autocast(device_type="cuda", enabled=False) if image.device.type == "cuda" else nullcontext()
        if self.backbone_trainable:
            with autocast:
                return _extract_feature(self.backbone(image.float()))
        with torch.no_grad(), autocast:
            return _extract_feature(self.backbone(image.float()))

    def forward(self, image: torch.Tensor, has_eo: torch.Tensor) -> torch.Tensor:
        token = self._backbone_token(image)
        token = self.project(token)
        missing = has_eo <= 0
        if self.training and self.dropout_probability > 0:
            missing = missing | (torch.rand_like(has_eo) < self.dropout_probability)
        return torch.where(missing.unsqueeze(-1), self.missing.expand_as(token), token)
