"""Optional local DINOv3 EO patch-token encoder."""
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
_VERIFIED_CHECKPOINTS: set[str] = set()


def _require_local_directory(value: str, option: str, required_file: str) -> Path:
    path = Path(value).expanduser().resolve() if value else Path()
    if not value or not path.is_dir() or not (path / required_file).is_file():
        raise RuntimeError(f"{option} must point to a local DINOv3 checkout containing {required_file}: {path}")
    return path


def _require_local_file(value: str, option: str) -> Path:
    path = Path(value).expanduser().resolve() if value else Path()
    if not value or not path.is_file():
        raise RuntimeError(f"{option} must point to the local DINOv3 ViT-S+/16 checkpoint: {path}")
    return path


def _load_backbone_from_local_repo(repo_dir: Path) -> nn.Module:
    """Import only DINO's backbone factory, avoiding optional hub segmentors."""
    expected = (repo_dir / "dinov3" / "hub" / "backbones.py").resolve()
    if not expected.is_file():
        raise RuntimeError(f"DINOv3 backbone module is missing: {expected}")
    existing = sys.modules.get("dinov3")
    if existing is not None:
        source = getattr(existing, "__file__", None)
        if source is None or repo_dir not in Path(source).resolve().parents:
            raise RuntimeError("A different dinov3 package is already imported; restart with the requested local checkout.")
    repo_text = str(repo_dir)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    importlib.invalidate_caches()
    module = importlib.import_module("dinov3.hub.backbones")
    if Path(getattr(module, "__file__", "")).resolve() != expected:
        raise RuntimeError("DINOv3 backbone import did not resolve to the requested repository")
    factory = getattr(module, _DINOV3_MODEL_NAME, None)
    if not callable(factory):
        raise RuntimeError(f"DINOv3 factory {_DINOV3_MODEL_NAME!r} is not available")
    return factory(pretrained=False)


def _verify_checkpoint(path: Path) -> None:
    path_text = str(path)
    if path_text in _VERIFIED_CHECKPOINTS:
        return
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if not digest.hexdigest().startswith(_DINOV3_VITSPLUS_SHA256_PREFIX):
        raise RuntimeError(
            "Unexpected DINOv3 ViT-S+/16 checkpoint hash. Expected official SHA256 prefix "
            f"{_DINOV3_VITSPLUS_SHA256_PREFIX}, got {digest.hexdigest()[:8]}."
        )
    _VERIFIED_CHECKPOINTS.add(path_text)


def _safe_load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise RuntimeError("DINOv3 checkpoint is not a PyTorch state dict")
    return {key.removeprefix("module."): value for key, value in state.items()}


def _extract_patch_tokens(output: Any) -> torch.Tensor:
    """Return CLS+patch tokens from the official DINO forward-features API."""
    if isinstance(output, dict):
        patch = output.get("x_norm_patchtokens")
        cls = output.get("x_norm_clstoken")
        if isinstance(patch, torch.Tensor) and patch.ndim == 3:
            if isinstance(cls, torch.Tensor) and cls.ndim == 2:
                patch = torch.cat([cls.unsqueeze(1), patch], dim=1)
            token = patch
        else:
            candidates = [value for value in output.values() if isinstance(value, torch.Tensor) and value.ndim == 3]
            if not candidates:
                raise RuntimeError("DINOv3 forward features has no patch-token tensor")
            token = candidates[0]
    elif isinstance(output, torch.Tensor):
        token = output
    elif isinstance(output, (tuple, list)) and output:
        return _extract_patch_tokens(output[0])
    else:
        raise RuntimeError(f"Unexpected DINOv3 feature output type: {type(output)!r}")
    if token.ndim == 2:
        token = token.unsqueeze(1)
    if token.ndim != 3 or token.shape[-1] != _DINOV3_EMBED_DIM:
        raise RuntimeError(f"DINOv3 ViT-S+/16 tokens must be [B,P,384], got {tuple(token.shape)}")
    return token


class EOEncoder(nn.Module):
    """EO patch tokens with an explicit missing-modality token and mask."""

    def __init__(
        self,
        dim: int,
        dropout_probability: float,
        enabled: bool,
        pretrained: bool = True,
        pretrained_path: str = "",
        dinov3_repo_dir: str = "",
        train_last_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.dropout_probability = float(dropout_probability)
        self.missing = nn.Parameter(torch.zeros(1, 1, dim))
        self.backbone_trainable = False
        if not self.enabled:
            self.backbone = None
            self.project = nn.Identity()
            return
        if not pretrained:
            raise RuntimeError("EO is enabled but pretrained DINOv3 weights are disabled; use --disable-eo instead.")
        if train_last_blocks < 0:
            raise ValueError("eo_train_last_blocks must be non-negative")
        repo_dir = _require_local_directory(dinov3_repo_dir, "--dinov3-repo-dir", "dinov3/hub/backbones.py")
        weight_path = _require_local_file(pretrained_path, "--eo-pretrained-path")
        _verify_checkpoint(weight_path)
        try:
            self.backbone = _load_backbone_from_local_repo(repo_dir)
            self.backbone.load_state_dict(_safe_load_state_dict(weight_path), strict=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load local DINOv3 ViT-S+/16. The loader imports only backbone code; "
                "verify the repository and checkpoint paths/hash."
            ) from exc
        if int(getattr(self.backbone, "embed_dim", _DINOV3_EMBED_DIM)) != _DINOV3_EMBED_DIM:
            raise RuntimeError("Configured DINOv3 model does not have the expected 384-dimensional token")
        self._configure_trainable_blocks(train_last_blocks)
        self.project = nn.Sequential(nn.LayerNorm(_DINOV3_EMBED_DIM), nn.Linear(_DINOV3_EMBED_DIM, dim), nn.LayerNorm(dim))

    def _configure_trainable_blocks(self, train_last_blocks: int) -> None:
        assert self.backbone is not None
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        blocks = list(getattr(self.backbone, "blocks", []))
        if train_last_blocks and not blocks:
            raise RuntimeError("DINOv3 backbone does not expose transformer blocks for requested EO fine-tuning")
        for block in blocks[-train_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        if train_last_blocks:
            for name in ("norm", "cls_norm"):
                module = getattr(self.backbone, name, None)
                if isinstance(module, nn.Module):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        self.backbone_trainable = any(parameter.requires_grad for parameter in self.backbone.parameters())

    def train(self, mode: bool = True) -> "EOEncoder":
        super().train(mode)
        if self.enabled and not self.backbone_trainable and isinstance(self.backbone, nn.Module):
            self.backbone.eval()
        return self

    def _forward_features(self, image: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.backbone, nn.Module)
        autocast = torch.autocast(device_type="cuda", enabled=False) if image.device.type == "cuda" else nullcontext()
        def run() -> torch.Tensor:
            method = getattr(self.backbone, "forward_features", None)
            output = method(image.float()) if callable(method) else self.backbone(image.float())
            return _extract_patch_tokens(output)
        if self.backbone_trainable:
            with autocast:
                return run()
        with torch.no_grad(), autocast:
            return run()

    def forward(self, image: torch.Tensor, has_eo: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = image.shape[0]
        if not self.enabled:
            token = self.missing.expand(batch_size, -1, -1)
            return {"tokens": token, "key_padding_mask": torch.zeros(batch_size, 1, dtype=torch.bool, device=image.device)}

        present = has_eo > 0
        if self.training and self.dropout_probability > 0.0:
            present = present & ~(torch.rand_like(has_eo) < self.dropout_probability)
        # If an entire batch lacks EO, do not run the large vision backbone just
        # to discard its result. A valid learned missing token remains available
        # to every decoder query.
        if not bool(present.any()):
            token = self.missing.expand(batch_size, -1, -1)
            return {"tokens": token, "key_padding_mask": torch.zeros(batch_size, 1, dtype=torch.bool, device=image.device)}

        tokens = self.project(self._forward_features(image))
        mask = torch.zeros(batch_size, tokens.shape[1], dtype=torch.bool, device=image.device)
        missing_rows = ~present
        if bool(missing_rows.any()):
            tokens = tokens.clone()
            tokens[missing_rows] = 0.0
            tokens[missing_rows, 0:1] = self.missing.expand(int(missing_rows.sum()), -1, -1)
            mask[missing_rows, 1:] = True
        return {"tokens": tokens, "key_padding_mask": mask}
