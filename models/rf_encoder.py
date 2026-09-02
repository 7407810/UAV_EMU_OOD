"""Shared Raw-IQ and native-rate STFT RF token encoder."""
from __future__ import annotations

import torch
from torch import nn


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv1d(channels, channels * 3, 1)
        self.project = nn.Conv1d(channels * 3, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(value)
        value = self.norm(value)
        value = torch.nn.functional.gelu(self.expand(value))
        return residual + self.project(self.dropout(value))


class ConvNeXt2DBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv2d(channels, channels * 3, 1)
        self.project = nn.Conv2d(channels * 3, channels, 1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(value)
        value = self.norm(value)
        value = torch.nn.functional.gelu(self.expand(value))
        return residual + self.project(self.dropout(value))


class RawIQEncoder(nn.Module):
    def __init__(self, width: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 15, stride=4, padding=7), nn.GroupNorm(1, width), nn.GELU(),
            nn.Conv1d(width, width, 7, stride=2, padding=3), nn.GELU(),
        )
        self.blocks1 = nn.Sequential(ConvNeXt1DBlock(width, dropout), ConvNeXt1DBlock(width, dropout))
        self.down = nn.Conv1d(width, width * 2, 5, stride=2, padding=2)
        self.blocks2 = nn.Sequential(ConvNeXt1DBlock(width * 2, dropout), ConvNeXt1DBlock(width * 2, dropout))
        self.output = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width * 2, output_dim), nn.LayerNorm(output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks2(self.down(self.blocks1(self.stem(value)))))


class SpectrogramEncoder(nn.Module):
    def __init__(self, width: int, output_dim: int, dropout: float, input_channels: int = 9) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 5, stride=2, padding=2), nn.GroupNorm(1, width), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
        )
        self.blocks1 = nn.Sequential(ConvNeXt2DBlock(width, dropout), ConvNeXt2DBlock(width, dropout))
        self.down = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.blocks2 = nn.Sequential(ConvNeXt2DBlock(width * 2, dropout), ConvNeXt2DBlock(width * 2, dropout))
        self.output = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 2, output_dim), nn.LayerNorm(output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks2(self.down(self.blocks1(self.stem(value)))))


class RFEncoder(nn.Module):
    """Four shared node encoders produce unbound RF modality tokens.

    Node identity is represented by a learnable token index, while RF receiver
    ENU geometry is intentionally absent: no undocumented range/distance law is
    imposed on the waveform branch.
    """

    def __init__(self, dim: int, raw_width: int, spec_width: int, heads: int, layers: int, dropout: float, num_nodes: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_nodes = int(num_nodes)
        self.raw = RawIQEncoder(raw_width, dim, dropout)
        self.spectral = SpectrogramEncoder(spec_width, dim, dropout, input_channels=9)
        self.missing_raw = nn.Parameter(torch.zeros(1, 1, dim))
        self.missing_spectral = nn.Parameter(torch.zeros(1, 1, dim))
        self.scalar = nn.Sequential(nn.Linear(5, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.node_index = nn.Parameter(torch.randn(1, num_nodes, dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=dim * 4, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(
        self,
        raw_iq: torch.Tensor,
        stft: torch.Tensor,
        rf_scalars: torch.Tensor,
        node_present: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, nodes = raw_iq.shape[:2]
        if nodes != self.num_nodes:
            raise ValueError(f"RFEncoder expects {self.num_nodes} nodes, got {nodes}")
        raw = self.raw(raw_iq.flatten(0, 1)).view(batch_size, nodes, self.dim)
        spectral = self.spectral(stft.flatten(0, 1)).view(batch_size, nodes, self.dim)
        present = node_present.unsqueeze(-1)
        raw = raw * present + self.missing_raw * (1.0 - present)
        spectral = spectral * present + self.missing_spectral * (1.0 - present)
        scalar_input = torch.cat([rf_scalars, present], dim=-1)
        node_tokens = raw + spectral + self.scalar(scalar_input) + self.node_index
        tokens = torch.cat([self.cls.expand(batch_size, -1, -1), node_tokens], dim=1)
        # A missing RF node is represented by learned missing content rather
        # than masked away, so a completely missing RF sample remains finite.
        encoded = self.fusion(tokens)
        return {
            "tokens": encoded,
            "key_padding_mask": torch.zeros(batch_size, encoded.shape[1], dtype=torch.bool, device=encoded.device),
        }
