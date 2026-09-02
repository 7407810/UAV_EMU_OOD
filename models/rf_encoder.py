"""RF-first shared-node raw IQ and STFT encoders."""
from __future__ import annotations

import torch
from torch import nn


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, channels: int, drop: float = 0.0) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv1d(channels, channels * 3, 1)
        self.project = nn.Conv1d(channels * 3, channels, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(value)
        value = self.norm(value)
        value = self.expand(value)
        value = self.act(value)
        value = self.drop(value)
        return residual + self.project(value)


class ConvNeXt2DBlock(nn.Module):
    def __init__(self, channels: int, drop: float = 0.0) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv2d(channels, channels * 3, 1)
        self.project = nn.Conv2d(channels * 3, channels, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(drop)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(value)
        value = self.norm(value)
        value = self.act(self.expand(value))
        value = self.drop(value)
        return residual + self.project(value)


class RawIQEncoder(nn.Module):
    def __init__(self, width: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 15, stride=4, padding=7), nn.GroupNorm(1, width), nn.GELU(),
            nn.Conv1d(width, width, 7, stride=2, padding=3), nn.GELU(),
        )
        self.blocks = nn.Sequential(ConvNeXt1DBlock(width, dropout), ConvNeXt1DBlock(width, dropout))
        self.down = nn.Conv1d(width, width * 2, 5, stride=2, padding=2)
        self.blocks2 = nn.Sequential(ConvNeXt1DBlock(width * 2, dropout), ConvNeXt1DBlock(width * 2, dropout))
        self.output = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width * 2, output_dim), nn.LayerNorm(output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks2(self.down(self.blocks(self.stem(value)))))


class SpectrogramEncoder(nn.Module):
    def __init__(self, width: int, output_dim: int, dropout: float, input_channels: int = 9) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, 5, stride=2, padding=2), nn.GroupNorm(1, width), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
        )
        self.blocks = nn.Sequential(ConvNeXt2DBlock(width, dropout), ConvNeXt2DBlock(width, dropout))
        self.down = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.blocks2 = nn.Sequential(ConvNeXt2DBlock(width * 2, dropout), ConvNeXt2DBlock(width * 2, dropout))
        self.output = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 2, output_dim), nn.LayerNorm(output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks2(self.down(self.blocks(self.stem(value)))))


class RFEncoder(nn.Module):
    """Shared per-node RF encoder followed by geometry-aware node fusion."""

    def __init__(self, dim: int, raw_width: int, spec_width: int, heads: int, layers: int, dropout: float, slots: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.raw = RawIQEncoder(raw_width, dim, dropout)
        # Per scale: log power + physical-frequency coordinate + physical-time
        # coordinate.  The dataset creates three native-rate STFT scales.
        self.spectral = SpectrogramEncoder(spec_width, dim, dropout, input_channels=9)
        self.missing_raw = nn.Parameter(torch.zeros(1, 1, dim))
        self.missing_spec = nn.Parameter(torch.zeros(1, 1, dim))
        self.scalar_geometry = nn.Sequential(
            nn.Linear(8, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, dim),
        )
        self.node_position = nn.Parameter(torch.randn(1, 4, dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        encoder = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 4, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.node_transformer = nn.TransformerEncoder(encoder, num_layers=layers)
        self.model_queries = nn.Parameter(torch.randn(1, slots, dim) * 0.02)
        self.model_cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.model_norm = nn.LayerNorm(dim)
        self.presence_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(
        self,
        raw_iq: torch.Tensor,
        stft: torch.Tensor,
        rf_scalars: torch.Tensor,
        node_present: torch.Tensor,
        node_enu: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, nodes = raw_iq.shape[:2]
        if nodes != 4:
            raise ValueError(f"RFEncoder expects 4 nodes, received {nodes}")
        raw_token = self.raw(raw_iq.flatten(0, 1)).view(batch, nodes, self.dim)
        spec_token = self.spectral(stft.flatten(0, 1)).view(batch, nodes, self.dim)
        present = node_present.unsqueeze(-1)
        raw_token = raw_token * present + self.missing_raw * (1.0 - present)
        spec_token = spec_token * present + self.missing_spec * (1.0 - present)
        # ENU is scaled only for numerical conditioning; absolute RF power and
        # sampling-rate scalars remain intact as explicit token inputs.
        scalar_input = torch.cat([rf_scalars, node_enu / 300.0, node_present.unsqueeze(-1)], dim=-1)
        tokens = raw_token + spec_token + self.scalar_geometry(scalar_input) + self.node_position
        cls = self.cls.expand(batch, -1, -1)
        # Missing nodes get a learned token instead of being silently normalized
        # into fake zero-power measurements. CLS is always valid.
        padding = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device), node_present <= 0], dim=1)
        encoded = self.node_transformer(torch.cat([cls, tokens], dim=1), src_key_padding_mask=padding)
        global_token, node_tokens = encoded[:, 0], encoded[:, 1:]
        queries = self.model_queries.expand(batch, -1, -1)
        model_tokens, _ = self.model_cross_attention(queries, node_tokens, node_tokens, need_weights=False)
        model_tokens = self.model_norm(model_tokens + queries)
        return {
            "rf_global": global_token,
            "rf_model_tokens": model_tokens,
            "rf_presence_logits": self.presence_head(model_tokens).squeeze(-1),
            "rf_node_tokens": node_tokens,
        }
