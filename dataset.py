"""One indexed Dataset/preprocess path for train, validation, parity and test."""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

from config import DataConfig, OptimConfig
from utils import MAX_TARGETS, MODEL_COUNT, causal_radar_points, indexed_path, parse_int_list


class IndexedUAVDataset(Dataset[dict[str, torch.Tensor]]):
    """Loads only rows explicitly referenced by an official ``index.csv``.

    ``label_signature`` is neither read as an input nor emitted as a target.
    Padded labels have deterministic storage order but queries are deliberately
    unordered and are matched with Hungarian assignment in the loss.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        rows: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        data_cfg: DataConfig,
        labels: Sequence[Mapping[str, Any]] | None = None,
        training: bool = False,
        seed: int = 3407,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.split = str(split)
        self.rows = rows
        self.indices = np.asarray(indices, dtype=np.int64)
        self.cfg = data_cfg
        self.labels = labels
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        if self.training and self.labels is None:
            raise ValueError("A training Dataset requires labels")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def _rng(self, sample_id: int) -> np.random.Generator:
        # Eval has epoch=0 forever. The same sample/fold seed therefore selects
        # exactly the same Radar subset in validation, formal inference and test.
        state = (self.seed * 1_000_003 + self.epoch * 9_176 + int(sample_id) * 37) & 0xFFFFFFFF
        return np.random.default_rng(state)

    def _native_iq_window(self, waveform: np.ndarray, sampling_rate: float) -> np.ndarray:
        if len(waveform) == 0 or sampling_rate <= 0.0:
            return np.zeros(0, dtype=np.complex64)
        maximum = max(1, int(round(self.cfg.iq_window_seconds * sampling_rate)))
        return waveform[: min(len(waveform), maximum)].astype(np.complex64, copy=False)

    def _resample_raw_iq(self, waveform: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, int]:
        """Resample IQ onto a fixed *time* window, not a fixed source count."""
        target_length = self.cfg.raw_iq_length
        source = self._native_iq_window(waveform, sampling_rate)
        if len(source) == 0:
            return np.zeros(target_length, dtype=np.complex64), 0
        target_rate = target_length / self.cfg.iq_window_seconds
        ratio = Fraction(float(target_rate / sampling_rate)).limit_denominator(8192)
        if ratio.numerator == ratio.denominator:
            resampled = source
        else:
            real = resample_poly(source.real.astype(np.float32, copy=False), ratio.numerator, ratio.denominator, padtype="constant")
            imag = resample_poly(source.imag.astype(np.float32, copy=False), ratio.numerator, ratio.denominator, padtype="constant")
            resampled = (real + 1j * imag).astype(np.complex64, copy=False)
        expected = min(target_length, max(1, int(round(len(source) / sampling_rate * target_rate))))
        output = np.zeros(target_length, dtype=np.complex64)
        count = min(expected, len(resampled), target_length)
        output[:count] = resampled[:count]
        return output, count

    def _make_stft(self, waveform: np.ndarray, sampling_rate: float) -> torch.Tensor:
        """Native-rate multi-scale STFT with rate-derived time/frequency axes."""
        source_wave = self._native_iq_window(waveform, sampling_rate)
        channels = 3 * len(self.cfg.stft_window_seconds)
        if len(source_wave) == 0:
            return torch.zeros(channels, self.cfg.stft_freq_bins, self.cfg.stft_time_bins)
        source = torch.from_numpy(source_wave)
        result: list[torch.Tensor] = []
        for window_seconds in self.cfg.stft_window_seconds:
            requested_fft = max(16, int(round(float(window_seconds) * sampling_rate)))
            n_fft = max(16, min(requested_fft, max(len(source_wave), 16)))
            signal = F.pad(source, (0, n_fft - len(source_wave))) if len(source_wave) < n_fft else source
            hop = max(1, int(round(n_fft * self.cfg.stft_hop_ratio)))
            window = torch.hann_window(n_fft, dtype=torch.float32)
            spectrogram = torch.stft(signal, n_fft=n_fft, hop_length=hop, window=window, return_complex=True, center=False)
            spectrogram = torch.fft.fftshift(spectrogram, dim=0)
            log_power = torch.log1p(spectrogram.abs().square())
            frequencies = torch.fft.fftshift(torch.fft.fftfreq(n_fft, d=1.0 / sampling_rate))
            frequency_axis = torch.sign(frequencies) * torch.log1p(frequencies.abs())
            frequency_axis = frequency_axis / math.log1p(max(0.5 * sampling_rate, 1.0))
            frame_time = (torch.arange(spectrogram.shape[1], dtype=torch.float32) * hop + 0.5 * n_fft) / sampling_rate
            time_axis = frame_time / self.cfg.iq_window_seconds

            def resize(value: torch.Tensor) -> torch.Tensor:
                return F.interpolate(
                    value.unsqueeze(0).unsqueeze(0),
                    size=(self.cfg.stft_freq_bins, self.cfg.stft_time_bins),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)

            result.extend([
                resize(log_power),
                resize(frequency_axis[:, None].expand(-1, spectrogram.shape[1])),
                resize(time_axis[None, :].expand(spectrogram.shape[0], -1)),
            ])
        return torch.cat(result, dim=0)

    def _load_iq(self, row: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path = indexed_path(self.data_root, self.split, str(row["iq_npz_relpath"]))
        raw_features: list[torch.Tensor] = []
        stft_features: list[torch.Tensor] = []
        scalar_features: list[list[float]] = []
        node_present: list[float] = []
        with np.load(path, allow_pickle=False) as npz:
            for node in range(self.cfg.num_nodes):
                interleaved = np.asarray(npz[f"iq_node{node}"], dtype=np.int16)
                sampling_rate = float(np.asarray(npz[f"sr_node{node}"]).reshape(-1)[0])
                indexed_present = int(row.get(f"iq_node{node}", 0)) > 0
                usable = len(interleaved) // 2
                if not indexed_present or usable == 0 or sampling_rate <= 0.0:
                    waveform = np.zeros(0, dtype=np.complex64)
                    present = 0.0
                else:
                    iq = interleaved[: 2 * usable].astype(np.float32) / 32768.0
                    waveform = (iq[0::2] + 1j * iq[1::2]).astype(np.complex64, copy=False)
                    present = 1.0
                resampled, valid_length = self._resample_raw_iq(waveform, sampling_rate)
                magnitude = np.abs(resampled)
                phase_delta = np.zeros_like(magnitude, dtype=np.float32)
                if valid_length > 1:
                    phase_delta[1:valid_length] = np.angle(
                        resampled[1:valid_length] * np.conj(resampled[: valid_length - 1])
                    ).astype(np.float32)
                raw_features.append(torch.from_numpy(np.stack([
                    resampled.real, resampled.imag, magnitude, phase_delta,
                ], axis=0).astype(np.float32)))
                native = self._native_iq_window(waveform, sampling_rate)
                duration_fraction = min(len(native) / max(sampling_rate * self.cfg.iq_window_seconds, 1e-12), 1.0) if len(native) else 0.0
                rms = float(np.sqrt(np.mean(np.abs(native) ** 2) + 1e-12)) if len(native) else 0.0
                # Absolute power and SR remain explicit. Per-node normalization
                # would erase meaningful RF amplitude/rate information.
                scalar_features.append([
                    math.log1p(rms), math.log1p(rms * rms), math.log1p(max(sampling_rate, 0.0)), duration_fraction,
                ])
                stft_features.append(
                    self._make_stft(waveform, sampling_rate)
                    if present else torch.zeros(3 * len(self.cfg.stft_window_seconds), self.cfg.stft_freq_bins, self.cfg.stft_time_bins)
                )
                node_present.append(present)
        return (
            torch.stack(raw_features),
            torch.stack(stft_features),
            torch.tensor(scalar_features, dtype=torch.float32),
            torch.tensor(node_present, dtype=torch.float32),
        )

    def _load_radar(self, row: Mapping[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        path = indexed_path(self.data_root, self.split, str(row["radar_npy_relpath"]))
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(f"Radar must have shape (N,4): {path} -> {values.shape}")
        values = causal_radar_points(values)
        # File row ordering is not part of the data contract. Uniform random
        # subsampling is therefore index-order invariant and does not encode a
        # hand-crafted temporal track rule.
        if len(values) > self.cfg.max_radar_points:
            choice = rng.choice(len(values), size=self.cfg.max_radar_points, replace=False)
            values = values[choice]
        if self.training and len(values) > 1 and self.cfg.radar_point_dropout > 0.0:
            keep = rng.random(len(values)) >= self.cfg.radar_point_dropout
            if not np.any(keep):
                keep[int(rng.integers(len(values)))] = True
            values = values[keep]
        points = np.zeros((self.cfg.max_radar_points, 4), dtype=np.float32)
        mask = np.zeros(self.cfg.max_radar_points, dtype=np.bool_)
        if len(values):
            points[: len(values)] = values
            mask[: len(values)] = True
        else:
            # Keep attention numerically defined for a pathological empty file;
            # the all-zero dummy is distinguishable through this valid mask.
            mask[0] = True
        return points, mask

    def _load_eo(self, row: Mapping[str, Any]) -> tuple[torch.Tensor, float]:
        if not int(row.get("has_eo", 0)):
            return torch.zeros(3, self.cfg.eo_size, self.cfg.eo_size), 0.0
        path = indexed_path(self.data_root, self.split, str(row["eo_jpg_relpath"]))
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            scale = min(self.cfg.eo_size / max(width, 1), self.cfg.eo_size / max(height, 1))
            resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (self.cfg.eo_size, self.cfg.eo_size), color=(0, 0, 0))
            offset = ((self.cfg.eo_size - resized.width) // 2, (self.cfg.eo_size - resized.height) // 2)
            canvas.paste(resized, offset)
            array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
        array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )[:, None, None]
        return torch.from_numpy(array.copy()), 1.0

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        source_index = int(self.indices[item])
        row = self.rows[source_index]
        sample_id = int(row["sample_id"])
        rng = self._rng(sample_id)
        raw_iq, stft, rf_scalars, node_present = self._load_iq(row)
        radar_points, radar_mask = self._load_radar(row, rng)
        eo_image, has_eo = self._load_eo(row)
        allowlist = parse_int_list(row.get("allowlist", ""))
        allowlist_mask = np.zeros(MODEL_COUNT, dtype=np.bool_)
        allowlist_mask[allowlist] = True

        if self.labels is None:
            target_model_ids = np.full(MAX_TARGETS, -100, dtype=np.int64)
            target_positions = np.zeros((MAX_TARGETS, 3), dtype=np.float32)
            target_mask = np.zeros(MAX_TARGETS, dtype=np.bool_)
        else:
            label = self.labels[source_index]
            target_model_ids = np.asarray(label["target_model_ids"], dtype=np.int64).copy()
            target_positions = np.asarray(label["target_positions"], dtype=np.float32).copy()
            target_mask = np.asarray(label["target_mask"], dtype=np.bool_).copy()

        return {
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
            "source_index": torch.tensor(source_index, dtype=torch.long),
            "raw_iq": raw_iq,
            "stft": stft,
            "rf_scalars": rf_scalars,
            "node_present": node_present,
            "radar_points": torch.from_numpy(radar_points),
            "radar_mask": torch.from_numpy(radar_mask),
            "eo_image": eo_image,
            "has_eo": torch.tensor(has_eo, dtype=torch.float32),
            "allowlist_mask": torch.from_numpy(allowlist_mask),
            "target_model_ids": torch.from_numpy(target_model_ids),
            "target_positions": torch.from_numpy(target_positions),
            "target_mask": torch.from_numpy(target_mask),
        }


def make_dataloader(dataset: Dataset[Any], optim_cfg: OptimConfig, training: bool) -> DataLoader[Any]:
    workers = int(optim_cfg.workers)
    kwargs: dict[str, Any] = {
        "batch_size": int(optim_cfg.batch_size if training else optim_cfg.eval_batch_size),
        "shuffle": bool(training),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        # Fresh eval workers prevent stale Dataset state from invalidating parity.
        "persistent_workers": False,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)
