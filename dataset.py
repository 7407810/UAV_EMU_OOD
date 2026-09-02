"""Single indexed dataset/preprocess implementation for train, CV and test."""
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
from utils import SLOT_COUNT, causal_radar_points, indexed_path, parse_int_list


class IndexedUAVDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset restricted to an explicit list of rows from index.csv.

    ``label_signature`` is audit/CV metadata only.  It is never emitted as an
    model input or an auxiliary training target.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        rows: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        node_enu: np.ndarray,
        data_cfg: DataConfig,
        labels: Sequence[Mapping[str, Any]] | None = None,
        training: bool = False,
        seed: int = 3407,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.split = split
        self.rows = rows
        self.indices = np.asarray(indices, dtype=np.int64)
        self.node_enu = np.asarray(node_enu, dtype=np.float32)
        self.cfg = data_cfg
        self.labels = labels
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        if self.node_enu.shape != (4, 3):
            raise ValueError(f"Expected 4x3 RF node ENU, got {self.node_enu.shape}")
        if self.training and labels is None:
            raise ValueError("Training dataset requires labels")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def _rng(self, sample_id: int) -> np.random.Generator:
        value = (self.seed * 1_000_003 + self.epoch * 9_176 + int(sample_id) * 37) & 0xFFFFFFFF
        return np.random.default_rng(value)

    def _native_iq_window(self, waveform: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Keep the documented physical capture window rather than a sample crop."""
        if len(waveform) == 0 or sampling_rate <= 0.0:
            return np.zeros(0, dtype=np.complex64)
        maximum_samples = max(1, int(round(self.cfg.iq_window_seconds * sampling_rate)))
        return waveform[: min(len(waveform), maximum_samples)].astype(np.complex64, copy=False)

    def _resample_raw_iq(self, waveform: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, int]:
        """Band-limited resampling onto a common 10 ms physical-time grid.

        A fixed 8192-*sample* crop changes temporal coverage when ``sr`` varies.
        This operation uses ``sr`` to preserve capture duration, while the native
        waveform is retained separately for the STFT branch.
        """
        target = self.cfg.raw_iq_length
        source = self._native_iq_window(waveform, sampling_rate)
        if len(source) == 0:
            return np.zeros(target, dtype=np.complex64), 0
        target_rate = target / self.cfg.iq_window_seconds
        ratio = Fraction(float(target_rate / sampling_rate)).limit_denominator(8192)
        if ratio.numerator == ratio.denominator:
            resampled = source
        else:
            real = resample_poly(source.real.astype(np.float32, copy=False), ratio.numerator, ratio.denominator, padtype="constant")
            imag = resample_poly(source.imag.astype(np.float32, copy=False), ratio.numerator, ratio.denominator, padtype="constant")
            resampled = (real + 1j * imag).astype(np.complex64, copy=False)
        expected = min(target, max(1, int(round(len(source) / sampling_rate * target_rate))))
        output = np.zeros(target, dtype=np.complex64)
        count = min(expected, len(resampled), target)
        output[:count] = resampled[:count]
        return output, count

    def _make_stft(self, waveform: np.ndarray, sampling_rate: float) -> torch.Tensor:
        """Native-rate multi-scale STFT with explicit physical time/frequency axes."""
        source_wave = self._native_iq_window(waveform, sampling_rate)
        if len(source_wave) == 0:
            return torch.zeros(3 * len(self.cfg.stft_window_seconds), self.cfg.stft_freq_bins, self.cfg.stft_time_bins)
        source = torch.from_numpy(source_wave)
        outputs: list[torch.Tensor] = []
        for duration_s in self.cfg.stft_window_seconds:
            requested_fft = max(16, int(round(float(duration_s) * sampling_rate)))
            n_fft = max(16, min(requested_fft, max(len(source_wave), 16)))
            signal = F.pad(source, (0, n_fft - len(source_wave))) if len(source_wave) < n_fft else source
            hop = max(1, int(round(n_fft * self.cfg.stft_hop_ratio)))
            window = torch.hann_window(n_fft, dtype=torch.float32)
            spec = torch.stft(signal, n_fft=n_fft, hop_length=hop, window=window, return_complex=True, center=False)
            spec = torch.fft.fftshift(spec, dim=0)
            log_power = torch.log1p(spec.abs().square())
            frequencies = torch.fft.fftshift(torch.fft.fftfreq(n_fft, d=1.0 / sampling_rate))
            frequency_axis = torch.sign(frequencies) * torch.log1p(frequencies.abs())
            frequency_axis = frequency_axis / math.log1p(max(0.5 * sampling_rate, 1.0))
            frame_time_s = (torch.arange(spec.shape[1], dtype=torch.float32) * hop + 0.5 * n_fft) / sampling_rate
            time_axis = frame_time_s / self.cfg.iq_window_seconds

            def resize(value: torch.Tensor) -> torch.Tensor:
                return F.interpolate(
                    value.unsqueeze(0).unsqueeze(0),
                    size=(self.cfg.stft_freq_bins, self.cfg.stft_time_bins),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)

            outputs.extend([
                resize(log_power),
                resize(frequency_axis[:, None].expand(-1, spec.shape[1])),
                resize(time_axis[None, :].expand(spec.shape[0], -1)),
            ])
        return torch.cat(outputs, dim=0)

    def _load_iq(self, row: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path = indexed_path(self.data_root, self.split, str(row["iq_npz_relpath"]))
        raw_features, stfts, scalars, present = [], [], [], []
        with np.load(path, allow_pickle=False) as npz:
            for node in range(4):
                interleaved = np.asarray(npz[f"iq_node{node}"], dtype=np.int16)
                sampling_rate = float(np.asarray(npz[f"sr_node{node}"]).reshape(-1)[0])
                expected_present = int(row.get(f"iq_node{node}", 0)) > 0
                usable = len(interleaved) // 2
                if not expected_present or usable == 0 or sampling_rate <= 0:
                    wave = np.zeros(0, dtype=np.complex64)
                    is_present = 0.0
                else:
                    iq = interleaved[: usable * 2].astype(np.float32) / 32768.0
                    wave = (iq[0::2] + 1j * iq[1::2]).astype(np.complex64, copy=False)
                    is_present = 1.0
                raw_wave, valid_length = self._resample_raw_iq(wave, sampling_rate)
                magnitude = np.abs(raw_wave)
                phase_delta = np.zeros_like(magnitude, dtype=np.float32)
                if valid_length > 1:
                    phase_delta[1:valid_length] = np.angle(raw_wave[1:valid_length] * np.conj(raw_wave[: valid_length - 1])).astype(np.float32)
                channels = np.stack([raw_wave.real, raw_wave.imag, magnitude, phase_delta], axis=0).astype(np.float32)
                native = self._native_iq_window(wave, sampling_rate)
                duration_fraction = min(len(native) / max(sampling_rate * self.cfg.iq_window_seconds, 1e-12), 1.0) if len(native) else 0.0
                rms = float(np.sqrt(np.mean(np.abs(native) ** 2) + 1e-12)) if len(native) else 0.0
                # Keep absolute RF power (rather than normalizing individual nodes away).
                scalars.append([math.log1p(rms), math.log1p(rms * rms), math.log1p(max(sampling_rate, 0.0)), duration_fraction])
                raw_features.append(torch.from_numpy(channels))
                stfts.append(self._make_stft(wave, sampling_rate) if is_present else torch.zeros(3 * len(self.cfg.stft_window_seconds), self.cfg.stft_freq_bins, self.cfg.stft_time_bins))
                present.append(is_present)
        return (
            torch.stack(raw_features), torch.stack(stfts), torch.tensor(scalars, dtype=torch.float32),
            torch.tensor(present, dtype=torch.float32),
        )

    def _load_radar(self, row: Mapping[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        path = indexed_path(self.data_root, self.split, str(row["radar_npy_relpath"]))
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(f"Radar array must have shape (N,4): {path} -> {values.shape}")
        # Train contains a small set of rel_time_s > 0 rows while public test
        # contains none.  The data contract defines Radar as historical at t=0,
        # so exclude those rows for every split before any ordering/subsampling.
        # Do not clip or negate them: either operation would invent a timestamp.
        values = causal_radar_points(values)
        # The specification gives semantic time but does not guarantee file-row
        # ordering.  Sort by rel_time_s before any subsampling.
        if len(values):
            values = values[np.argsort(values[:, 3], kind="stable")]
        max_points = self.cfg.max_radar_points
        if len(values) > max_points:
            # One point per sorted-time stratum preserves the documented 3 s
            # window without assuming anything about source-file row order.
            boundaries = np.linspace(0, len(values), max_points + 1, dtype=np.int64)
            if self.training and self.cfg.train_random_radar_sample:
                choose = np.asarray([
                    int(rng.integers(boundaries[i], max(boundaries[i] + 1, boundaries[i + 1])))
                    for i in range(max_points)
                ], dtype=np.int64)
            else:
                choose = ((boundaries[:-1] + boundaries[1:] - 1) // 2).astype(np.int64)
            values = values[choose]
        points = np.zeros((max_points, 4), dtype=np.float32)
        mask = np.zeros(max_points, dtype=np.bool_)
        if len(values):
            points[: len(values)] = values
            mask[: len(values)] = True
        else:
            # Keep Transformer attention numerically defined for a pathological
            # empty point cloud; the zero point is distinguishable by rel_time.
            mask[0] = True
        return points, mask

    def _load_eo(self, row: Mapping[str, Any]) -> tuple[torch.Tensor, float]:
        if not int(row.get("has_eo", 0)):
            return torch.zeros(3, self.cfg.eo_size, self.cfg.eo_size), 0.0
        path = indexed_path(self.data_root, self.split, str(row["eo_jpg_relpath"]))
        with Image.open(path) as image:
            image = image.convert("RGB").resize((self.cfg.eo_size, self.cfg.eo_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        # ImageNet normalization remains fixed and does not leak any fold labels.
        array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
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
        allow_mask = np.zeros(SLOT_COUNT, dtype=np.float32)
        allow_mask[allowlist] = 1.0

        if self.labels is not None:
            label = self.labels[source_index]
            presence = label["presence"].astype(np.float32).copy()
            positions = label["positions"].astype(np.float32).copy()
            count_target = int(presence.sum()) - 1
        else:
            presence = np.zeros(SLOT_COUNT, dtype=np.float32)
            positions = np.zeros((SLOT_COUNT, 3), dtype=np.float32)
            count_target = -100

        if self.labels is not None:
            # Unused slots must remain numerically inert in vectorized losses.
            positions[presence <= 0] = 0.0

        return {
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
            "source_index": torch.tensor(source_index, dtype=torch.long),
            "raw_iq": raw_iq,
            "stft": stft,
            "rf_scalars": rf_scalars,
            "node_present": node_present,
            "node_enu": torch.from_numpy(self.node_enu.copy()),
            "radar_points": torch.from_numpy(radar_points),
            "radar_mask": torch.from_numpy(radar_mask),
            "eo_image": eo_image,
            "has_eo": torch.tensor(has_eo, dtype=torch.float32),
            "allowlist_mask": torch.from_numpy(allow_mask),
            "presence_target": torch.from_numpy(presence),
            "position_target": torch.from_numpy(positions),
            "count_target": torch.tensor(count_target, dtype=torch.long),
        }


def make_dataloader(dataset: Dataset[Any], optim_cfg: OptimConfig, training: bool) -> DataLoader[Any]:
    workers = int(optim_cfg.workers)
    kwargs: dict[str, Any] = {
        "batch_size": int(optim_cfg.batch_size if training else optim_cfg.eval_batch_size),
        "shuffle": bool(training),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        # Recreating eval workers is intentional: each parity/TTA pass receives
        # the current deterministic crop state instead of a stale worker copy.
        "persistent_workers": False,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)
