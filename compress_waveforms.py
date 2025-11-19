"""Compression utility for three-channel waveform data.

This script groups resampled oscilloscope traces into 128-sample cycles and
stores them in a compact representation.  Cycles that closely follow a
representative waveform (normal operating conditions) are described through a
single scale factor per channel.  Abnormal regions – detected primarily through
channel 2 – retain up to ``boundary_cycles`` raw cycles at the beginning and the
end, while the interior of the event is compressed whenever its shape remains
stable.

The resulting file is a JSON document with three sections:

``metadata``
    Sampling information used by :mod:`decompress_waveforms` to rebuild the
    128-sample cycles.
``templates``
    The representative waveforms (one per channel) that are multiplied by the
    per-cycle gains.
``cycles``
    A list describing each cycle in chronological order.  Entries either
    contain raw samples (``kind = "abnormal_raw"`` or ``"raw"``) or only the
    gains/errors required to reproduce the original waveform.

The normal/abnormal decision is intentionally conservative and configurable
through CLI flags so the workflow can be tuned to new datasets without code
changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class CompressionConfig:
    input_path: Path
    output_path: Path
    channels: Sequence[str]
    event_channel: str
    samples_per_cycle: int = 128
    sample_rate: float = 128.0
    time_column: str | None = "time"
    normal_threshold: float = 0.05
    event_threshold: float = 0.08
    raw_threshold: float = 0.15
    boundary_cycles: int = 3


def parse_args(args: Iterable[str] | None = None) -> CompressionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV file containing 3 channels")
    parser.add_argument("output", type=Path, help="Path to the JSON archive")
    parser.add_argument(
        "--channels",
        nargs=3,
        metavar=("CH1", "CH2", "CH3"),
        default=["ch1", "ch2", "ch3"],
        help="Names of the three value columns inside the CSV file",
    )
    parser.add_argument(
        "--event-channel",
        default="ch2",
        help="Channel used to delimit abnormal segments",
    )
    parser.add_argument(
        "--samples-per-cycle",
        type=int,
        default=128,
        help="Number of samples per electrical cycle (default: 128)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=128.0,
        help="Sampling rate of the input data (used for reconstruction)",
    )
    parser.add_argument(
        "--time-column",
        default="time",
        help="Optional column with timestamps (default: 'time')",
    )
    parser.add_argument(
        "--normal-threshold",
        type=float,
        default=0.05,
        help="Max. NRMSE for a cycle to be considered normal",
    )
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=0.08,
        help="NRMSE (channel 2) above which an abnormal segment starts",
    )
    parser.add_argument(
        "--raw-threshold",
        type=float,
        default=0.15,
        help="NRMSE above which the full cycle is stored as raw samples",
    )
    parser.add_argument(
        "--boundary-cycles",
        type=int,
        default=3,
        help="Number of raw cycles kept at each edge of an event",
    )

    ns = parser.parse_args(args)
    return CompressionConfig(
        input_path=ns.input,
        output_path=ns.output,
        channels=ns.channels,
        event_channel=ns.event_channel,
        samples_per_cycle=ns.samples_per_cycle,
        sample_rate=ns.sample_rate,
        time_column=ns.time_column,
        normal_threshold=ns.normal_threshold,
        event_threshold=ns.event_threshold,
        raw_threshold=ns.raw_threshold,
        boundary_cycles=ns.boundary_cycles,
    )


def _load_channels(df: pd.DataFrame, channel_names: Sequence[str]) -> np.ndarray:
    missing = [name for name in channel_names if name not in df]
    if missing:
        raise KeyError(f"Missing channels in CSV: {missing}")
    return df[channel_names].to_numpy(dtype=float)


def _reshape_cycles(values: np.ndarray, samples_per_cycle: int) -> np.ndarray:
    total_samples, num_channels = values.shape
    if total_samples % samples_per_cycle:
        raise ValueError(
            "Number of samples is not divisible by samples_per_cycle. "
            f"Got {total_samples} samples for {samples_per_cycle}-sample cycles."
        )
    num_cycles = total_samples // samples_per_cycle
    reshaped = values.reshape(num_cycles, samples_per_cycle, num_channels)
    return np.transpose(reshaped, (0, 2, 1))  # (cycles, channels, samples)


def _compute_templates(waveforms: np.ndarray) -> np.ndarray:
    return waveforms.mean(axis=0)


def _cycle_gains_and_errors(
    waveforms: np.ndarray, templates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    num_cycles, num_channels, samples = waveforms.shape
    gains = np.zeros((num_cycles, num_channels), dtype=float)
    errors = np.zeros_like(gains)
    template_norm_sq = np.sum(templates * templates, axis=1)
    template_rms = np.sqrt(np.mean(templates * templates, axis=1))

    for cycle in range(num_cycles):
        current = waveforms[cycle]
        for ch in range(num_channels):
            template = templates[ch]
            numerator = float(np.dot(current[ch], template))
            gains[cycle, ch] = numerator / (template_norm_sq[ch] + EPS)
            approx = gains[cycle, ch] * template
            diff = current[ch] - approx
            rms = np.sqrt(np.mean(diff * diff))
            errors[cycle, ch] = rms / (template_rms[ch] + EPS)
    return gains, errors


def _find_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _scaled_entry(
    kind: str,
    idx: int,
    channels: Sequence[str],
    gains: np.ndarray,
    errors: np.ndarray,
) -> dict:
    return {
        "index": idx,
        "kind": kind,
        "gains": {name: float(gains[idx, ch]) for ch, name in enumerate(channels)},
        "errors": {name: float(errors[idx, ch]) for ch, name in enumerate(channels)},
    }


def _raw_entry(
    kind: str,
    idx: int,
    channels: Sequence[str],
    waveforms: np.ndarray,
    errors: np.ndarray,
) -> dict:
    return {
        "index": idx,
        "kind": kind,
        "waveforms": {
            name: waveforms[idx, ch].astype(float).tolist()
            for ch, name in enumerate(channels)
        },
        "errors": {name: float(errors[idx, ch]) for ch, name in enumerate(channels)},
    }


def compress_waveforms(config: CompressionConfig) -> dict:
    df = pd.read_csv(config.input_path)
    values = _load_channels(df, config.channels)
    waveforms = _reshape_cycles(values, config.samples_per_cycle)
    templates = _compute_templates(waveforms)
    gains, errors = _cycle_gains_and_errors(waveforms, templates)

    event_idx = config.channels.index(config.event_channel)
    event_mask = errors[:, event_idx] >= config.event_threshold
    segments = _find_segments(event_mask)

    cycles: list[dict] = []
    num_cycles = waveforms.shape[0]
    pointer = 0

    for start, end in segments:
        while pointer < start:
            if errors[pointer].max() <= config.normal_threshold:
                cycles.append(
                    _scaled_entry("normal", pointer, config.channels, gains, errors)
                )
            else:
                cycles.append(
                    _raw_entry("raw", pointer, config.channels, waveforms, errors)
                )
            pointer += 1

        for idx in range(start, end + 1):
            distance = min(idx - start, end - idx)
            force_raw = distance < config.boundary_cycles
            store_raw = force_raw or errors[idx].max() >= config.raw_threshold
            if store_raw:
                kind = "abnormal_raw"
                cycles.append(
                    _raw_entry(kind, idx, config.channels, waveforms, errors)
                )
            else:
                kind = "abnormal_scaled"
                cycles.append(
                    _scaled_entry(kind, idx, config.channels, gains, errors)
                )
        pointer = end + 1

    while pointer < num_cycles:
        if errors[pointer].max() <= config.normal_threshold:
            cycles.append(
                _scaled_entry("normal", pointer, config.channels, gains, errors)
            )
        else:
            cycles.append(_raw_entry("raw", pointer, config.channels, waveforms, errors))
        pointer += 1

    time_col = config.time_column
    if time_col and time_col in df:
        times = df[time_col].to_numpy(dtype=float)
        time_start = float(times[0]) if len(times) else 0.0
        if len(times) > 1:
            dt = float(np.median(np.diff(times)))
        else:
            dt = 1.0 / config.sample_rate
    else:
        time_start = 0.0
        dt = 1.0 / config.sample_rate

    payload = {
        "metadata": {
            "channels": list(config.channels),
            "event_channel": config.event_channel,
            "samples_per_cycle": config.samples_per_cycle,
            "sample_rate": config.sample_rate,
            "normal_threshold": config.normal_threshold,
            "event_threshold": config.event_threshold,
            "raw_threshold": config.raw_threshold,
            "boundary_cycles": config.boundary_cycles,
            "time_start": time_start,
            "dt": dt,
        },
        "templates": {
            name: templates[ch].astype(float).tolist()
            for ch, name in enumerate(config.channels)
        },
        "cycles": cycles,
    }

    return payload


def main() -> None:
    config = parse_args()
    payload = compress_waveforms(config)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
