"""Compression utility for three-channel waveform data.

This script groups resampled oscilloscope traces into 128-sample cycles and
stores them in a compact representation.  Cycles that closely follow a
representative waveform (normal operating conditions) are described through a
single scale factor per channel.  Abnormal regions – detected independently on
each channel – retain up to ``boundary_cycles`` raw cycles at the beginning and
the end, while the interior of the event is compressed whenever its shape
remains stable.

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
    input_paths: Sequence[Path]
    output_path: Path
    channels: Sequence[str]
    value_columns: Sequence[str]
    samples_per_cycle: int = 128
    sample_rate: float = 128.0
    time_column: str | None = "time"
    normal_threshold: float = 0.05
    event_threshold: float = 0.08
    event_channel: int | None = 1
    raw_threshold: float = 0.15
    boundary_cycles: int = 3
    nrmse_output: Path | None = None


def parse_args(args: Iterable[str] | None = None) -> CompressionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs=3,
        metavar=("CH1_CSV", "CH2_CSV", "CH3_CSV"),
        help="Resampled CSV files for each channel",
    )
    parser.add_argument("output", type=Path, help="Path to the JSON archive")
    parser.add_argument(
        "--nrmse-csv",
        type=Path,
        default=None,
        help="Optional path for per-cycle NRMSE export (default: <output>_nrmse.csv)",
    )
    parser.add_argument(
        "--channels",
        nargs=3,
        metavar=("CH1", "CH2", "CH3"),
        default=["ch1", "ch2", "ch3"],
        help="Logical names assigned to the three channels in the archive",
    )
    parser.add_argument(
        "--value-columns",
        nargs=3,
        metavar=("VAL1", "VAL2", "VAL3"),
        default=["value", "value", "value"],
        help="Name of the value column inside each CSV file",
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
        "--event-channel",
        type=int,
        default=2,
        help="1-based channel index used to detect abnormal events (0 to use any)",
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
        input_paths=ns.inputs,
        output_path=ns.output,
        nrmse_output=ns.nrmse_csv,
        channels=ns.channels,
        value_columns=ns.value_columns,
        samples_per_cycle=ns.samples_per_cycle,
        sample_rate=ns.sample_rate,
        time_column=ns.time_column,
        normal_threshold=ns.normal_threshold,
        event_threshold=ns.event_threshold,
        event_channel=(None if ns.event_channel == 0 else ns.event_channel - 1),
        raw_threshold=ns.raw_threshold,
        boundary_cycles=ns.boundary_cycles,
    )


def _load_channels(
    input_paths: Sequence[Path],
    value_columns: Sequence[str],
    time_column: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(input_paths) != 3:
        raise ValueError("Exactly three CSV files must be provided.")

    if len(value_columns) != 3:
        raise ValueError("Three value column names are required.")

    value_arrays: list[np.ndarray] = []
    times: np.ndarray | None = None
    expected_len: int | None = None

    for idx, (path, value_col) in enumerate(zip(input_paths, value_columns)):
        df = pd.read_csv(path)
        if value_col not in df:
            raise KeyError(f"Column '{value_col}' missing from {path}")

        values = df[value_col].to_numpy(dtype=float)
        if expected_len is None:
            expected_len = len(values)
        elif len(values) != expected_len:
            raise ValueError("All input CSV files must contain the same number of rows.")
        value_arrays.append(values)

        if time_column and time_column in df:
            candidate = df[time_column].to_numpy(dtype=float)
            if times is None:
                times = candidate
            else:
                if len(candidate) != len(times):
                    raise ValueError(
                        "Time column lengths differ across CSV files."
                    )
                if not np.allclose(candidate, times):
                    raise ValueError("Time columns do not match across CSV files.")
        elif time_column:
            raise KeyError(f"Time column '{time_column}' missing from {path}")

    stacked = np.stack(value_arrays, axis=1)  # (samples, channels)
    return stacked, times


def _reshape_cycles(
    values: np.ndarray, samples_per_cycle: int
) -> tuple[np.ndarray, int, int]:
    """Reshape into cycles, trimming any trailing partial cycle."""

    original_samples, num_channels = values.shape
    remainder = original_samples % samples_per_cycle

    if original_samples < samples_per_cycle:
        raise ValueError(
            "Not enough samples to form a single cycle. "
            f"Got {original_samples} samples, need at least {samples_per_cycle}."
        )

    if remainder:
        values = values[: original_samples - remainder]

    total_samples = values.shape[0]
    num_cycles = total_samples // samples_per_cycle
    if num_cycles == 0:
        raise ValueError("No complete cycles after trimming partial samples.")

    reshaped = values.reshape(num_cycles, samples_per_cycle, num_channels)
    return np.transpose(reshaped, (0, 2, 1)), remainder, original_samples


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
    if len(config.channels) != 3:
        raise ValueError("Exactly three channel names must be provided.")

    values, times = _load_channels(
        config.input_paths, config.value_columns, config.time_column
    )
    waveforms, dropped_samples, original_samples = _reshape_cycles(
        values, config.samples_per_cycle
    )
    templates = _compute_templates(waveforms)
    gains, errors = _cycle_gains_and_errors(waveforms, templates)

    if config.event_channel is None:
        event_mask = (errors >= config.event_threshold).any(axis=1)
    elif config.event_channel < 0 or config.event_channel >= errors.shape[1]:
        raise ValueError("event_channel index is out of range for the provided data")
    else:
        event_mask = errors[:, config.event_channel] >= config.event_threshold
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

    if times is not None and len(times):
        time_start = float(times[0])
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
            "value_columns": list(config.value_columns),
            "samples_per_cycle": config.samples_per_cycle,
            "sample_rate": config.sample_rate,
            "original_samples": original_samples,
            "used_samples": int(waveforms.shape[0] * config.samples_per_cycle),
            "dropped_samples": dropped_samples,
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

    nrmse_path = config.nrmse_output or config.output_path.with_name(
        f"{config.output_path.stem}_nrmse.csv"
    )
    nrmse_df = pd.DataFrame(errors, columns=config.channels)
    nrmse_df.insert(0, "cycle", np.arange(len(errors)))
    nrmse_df.to_csv(nrmse_path, index=False, encoding="utf-8")

    return payload


def main() -> None:
    config = parse_args()
    payload = compress_waveforms(config)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
