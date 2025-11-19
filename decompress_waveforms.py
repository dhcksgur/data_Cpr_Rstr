"""Restore waveform data from the JSON archive produced by
:mod:`compress_waveforms`.

The script multiplies the representative templates with the stored gains to
recreate the original cycles.  Whenever a cycle was stored as raw data the
exact samples are copied back into the output stream.  The reconstructed traces
are written as a CSV file containing the time column (if the compression step
had access to timestamps) followed by the channel columns.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class DecompressionConfig:
    input_path: Path
    output_path: Path
    time_column: str = "time"


def parse_args(args: Iterable[str] | None = None) -> DecompressionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON archive to decompress")
    parser.add_argument("output", type=Path, help="Target CSV file")
    parser.add_argument(
        "--time-column",
        default="time",
        help="Name of the time column (set to '' to skip)",
    )
    ns = parser.parse_args(args)
    return DecompressionConfig(
        input_path=ns.input,
        output_path=ns.output,
        time_column=ns.time_column,
    )


def _load_archive(path: Path) -> tuple[dict, dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    templates = payload["templates"]
    cycles = payload["cycles"]
    return metadata, templates, cycles


def _template_matrix(
    channels: list[str], templates: dict[str, list[float]]
) -> np.ndarray:
    return np.stack([templates[name] for name in channels], axis=0)


def _reconstruct_cycles(
    channels: list[str],
    template_matrix: np.ndarray,
    cycles: list[dict],
) -> np.ndarray:
    samples = []
    for entry in cycles:
        if "waveforms" in entry:
            channel_samples = [entry["waveforms"][name] for name in channels]
            samples.append(np.asarray(channel_samples, dtype=float))
        elif "gains" in entry:
            gains = np.array([entry["gains"][name] for name in channels], dtype=float)
            samples.append(gains[:, None] * template_matrix)
        else:
            raise ValueError(f"Cycle entry lacks waveforms/gains fields: {entry}")
    stacked = np.concatenate(samples, axis=1)
    return stacked


def decompress_waveforms(config: DecompressionConfig) -> pd.DataFrame:
    metadata, templates, cycles = _load_archive(config.input_path)
    channels = list(metadata.get("channels", templates.keys()))
    template_matrix = _template_matrix(channels, templates)
    channel_data = _reconstruct_cycles(channels, template_matrix, cycles)

    num_samples = channel_data.shape[1]
    dt = float(metadata.get("dt", 1.0 / metadata.get("sample_rate", 128.0)))
    time_start = float(metadata.get("time_start", 0.0))

    data = {name: channel_data[idx] for idx, name in enumerate(channels)}
    if config.time_column:
        times = time_start + dt * np.arange(num_samples)
        data = {config.time_column: times, **data}

    return pd.DataFrame(data)


def main() -> None:
    config = parse_args()
    df = decompress_waveforms(config)
    df.to_csv(config.output_path, index=False)


if __name__ == "__main__":
    main()
