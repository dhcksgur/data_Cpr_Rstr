"""Waveform resampling utility.

This module converts the raw oscilloscope CSV exports into evenly spaced
samples that match the desired sampling frequency (128 Hz by default).

The oscilloscope exports contain the time stamps in column ``D`` and the
measured values in column ``E``.  Since those files are fairly large (because of
their high sampling frequency) this script creates a lightweight version where
exactly ``sample_rate`` samples are stored per second.  The script performs a
linear interpolation so that no information about the shape of the signal is
lost in the process.

Example usage::

    python resample_waveforms.py raw_voltage.csv voltage_128hz.csv \
        --sample-rate 128

The script accepts Excel-style column letters (``D``/``E``) as well as column
names when selecting the time and value columns.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def excel_column_to_index(column: str) -> int:
    """Return the zero-based index that corresponds to an Excel column letter.

    ``A`` becomes 0, ``B`` becomes 1 and so on.  Multi letter columns (``AA``)
    are also supported.
    """

    column = column.strip().upper()
    if not column.isalpha():  # ``column`` may represent something else (name)
        raise ValueError("Not an Excel column")

    index = 0
    for char in column:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def resolve_column(df: pd.DataFrame, selector: str) -> str:
    """Resolve ``selector`` into the name of a column inside ``df``."""

    selector = selector.strip()
    if selector in df.columns:
        return selector

    # Try to interpret the selector as an Excel-style column letter.
    try:
        idx = excel_column_to_index(selector)
    except ValueError:
        idx = None

    if idx is not None and 0 <= idx < len(df.columns):
        return df.columns[idx]

    # Finally allow numeric indexes (0 based).
    if selector.isdigit():
        idx = int(selector)
        if 0 <= idx < len(df.columns):
            return df.columns[idx]

    raise KeyError(
        f"Column selector '{selector}' does not match any column in the file."
    )


@dataclass
class ResampleConfig:
    input_path: Path
    output_path: Path
    sample_rate: float = 128.0
    time_column: str = "D"
    value_column: str = "E"


def parse_args(args: Iterable[str] | None = None) -> ResampleConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the original CSV file")
    parser.add_argument(
        "output",
        type=Path,
        help="Path where the resampled CSV file will be written",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=128.0,
        help="Target sampling rate in Hz (default: 128)",
    )
    parser.add_argument(
        "--time-column",
        default="D",
        help="Column that contains the timestamps (name or Excel letter)",
    )
    parser.add_argument(
        "--value-column",
        default="E",
        help="Column that contains the measured values (name or Excel letter)",
    )

    ns = parser.parse_args(args)
    return ResampleConfig(
        input_path=ns.input,
        output_path=ns.output,
        sample_rate=ns.sample_rate,
        time_column=ns.time_column,
        value_column=ns.value_column,
    )


def resample_waveform(config: ResampleConfig) -> None:
    df = pd.read_csv(config.input_path)

    time_col = resolve_column(df, config.time_column)
    value_col = resolve_column(df, config.value_column)

    times = df[time_col].to_numpy(dtype=float)
    values = df[value_col].to_numpy(dtype=float)

    if len(times) == 0:
        raise ValueError("Input CSV does not contain any rows.")

    start, end = times.min(), times.max()
    if end <= start:
        raise ValueError("Time column must contain increasing values.")

    num_samples = int(np.floor((end - start) * config.sample_rate)) + 1
    uniform_times = np.linspace(start, start + (num_samples - 1) / config.sample_rate, num_samples)

    resampled_values = np.interp(uniform_times, times, values)

    result = pd.DataFrame({"time": uniform_times, "value": resampled_values})
    result.to_csv(config.output_path, index=False)


def main() -> None:
    config = parse_args()
    resample_waveform(config)


if __name__ == "__main__":
    main()
