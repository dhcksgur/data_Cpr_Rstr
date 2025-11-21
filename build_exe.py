"""Utility to generate a standalone Windows executable for the GUI.

Usage:
    python build_exe.py

This script wraps PyInstaller with the right options for a single-file, windowed
build of ``waveform_tool_gui.py`` so the tool can run without Python installed.
"""
from __future__ import annotations

from pathlib import Path

import PyInstaller.__main__

PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = PROJECT_ROOT / "waveform_tool_gui.py"


def build() -> None:
    PyInstaller.__main__.run(
        [
            str(MAIN_SCRIPT),
            "--onefile",
            "--windowed",
            "--name=WaveformToolkit",
            "--noconfirm",
            "--clean",
        ]
    )


if __name__ == "__main__":
    build()
