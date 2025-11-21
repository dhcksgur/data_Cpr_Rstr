"""Graphical interface for waveform resampling, compression, and restoration.

This tool wraps the existing command-line utilities and exposes them in a
single window with three tabs:

1. **오실로스코프 파형 샘플링 추출** — resamples raw CSV data to a uniform
   sampling rate based on the requested base frequency and samples per cycle.
2. **CSV 데이터 압축** — compresses three synchronized channels into a compact
   JSON archive while marking abnormal segments.
3. **압축 데이터 CSV 복원** — previews compressed data (templates and abnormal
   areas), then reconstructs a CSV file from the archive.

Run with ``python waveform_tool_gui.py``. For distribution, ``pyinstaller
--onefile --windowed waveform_tool_gui.py`` produces a standalone executable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from compress_waveforms import CompressionConfig, compress_waveforms
from decompress_waveforms import (
    DecompressionConfig,
    _load_archive,
    decompress_waveforms,
)
from resample_waveforms import ResampleConfig, resample_waveform

matplotlib.use("TkAgg")


@dataclass
class SampleSettings:
    frequency_hz: float
    samples_per_cycle: int

    @property
    def sample_rate(self) -> float:
        return max(self.frequency_hz * self.samples_per_cycle, 1e-9)


# ---------- Matplotlib helpers ----------
def _clear_canvas(canvas):
    if canvas.TKCanvas is None:
        return
    for child in canvas.TKCanvas.winfo_children():
        child.destroy()


def _draw_figure_on_canvas(canvas, figure):
    _clear_canvas(canvas)
    fig_canvas = FigureCanvasTkAgg(figure, master=canvas.TKCanvas)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
    return fig_canvas


# ---------- Core operations ----------
def handle_resample(values: dict) -> None:
    try:
        settings = SampleSettings(
            frequency_hz=float(values["resample_freq"]),
            samples_per_cycle=int(values["resample_samples"]),
        )
        config = ResampleConfig(
            input_path=Path(values["resample_input"]),
            output_path=Path(values["resample_output"]),
            sample_rate=settings.sample_rate,
            time_column=values["resample_time_col"],
            value_column=values["resample_value_col"],
        )
        resample_waveform(config)
        sg.popup("재샘플링이 완료되었습니다.")
    except Exception as exc:  # noqa: BLE001
        sg.popup_error(f"재샘플링 중 오류 발생: {exc}")


def handle_compress(values: dict) -> None:
    try:
        settings = SampleSettings(
            frequency_hz=float(values["compress_freq"]),
            samples_per_cycle=int(values["compress_samples"]),
        )
        config = CompressionConfig(
            input_paths=[
                Path(values["compress_ch1"]),
                Path(values["compress_ch2"]),
                Path(values["compress_ch3"]),
            ],
            output_path=Path(values["compress_output"]),
            channels=[values["channel1_name"], values["channel2_name"], values["channel3_name"]],
            value_columns=[values["channel1_col"], values["channel2_col"], values["channel3_col"]],
            samples_per_cycle=settings.samples_per_cycle,
            sample_rate=settings.sample_rate,
            normal_threshold=float(values["normal_thresh"]),
            event_threshold=float(values["event_thresh"]),
            raw_threshold=float(values["raw_thresh"]),
            boundary_cycles=int(values["boundary_cycles"]),
        )
        payload = compress_waveforms(config)
        config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        sg.popup("압축이 완료되었습니다.")
    except Exception as exc:  # noqa: BLE001
        sg.popup_error(f"압축 중 오류 발생: {exc}")


def _plot_templates(channels: list[str], templates: dict[str, list[float]]):
    fig, axes = plt.subplots(len(channels), 1, figsize=(6, 3 + len(channels)))
    if len(channels) == 1:
        axes = [axes]
    for ax, name in zip(axes, channels):
        ax.plot(templates[name])
        ax.set_title(f"대표파형: {name}")
        ax.set_xlabel("샘플")
        ax.set_ylabel("값")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_abnormal_segments(
    channels: list[str],
    templates: dict[str, list[float]],
    cycles: list[dict],
):
    template_matrix = np.stack([templates[name] for name in channels], axis=0)
    abnormal = []
    for entry in cycles:
        kind = entry.get("kind", "")
        if not str(kind).startswith("abnormal"):
            continue
        if "waveforms" in entry:
            abnormal.append(
                np.array([entry["waveforms"][name] for name in channels], dtype=float)
            )
        elif "gains" in entry:
            gains = np.array([entry["gains"][name] for name in channels], dtype=float)
            abnormal.append(gains[:, None] * template_matrix)
    fig, axes = plt.subplots(len(channels), 1, figsize=(6, 3 + len(channels)))
    if len(channels) == 1:
        axes = [axes]
    if abnormal:
        stacked = np.concatenate(abnormal, axis=1)
        x = np.arange(stacked.shape[1])
        for ax, name, row in zip(axes, channels, stacked):
            ax.plot(x, row)
            ax.set_title(f"이상 구간 파형: {name}")
            ax.set_xlabel("샘플")
            ax.set_ylabel("값")
            ax.grid(True, alpha=0.3)
    else:
        for ax, name in zip(axes, channels):
            ax.set_title(f"이상 구간 없음: {name}")
            ax.axis("off")
    fig.tight_layout()
    return fig


def load_archive_for_preview(path: Path):
    try:
        metadata, templates, cycles = _load_archive(path)
        channels = list(metadata.get("channels", templates.keys()))
        return channels, templates, cycles
    except Exception as exc:  # noqa: BLE001
        sg.popup_error(f"압축 파일을 읽는 중 오류가 발생했습니다: {exc}")
        return None


def handle_decompress(values: dict, template_canvas, abnormal_canvas, figures: dict) -> None:
    archive_path = Path(values["decompress_input"])
    preview = load_archive_for_preview(archive_path)
    if preview is None:
        return
    channels, templates, cycles = preview

    try:
        config = DecompressionConfig(
            input_path=archive_path,
            output_path=Path(values["decompress_output"]),
            time_column=values["time_column"],
        )
        df = decompress_waveforms(config)
        df.to_csv(config.output_path, index=False)
    except Exception as exc:  # noqa: BLE001
        sg.popup_error(f"복원 중 오류 발생: {exc}")
        return

    figures["templates"] = _draw_figure_on_canvas(
        template_canvas, _plot_templates(channels, templates)
    )
    figures["abnormal"] = _draw_figure_on_canvas(
        abnormal_canvas, _plot_abnormal_segments(channels, templates, cycles)
    )
    sg.popup("복원이 완료되었습니다.")


# ---------- GUI layouts ----------
def create_resample_tab() -> list[list[sg.Element]]:
    return [
        [sg.Text("원본 CSV 파일"), sg.Input(key="resample_input"), sg.FileBrowse()],
        [sg.Text("출력 CSV 파일"), sg.Input(key="resample_output"), sg.FileSaveAs(file_types=(("CSV", "*.csv"),))],
        [sg.Text("기본 주파수(Hz)"), sg.Input("60", key="resample_freq", size=(10, 1)),
         sg.Text("주기당 샘플 수"), sg.Input("128", key="resample_samples", size=(10, 1))],
        [sg.Text("시간 컬럼"), sg.Input("D", key="resample_time_col", size=(10, 1)),
         sg.Text("값 컬럼"), sg.Input("E", key="resample_value_col", size=(10, 1))],
        [sg.Button("재샘플링 실행", key="run_resample", expand_x=True)],
    ]


def create_compress_tab() -> list[list[sg.Element]]:
    return [
        [sg.Text("채널 1 CSV"), sg.Input(key="compress_ch1"), sg.FileBrowse()],
        [sg.Text("채널 2 CSV"), sg.Input(key="compress_ch2"), sg.FileBrowse()],
        [sg.Text("채널 3 CSV"), sg.Input(key="compress_ch3"), sg.FileBrowse()],
        [sg.Text("출력 JSON"), sg.Input(key="compress_output"), sg.FileSaveAs(file_types=(("JSON", "*.json"),))],
        [sg.Text("기본 주파수(Hz)"), sg.Input("60", key="compress_freq", size=(10, 1)),
         sg.Text("주기당 샘플 수"), sg.Input("128", key="compress_samples", size=(10, 1)),
         sg.Text("경계 주기"), sg.Input("3", key="boundary_cycles", size=(5, 1))],
        [sg.Text("채널 이름"),
         sg.Input("ch1", key="channel1_name", size=(8, 1)),
         sg.Input("ch2", key="channel2_name", size=(8, 1)),
         sg.Input("ch3", key="channel3_name", size=(8, 1))],
        [sg.Text("값 컬럼"),
         sg.Input("value", key="channel1_col", size=(8, 1)),
         sg.Input("value", key="channel2_col", size=(8, 1)),
         sg.Input("value", key="channel3_col", size=(8, 1))],
        [sg.Text("정상 NRMSE"), sg.Input("0.05", key="normal_thresh", size=(8, 1)),
         sg.Text("이상 NRMSE"), sg.Input("0.08", key="event_thresh", size=(8, 1)),
         sg.Text("RAW 임계"), sg.Input("0.15", key="raw_thresh", size=(8, 1))],
        [sg.Button("압축 실행", key="run_compress", expand_x=True)],
    ]


def create_decompress_tab() -> list[list[sg.Element]]:
    return [
        [sg.Text("압축 JSON"), sg.Input(key="decompress_input"), sg.FileBrowse(file_types=(("JSON", "*.json"),))],
        [sg.Text("출력 CSV"), sg.Input(key="decompress_output"), sg.FileSaveAs(file_types=(("CSV", "*.csv"),))],
        [sg.Text("시간 컬럼 이름"), sg.Input("time", key="time_column", size=(12, 1))],
        [sg.Button("복원 및 미리보기", key="run_decompress", expand_x=True)],
        [sg.Text("대표파형 미리보기")],
        [sg.Canvas(key="template_canvas", size=(640, 320))],
        [sg.Text("이상 구간 미리보기")],
        [sg.Canvas(key="abnormal_canvas", size=(640, 320))],
    ]


def build_window() -> sg.Window:
    sg.theme("LightBlue3")
    layout = [
        [
            sg.TabGroup(
                [
                    [
                        sg.Tab("오실로스코프 파형 샘플링 추출", create_resample_tab()),
                        sg.Tab("CSV 데이터 압축", create_compress_tab()),
                        sg.Tab("압축 데이터 CSV 복원", create_decompress_tab()),
                    ]
                ],
                expand_x=True,
                expand_y=True,
            )
        ]
    ]
    return sg.Window("Waveform Toolkit", layout, finalize=True, resizable=True)


# ---------- Event loop ----------
def main(args: Iterable[str] | None = None) -> None:  # noqa: ARG001
    window = build_window()
    figures: dict[str, FigureCanvasTkAgg] = {}

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        if event == "run_resample":
            handle_resample(values)
        elif event == "run_compress":
            handle_compress(values)
        elif event == "run_decompress":
            handle_decompress(values, window["template_canvas"], window["abnormal_canvas"], figures)

    window.close()


if __name__ == "__main__":
    main()
