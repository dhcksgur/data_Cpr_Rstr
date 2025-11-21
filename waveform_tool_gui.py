"""Graphical interface for waveform resampling, compression, and restoration.

This tool wraps the existing command-line utilities and exposes them in a
single window with three tabs built on Tkinter (no third‑party GUI license):

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
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
from matplotlib import font_manager
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
def _clear_canvas(canvas_frame: tk.Frame) -> None:
    for child in canvas_frame.winfo_children():
        child.destroy()


def _draw_figure_on_canvas(canvas_frame: tk.Frame, figure):
    _clear_canvas(canvas_frame)
    fig_canvas = FigureCanvasTkAgg(figure, master=canvas_frame)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    return fig_canvas


def _configure_korean_fonts() -> str:
    """Pick an available Korean-capable font for both Tk and Matplotlib."""

    candidates = [
        "Malgun Gothic",
        "맑은 고딕",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
    ]
    chosen = None
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except Exception:  # noqa: BLE001
            continue
        chosen = name
        break

    if chosen:
        matplotlib.rcParams["font.family"] = chosen
    matplotlib.rcParams["axes.unicode_minus"] = False

    try:
        default_font = tkfont.nametofont("TkDefaultFont")
        if chosen:
            default_font.configure(family=chosen)
    except tk.TclError:
        pass

    return chosen or "default"


# ---------- Core operations ----------
def handle_resample(values: dict) -> None:
    try:
        settings = SampleSettings(
            frequency_hz=float(values.get("resample_freq", 0)),
            samples_per_cycle=int(values.get("resample_samples", 0)),
        )
        config = ResampleConfig(
            input_path=Path(values["resample_input"]),
            output_path=Path(values["resample_output"]),
            sample_rate=settings.sample_rate,
            time_column=values["resample_time_col"],
            value_column=values["resample_value_col"],
        )
        resample_waveform(config)
        messagebox.showinfo("완료", "재샘플링이 완료되었습니다.")
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("오류", f"재샘플링 중 오류 발생: {exc}")


def handle_compress(values: dict) -> None:
    try:
        def _collect_thresholds(prefix: str, default: float) -> list[float]:
            collected: list[float] = []
            for idx in range(1, 4):
                raw = values.get(f"{prefix}{idx}", "").strip()
                collected.append(float(raw) if raw else float(default))
            return collected

        settings = SampleSettings(
            frequency_hz=float(values.get("compress_freq", 0)),
            samples_per_cycle=int(values.get("compress_samples", 0)),
        )
        time_col = values.get("compress_time_col", "").strip()
        event_channel = int(values.get("event_channel", 0))
        output_json = Path(values["compress_output"])
        nrmse_path = values.get("nrmse_output", "").strip()
        nrmse_output = (
            Path(nrmse_path)
            if nrmse_path
            else output_json.with_name(f"{output_json.stem}_nrmse.csv")
        )
        normal_thresholds = _collect_thresholds("normal_thresh", 0.05)
        event_thresholds = _collect_thresholds("event_thresh", 0.08)
        raw_thresholds = _collect_thresholds("raw_thresh", 0.15)
        config = CompressionConfig(
            input_paths=[
                Path(values["compress_ch1"]),
                Path(values["compress_ch2"]),
                Path(values["compress_ch3"]),
            ],
            output_path=output_json,
            nrmse_output=nrmse_output,
            channels=[
                values["channel1_name"],
                values["channel2_name"],
                values["channel3_name"],
            ],
            value_columns=[
                values["channel1_col"],
                values["channel2_col"],
                values["channel3_col"],
            ],
            samples_per_cycle=settings.samples_per_cycle,
            sample_rate=settings.sample_rate,
            time_column=time_col or None,
            normal_thresholds=normal_thresholds,
            event_thresholds=event_thresholds,
            event_channel=(None if event_channel == 0 else max(event_channel - 1, 0)),
            raw_thresholds=raw_thresholds,
            boundary_cycles=int(values.get("boundary_cycles", 0)),
        )
        payload = compress_waveforms(config)
        config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        dropped = payload.get("metadata", {}).get("dropped_samples", 0)
        note = ""
        if dropped:
            note = (
                f"\n마지막 {dropped}개 샘플은 "
                f"{settings.samples_per_cycle}배수에 맞춰 잘려 저장되었습니다."
            )
        messagebox.showinfo(
            "완료",
            "압축이 완료되었습니다."
            f"\nNRMSE CSV: {nrmse_output}"
            f"{note}",
        )
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("오류", f"압축 중 오류 발생: {exc}")


def _plot_templates(channels: list[str], templates: dict[str, list[float]]):
    fig_height = 2.8 * len(channels) + 1.2
    fig, axes = plt.subplots(len(channels), 1, figsize=(7, fig_height))
    if len(channels) == 1:
        axes = [axes]
    for ax, name in zip(axes, channels):
        ax.plot(templates[name])
        ax.set_title(f"대표파형: {name}")
        ax.set_xlabel("샘플")
        ax.set_ylabel("값")
        ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=1.2, h_pad=1.0)
    return fig


def _extract_abnormal_segments(cycles: list[dict]):
    segments: list[dict] = []
    current: list[dict] = []
    start_idx: int | None = None

    for entry in cycles:
        kind = str(entry.get("kind", ""))
        idx = int(entry.get("index", -1))
        is_abn = kind.startswith("abnormal")
        if is_abn:
            if start_idx is None:
                start_idx = idx
            current.append(entry)
        elif start_idx is not None:
            segments.append({
                "start": start_idx,
                "end": idx - 1,
                "entries": current,
            })
            current = []
            start_idx = None

    if start_idx is not None:
        segments.append({
            "start": start_idx,
            "end": current[-1].get("index", start_idx),
            "entries": current,
        })
    return segments


def _segment_waveforms(segment: dict, channels: list[str], templates: dict[str, list[float]]):
    template_matrix = np.stack([templates[name] for name in channels], axis=0)
    cycles: list[np.ndarray] = []
    for entry in segment.get("entries", []):
        if "waveforms" in entry:
            cycles.append(
                np.array([entry["waveforms"][name] for name in channels], dtype=float)
            )
        elif "gains" in entry:
            gains = np.array([entry["gains"][name] for name in channels], dtype=float)
            cycles.append(gains[:, None] * template_matrix)
    if not cycles:
        return None
    return np.concatenate(cycles, axis=1)


def _plot_abnormal_segments(
    channels: list[str],
    templates: dict[str, list[float]],
    segments: list[dict],
    selected_idx: int | None = None,
):
    fig_height = 2.8 * len(channels) + 1.2
    fig, axes = plt.subplots(len(channels), 1, figsize=(7, fig_height))
    if len(channels) == 1:
        axes = [axes]

    targets = segments if selected_idx is None else segments[selected_idx : selected_idx + 1]
    plotted = False

    for seg_no, segment in enumerate(targets, start=1):
        waveform_matrix = _segment_waveforms(segment, channels, templates)
        if waveform_matrix is None:
            continue
        x = np.arange(waveform_matrix.shape[1])
        for ax, name, row in zip(axes, channels, waveform_matrix):
            label = f"구간 {segments.index(segment) + 1}" if selected_idx is None else f"구간 {selected_idx + 1}"
            ax.plot(x, row, label=label)
            ax.set_title(f"이상 구간 파형: {name}")
            ax.set_xlabel("샘플")
            ax.set_ylabel("값")
            ax.grid(True, alpha=0.3)
        plotted = True

    if plotted:
        for ax in axes:
            ax.legend()
    else:
        for ax, name in zip(axes, channels):
            ax.set_title(f"이상 구간 없음: {name}")
            ax.axis("off")

    fig.tight_layout(pad=1.2, h_pad=1.0)
    return fig


def load_archive_for_preview(path: Path):
    try:
        metadata, templates, cycles = _load_archive(path)
        channels = list(metadata.get("channels", templates.keys()))
        return channels, templates, cycles
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("오류", f"압축 파일을 읽는 중 오류가 발생했습니다: {exc}")
        return None


class WaveformApp:
    """Tkinter GUI wrapper for waveform utilities."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Waveform Toolkit")
        self.values: dict[str, str] = {}
        self.figures: dict[str, FigureCanvasTkAgg] = {}
        self.entries: dict[str, ttk.Entry] = {}
        self.preview: dict | None = None
        self.segment_var = tk.StringVar(value="전체")

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        notebook.add(self._build_resample_tab(notebook), text="오실로스코프 파형 샘플링 추출")
        notebook.add(self._build_compress_tab(notebook), text="CSV 데이터 압축")
        notebook.add(self._build_decompress_tab(notebook), text="압축 데이터 CSV 복원")

    # ---------- helpers ----------
    def _add_labeled_entry(self, parent: ttk.Frame, row: int, label: str, key: str, default: str = "", width: int = 12) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        entry = ttk.Entry(parent, width=width)
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        self.values[key] = default
        entry.bind("<FocusOut>", lambda _e, k=key, widget=entry: self._update_value(k, widget.get()))
        entry.bind("<KeyRelease>", lambda _e, k=key, widget=entry: self._update_value(k, widget.get()))
        return entry

    def _update_value(self, key: str, value: str) -> None:
        self.values[key] = value

    def _browse_file(self, key: str, filetypes=None) -> None:
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.values[key] = path
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, path)

    def _save_file(self, key: str, defaultextension: str, filetypes=None) -> None:
        path = filedialog.asksaveasfilename(defaultextension=defaultextension, filetypes=filetypes)
        if path:
            self.values[key] = path
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, path)

    # ---------- tab builders ----------
    def _build_resample_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        frame.columnconfigure(1, weight=1)

        self.entries["resample_input"] = self._add_labeled_entry(frame, 0, "원본 CSV 파일", "resample_input", width=40)
        ttk.Button(frame, text="찾기", command=lambda: self._browse_file("resample_input", [("CSV", "*.csv"), ("모든 파일", "*.*")])).grid(row=0, column=2, padx=4, pady=2)

        self.entries["resample_output"] = self._add_labeled_entry(frame, 1, "출력 CSV 파일", "resample_output", width=40)
        ttk.Button(frame, text="저장 위치", command=lambda: self._save_file("resample_output", ".csv", [("CSV", "*.csv")])).grid(row=1, column=2, padx=4, pady=2)

        self.entries["resample_freq"] = self._add_labeled_entry(frame, 2, "기본 주파수(Hz)", "resample_freq", default="60")
        self.entries["resample_samples"] = self._add_labeled_entry(frame, 3, "주기당 샘플 수", "resample_samples", default="128")
        self.entries["resample_time_col"] = self._add_labeled_entry(frame, 4, "시간 컬럼", "resample_time_col", default="D")
        self.entries["resample_value_col"] = self._add_labeled_entry(frame, 5, "값 컬럼", "resample_value_col", default="E")

        ttk.Button(frame, text="재샘플링 실행", command=lambda: handle_resample(self.values)).grid(row=6, column=0, columnspan=3, sticky="ew", padx=4, pady=6)
        return frame

    def _build_compress_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        frame.columnconfigure(1, weight=1)

        for idx in range(1, 4):
            key = f"compress_ch{idx}"
            self.entries[key] = self._add_labeled_entry(frame, idx - 1, f"채널 {idx} CSV", key, width=40)
            ttk.Button(frame, text="찾기", command=lambda k=key: self._browse_file(k, [("CSV", "*.csv"), ("모든 파일", "*.*")])).grid(row=idx - 1, column=2, padx=4, pady=2)

        self.entries["compress_output"] = self._add_labeled_entry(frame, 3, "출력 JSON", "compress_output", width=40)
        ttk.Button(frame, text="저장 위치", command=lambda: self._save_file("compress_output", ".json", [("JSON", "*.json"), ("모든 파일", "*.*")])).grid(row=3, column=2, padx=4, pady=2)

        self.entries["nrmse_output"] = self._add_labeled_entry(frame, 4, "NRMSE CSV(비우면 자동)", "nrmse_output", width=40)
        ttk.Button(frame, text="저장 위치", command=lambda: self._save_file("nrmse_output", ".csv", [("CSV", "*.csv"), ("모든 파일", "*.*")])).grid(row=4, column=2, padx=4, pady=2)

        self.entries["compress_freq"] = self._add_labeled_entry(frame, 5, "기본 주파수(Hz)", "compress_freq", default="60")
        self.entries["compress_samples"] = self._add_labeled_entry(frame, 6, "주기당 샘플 수", "compress_samples", default="128")
        self.entries["boundary_cycles"] = self._add_labeled_entry(frame, 7, "경계 주기", "boundary_cycles", default="3")
        self.entries["event_channel"] = self._add_labeled_entry(
            frame,
            8,
            "이상 감지 채널(1~3, 0=전체)",
            "event_channel",
            default="2",
        )
        self.entries["compress_time_col"] = self._add_labeled_entry(
            frame,
            9,
            "시간 컬럼(비우면 무시)",
            "compress_time_col",
            default="",
        )

        names_frame = ttk.Frame(frame)
        names_frame.grid(row=10, column=0, columnspan=3, sticky="ew", padx=4, pady=2)
        ttk.Label(names_frame, text="채널 이름").grid(row=0, column=0, sticky="w", padx=4)
        for idx, default in enumerate(["ch1", "ch2", "ch3"], start=1):
            key = f"channel{idx}_name"
            entry = ttk.Entry(names_frame, width=8)
            entry.insert(0, default)
            entry.grid(row=0, column=idx, padx=4)
            self.entries[key] = entry
            self.values[key] = default
            entry.bind("<KeyRelease>", lambda _e, k=key, widget=entry: self._update_value(k, widget.get()))

        columns_frame = ttk.Frame(frame)
        columns_frame.grid(row=11, column=0, columnspan=3, sticky="ew", padx=4, pady=2)
        ttk.Label(columns_frame, text="값 컬럼").grid(row=0, column=0, sticky="w", padx=4)
        for idx in range(1, 4):
            key = f"channel{idx}_col"
            entry = ttk.Entry(columns_frame, width=8)
            entry.insert(0, "value")
            entry.grid(row=0, column=idx, padx=4)
            self.entries[key] = entry
            self.values[key] = "value"
            entry.bind("<KeyRelease>", lambda _e, k=key, widget=entry: self._update_value(k, widget.get()))

        thresholds_frame = ttk.Frame(frame)
        thresholds_frame.grid(row=12, column=0, columnspan=3, sticky="ew", padx=4, pady=2)
        ttk.Label(thresholds_frame, text="채널별 NRMSE 임계값").grid(row=0, column=0, columnspan=4, sticky="w", padx=4)

        ttk.Label(thresholds_frame, text="CH").grid(row=1, column=0, padx=4)
        for idx in range(1, 4):
            ttk.Label(thresholds_frame, text=str(idx)).grid(row=1, column=idx, padx=4)

        defaults = {"normal": "0.05", "event": "0.08", "raw": "0.15"}
        labels = [("normal", "정상"), ("event", "이상"), ("raw", "RAW")]
        for row_offset, (prefix, label) in enumerate(labels, start=2):
            ttk.Label(thresholds_frame, text=f"{label} NRMSE").grid(row=row_offset, column=0, padx=4, sticky="e")
            for ch in range(1, 4):
                key = f"{prefix}_thresh{ch}"
                entry = ttk.Entry(thresholds_frame, width=8)
                entry.insert(0, defaults[prefix])
                entry.grid(row=row_offset, column=ch, padx=4)
                self.entries[key] = entry
                self.values[key] = defaults[prefix]
                entry.bind(
                    "<KeyRelease>",
                    lambda _e, k=key, widget=entry: self._update_value(k, widget.get()),
                )

        ttk.Button(frame, text="압축 실행", command=lambda: handle_compress(self.values)).grid(row=13, column=0, columnspan=3, sticky="ew", padx=4, pady=6)
        return frame

    def _build_decompress_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        frame.columnconfigure(1, weight=1)

        self.entries["decompress_input"] = self._add_labeled_entry(frame, 0, "압축 JSON", "decompress_input", width=40)
        ttk.Button(frame, text="찾기", command=lambda: self._browse_file("decompress_input", [("JSON", "*.json"), ("모든 파일", "*.*")])).grid(row=0, column=2, padx=4, pady=2)

        self.entries["decompress_output"] = self._add_labeled_entry(frame, 1, "출력 CSV", "decompress_output", width=40)
        ttk.Button(frame, text="저장 위치", command=lambda: self._save_file("decompress_output", ".csv", [("CSV", "*.csv"), ("모든 파일", "*.*")])).grid(row=1, column=2, padx=4, pady=2)

        self.entries["time_column"] = self._add_labeled_entry(frame, 2, "시간 컬럼 이름", "time_column", default="time")

        ttk.Button(frame, text="복원 및 미리보기", command=self._handle_decompress).grid(row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=6)

        ttk.Label(frame, text="이상 구간 선택").grid(row=4, column=0, sticky="w", padx=4)
        self.segment_combo = ttk.Combobox(
            frame,
            textvariable=self.segment_var,
            state="readonly",
            values=["전체"],
            width=20,
        )
        self.segment_combo.grid(row=4, column=1, sticky="w", padx=4)
        self.segment_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_abnormal_plot())

        ttk.Label(frame, text="대표파형 미리보기").grid(row=5, column=0, columnspan=3, sticky="w", padx=4)
        self.template_canvas = ttk.Frame(frame)
        self.template_canvas.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=4, pady=2)

        ttk.Label(frame, text="이상 구간 미리보기").grid(row=7, column=0, columnspan=3, sticky="w", padx=4)
        self.abnormal_canvas = ttk.Frame(frame)
        self.abnormal_canvas.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=4, pady=2)
        frame.rowconfigure(6, weight=1)
        frame.rowconfigure(8, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        return frame

    # ---------- decompress helpers ----------
    def _handle_decompress(self) -> None:
        archive_path = Path(self.values.get("decompress_input", ""))
        preview = load_archive_for_preview(archive_path)
        if preview is None:
            return
        channels, templates, cycles = preview

        try:
            config = DecompressionConfig(
                input_path=archive_path,
                output_path=Path(self.values.get("decompress_output", "")),
                time_column=self.values.get("time_column", "time"),
            )
            df = decompress_waveforms(config)
            df.to_csv(config.output_path, index=False)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("오류", f"복원 중 오류 발생: {exc}")
            return

        segments = _extract_abnormal_segments(cycles)
        labels = ["전체"] + [f"구간 {idx + 1} ({seg['start']}~{seg['end']})" for idx, seg in enumerate(segments)]
        self.preview = {
            "channels": channels,
            "templates": templates,
            "segments": segments,
        }
        self.segment_combo.configure(values=labels)
        self.segment_var.set(labels[0])

        self.figures["templates"] = _draw_figure_on_canvas(
            self.template_canvas, _plot_templates(channels, templates)
        )
        self._update_abnormal_plot()
        messagebox.showinfo("완료", "복원이 완료되었습니다.")

    def _update_abnormal_plot(self) -> None:
        if not self.preview:
            return
        channels = self.preview["channels"]
        templates = self.preview["templates"]
        segments = self.preview["segments"]
        selection = self.segment_var.get()
        selected_idx: int | None = None
        if selection.startswith("구간"):
            try:
                selected_idx = int(selection.split()[1]) - 1
            except Exception:  # noqa: BLE001
                selected_idx = None

        self.figures["abnormal"] = _draw_figure_on_canvas(
            self.abnormal_canvas,
            _plot_abnormal_segments(channels, templates, segments, selected_idx),
        )


# ---------- Event loop ----------
def main(args: Iterable[str] | None = None) -> None:  # noqa: ARG001
    root = tk.Tk()
    chosen_font = _configure_korean_fonts()
    root.title(f"Waveform Toolkit ({chosen_font})")
    app = WaveformApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
