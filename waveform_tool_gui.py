"""Graphical interface for waveform resampling, compression, and restoration.

This tool wraps the existing command-line utilities and exposes them in a
single window with three tabs built on CustomTkinter:

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
from tkinter import filedialog, messagebox

import customtkinter as ctk
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
plt.style.use("seaborn-v0_8-darkgrid")

THEME_BG = "#0f172a"
THEME_PANEL = "#111827"
THEME_ACCENT = "#3b82f6"


@dataclass
class SampleSettings:
    frequency_hz: float
    samples_per_cycle: int

    @property
    def sample_rate(self) -> float:
        return max(self.frequency_hz * self.samples_per_cycle, 1e-9)


# ---------- Matplotlib helpers ----------
def _clear_canvas(canvas_frame: ctk.CTkFrame) -> None:
    for child in canvas_frame.winfo_children():
        child.destroy()


def _draw_figure_on_canvas(canvas_frame: ctk.CTkFrame, figure):
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


def _style_axes(ax) -> None:
    ax.set_facecolor(THEME_PANEL)
    for spine in ax.spines.values():
        spine.set_color("#7d8597")
    ax.tick_params(colors="#e0e6ed")
    ax.title.set_color("#e0e6ed")
    ax.xaxis.label.set_color("#e0e6ed")
    ax.yaxis.label.set_color("#e0e6ed")


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
    fig_height = 3.2 * len(channels) + 2.0
    fig, axes = plt.subplots(
        len(channels), 1, figsize=(8.5, fig_height), constrained_layout=True
    )
    if len(channels) == 1:
        axes = [axes]
    for ax, name in zip(axes, channels):
        ax.plot(templates[name], color=THEME_ACCENT, linewidth=2.0)
        ax.set_title(f"대표파형: {name}", pad=12)
        ax.set_xlabel("샘플", labelpad=8)
        ax.set_ylabel("값", labelpad=8)
        ax.grid(True, alpha=0.25)
        _style_axes(ax)
    fig.patch.set_facecolor(THEME_BG)
    fig.subplots_adjust(top=0.96, bottom=0.08, left=0.1, right=0.98, hspace=0.6)
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
    fig_height = 3.2 * len(channels) + 2.0
    fig, axes = plt.subplots(
        len(channels), 1, figsize=(8.5, fig_height), constrained_layout=True
    )
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
            label = (
                f"구간 {segments.index(segment) + 1}"
                if selected_idx is None
                else f"구간 {selected_idx + 1}"
            )
            ax.plot(x, row, label=label, linewidth=1.6, color=THEME_ACCENT)
            ax.set_title(f"이상 구간 파형: {name}", pad=12)
            ax.set_xlabel("샘플", labelpad=8)
            ax.set_ylabel("값", labelpad=8)
            ax.grid(True, alpha=0.25)
            _style_axes(ax)
        plotted = True

    if plotted:
        for ax in axes:
            ax.legend(facecolor=THEME_PANEL, framealpha=0.8, edgecolor="#7d8597", labelcolor="#e0e6ed")
    else:
        for ax, name in zip(axes, channels):
            ax.set_title(f"이상 구간 없음: {name}")
            ax.axis("off")

    fig.patch.set_facecolor(THEME_BG)
    fig.subplots_adjust(top=0.96, bottom=0.08, left=0.1, right=0.98, hspace=0.6)
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
    """CustomTkinter GUI wrapper for waveform utilities."""

    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Waveform Toolkit")
        self.root.configure(fg_color=THEME_BG)
        self.values: dict[str, str] = {}
        self.figures: dict[str, FigureCanvasTkAgg] = {}
        self.entries: dict[str, ctk.CTkEntry] = {}
        self.preview: dict | None = None
        self.segment_var = ctk.StringVar(value="전체")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        header = ctk.CTkLabel(
            self.root,
            text="Waveform Toolkit",
            font=("Segoe UI Semibold", 20),
            text_color="#e0e6ed",
        )
        header.pack(anchor="w", padx=16, pady=(10, 0))

        notebook = ctk.CTkTabview(
            self.root,
            fg_color=THEME_PANEL,
            segmented_button_fg_color="#0b1220",
            segmented_button_selected_color=THEME_ACCENT,
            segmented_button_unselected_color="#1f2937",
            segmented_button_border_color="#374151",
        )
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        resample_label = "오실로스코프 파형 샘플링 추출"
        compress_label = "CSV 데이터 압축"
        decompress_label = "압축 데이터 CSV 복원"
        notebook.add(resample_label)
        notebook.add(compress_label)
        notebook.add(decompress_label)

        self._build_resample_tab(notebook.tab(resample_label))
        self._build_compress_tab(notebook.tab(compress_label))
        self._build_decompress_tab(notebook.tab(decompress_label))

    # ---------- helpers ----------
    def _add_labeled_entry(
        self,
        parent: ctk.CTkFrame,
        row: int,
        label: str,
        key: str,
        default: str = "",
        width: int = 12,
    ) -> ctk.CTkEntry:
        ctk.CTkLabel(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        entry = ctk.CTkEntry(parent, width=width * 8)
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
    def _build_resample_tab(self, frame: ctk.CTkFrame) -> None:
        frame.configure(fg_color=THEME_PANEL)
        frame.grid_columnconfigure(1, weight=1)

        self.entries["resample_input"] = self._add_labeled_entry(frame, 0, "원본 CSV 파일", "resample_input", width=40)
        ctk.CTkButton(
            frame,
            text="찾기",
            command=lambda: self._browse_file("resample_input", [("CSV", "*.csv"), ("모든 파일", "*.*")]),
        ).grid(row=0, column=2, padx=4, pady=2)

        self.entries["resample_output"] = self._add_labeled_entry(frame, 1, "출력 CSV 파일", "resample_output", width=40)
        ctk.CTkButton(
            frame,
            text="저장 위치",
            command=lambda: self._save_file("resample_output", ".csv", [("CSV", "*.csv")]),
        ).grid(row=1, column=2, padx=4, pady=2)

        self.entries["resample_freq"] = self._add_labeled_entry(frame, 2, "기본 주파수(Hz)", "resample_freq", default="60")
        self.entries["resample_samples"] = self._add_labeled_entry(frame, 3, "주기당 샘플 수", "resample_samples", default="128")
        self.entries["resample_time_col"] = self._add_labeled_entry(frame, 4, "시간 컬럼", "resample_time_col", default="D")
        self.entries["resample_value_col"] = self._add_labeled_entry(frame, 5, "값 컬럼", "resample_value_col", default="E")

        ctk.CTkButton(frame, text="재샘플링 실행", command=lambda: handle_resample(self.values)).grid(
            row=6, column=0, columnspan=3, sticky="ew", padx=4, pady=6
        )

    def _build_compress_tab(self, frame: ctk.CTkFrame) -> None:
        frame.configure(fg_color=THEME_PANEL)
        frame.grid_columnconfigure(1, weight=1)

        for idx in range(1, 4):
            key = f"compress_ch{idx}"
            self.entries[key] = self._add_labeled_entry(frame, idx - 1, f"채널 {idx} CSV", key, width=40)
            ctk.CTkButton(
                frame,
                text="찾기",
                command=lambda k=key: self._browse_file(k, [("CSV", "*.csv"), ("모든 파일", "*.*")]),
            ).grid(row=idx - 1, column=2, padx=4, pady=2)

        self.entries["compress_output"] = self._add_labeled_entry(frame, 3, "출력 JSON", "compress_output", width=40)
        ctk.CTkButton(
            frame,
            text="저장 위치",
            command=lambda: self._save_file("compress_output", ".json", [("JSON", "*.json"), ("모든 파일", "*.*")]),
        ).grid(row=3, column=2, padx=4, pady=2)

        self.entries["nrmse_output"] = self._add_labeled_entry(frame, 4, "NRMSE CSV(비우면 자동)", "nrmse_output", width=40)
        ctk.CTkButton(
            frame,
            text="저장 위치",
            command=lambda: self._save_file("nrmse_output", ".csv", [("CSV", "*.csv"), ("모든 파일", "*.*")]),
        ).grid(row=4, column=2, padx=4, pady=2)

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

        names_frame = ctk.CTkFrame(frame)
        names_frame.grid(row=10, column=0, columnspan=3, sticky="ew", padx=4, pady=2)
        ctk.CTkLabel(names_frame, text="채널 이름").grid(row=0, column=0, sticky="w", padx=4)
        for idx, default in enumerate(["ch1", "ch2", "ch3"], start=1):
            key = f"channel{idx}_name"
            entry = ctk.CTkEntry(names_frame, width=70)
            entry.insert(0, default)
            entry.grid(row=0, column=idx, padx=4)
            self.entries[key] = entry
            self.values[key] = default
            entry.bind("<KeyRelease>", lambda _e, k=key, widget=entry: self._update_value(k, widget.get()))

        columns_frame = ctk.CTkFrame(frame)
        columns_frame.grid(row=11, column=0, columnspan=3, sticky="ew", padx=4, pady=2)
        ctk.CTkLabel(columns_frame, text="값 컬럼").grid(row=0, column=0, sticky="w", padx=4)
        for idx in range(1, 4):
            key = f"channel{idx}_col"
            entry = ctk.CTkEntry(columns_frame, width=70)
            entry.insert(0, "value")
            entry.grid(row=0, column=idx, padx=4)
            self.entries[key] = entry
            self.values[key] = "value"
            entry.bind("<KeyRelease>", lambda _e, k=key, widget=entry: self._update_value(k, widget.get()))

        thresholds_frame = ctk.CTkFrame(frame)
        thresholds_frame.grid(row=12, column=0, columnspan=3, sticky="ew", padx=4, pady=2)
        ctk.CTkLabel(thresholds_frame, text="채널별 NRMSE 임계값").grid(row=0, column=0, columnspan=4, sticky="w", padx=4)

        ctk.CTkLabel(thresholds_frame, text="CH").grid(row=1, column=0, padx=4)
        for idx in range(1, 4):
            ctk.CTkLabel(thresholds_frame, text=str(idx)).grid(row=1, column=idx, padx=4)

        defaults = {
            "normal": ("0.09", "0.9", "0.09"),
            "event": ("0.1", "1", "0.1"),
            "raw": ("0.2", "2", "0.2"),
        }
        labels = [("normal", "정상"), ("event", "이상"), ("raw", "RAW")]
        for row_offset, (prefix, label) in enumerate(labels, start=2):
            ctk.CTkLabel(thresholds_frame, text=f"{label} NRMSE").grid(row=row_offset, column=0, padx=4, sticky="e")
            for ch in range(1, 4):
                key = f"{prefix}_thresh{ch}"
                entry = ctk.CTkEntry(thresholds_frame, width=70)
                entry.insert(0, defaults[prefix][ch - 1])
                entry.grid(row=row_offset, column=ch, padx=4)
                self.entries[key] = entry
                self.values[key] = defaults[prefix][ch - 1]
                entry.bind(
                    "<KeyRelease>",
                    lambda _e, k=key, widget=entry: self._update_value(k, widget.get()),
                )

        ctk.CTkButton(frame, text="압축 실행", command=lambda: handle_compress(self.values)).grid(
            row=13, column=0, columnspan=3, sticky="ew", padx=4, pady=6
        )

    def _build_decompress_tab(self, frame: ctk.CTkFrame) -> None:
        frame.configure(fg_color=THEME_PANEL)
        frame.grid_columnconfigure(1, weight=1)

        self.entries["decompress_input"] = self._add_labeled_entry(frame, 0, "압축 JSON", "decompress_input", width=40)
        ctk.CTkButton(
            frame,
            text="찾기",
            command=lambda: self._browse_file("decompress_input", [("JSON", "*.json"), ("모든 파일", "*.*")]),
        ).grid(row=0, column=2, padx=4, pady=2)

        self.entries["decompress_output"] = self._add_labeled_entry(frame, 1, "출력 CSV", "decompress_output", width=40)
        ctk.CTkButton(
            frame,
            text="저장 위치",
            command=lambda: self._save_file("decompress_output", ".csv", [("CSV", "*.csv"), ("모든 파일", "*.*")]),
        ).grid(row=1, column=2, padx=4, pady=2)

        self.entries["time_column"] = self._add_labeled_entry(frame, 2, "시간 컬럼 이름", "time_column", default="time")

        ctk.CTkButton(frame, text="복원 및 미리보기", command=self._handle_decompress).grid(
            row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=6
        )
        preview_frame = ctk.CTkFrame(frame, fg_color="#0b1220", corner_radius=12)
        preview_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=4, pady=(10, 6))
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)

        tabs = ctk.CTkTabview(
            preview_frame,
            fg_color=THEME_PANEL,
            segmented_button_fg_color="#0b1220",
            segmented_button_selected_color=THEME_ACCENT,
            segmented_button_unselected_color="#1f2937",
            segmented_button_border_color="#374151",
        )
        tabs.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        template_tab = tabs.add("대표 파형")
        template_tab.grid_columnconfigure(0, weight=1)
        template_tab.grid_rowconfigure(0, weight=1)
        self.template_canvas = ctk.CTkFrame(template_tab, fg_color=THEME_PANEL)
        self.template_canvas.grid(row=0, column=0, sticky="nsew")

        abnormal_tab = tabs.add("이상 구간")
        abnormal_tab.grid_columnconfigure(0, weight=1)
        abnormal_tab.grid_rowconfigure(2, weight=1)
        selector = ctk.CTkFrame(abnormal_tab, fg_color="#0b1220")
        selector.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        selector.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(selector, text="이상 구간 선택").grid(row=0, column=0, sticky="w", padx=4)
        self.segment_combo = ctk.CTkComboBox(
            selector,
            variable=self.segment_var,
            state="readonly",
            values=["전체"],
            width=160,
            command=lambda _val=None: self._update_abnormal_plot(),
        )
        self.segment_combo.grid(row=0, column=1, sticky="w", padx=4)

        self.abnormal_canvas = ctk.CTkFrame(abnormal_tab, fg_color=THEME_PANEL)
        self.abnormal_canvas.grid(row=2, column=0, sticky="nsew", padx=0, pady=(6, 0))

        frame.grid_rowconfigure(4, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)

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

    def _on_close(self) -> None:
        plt.close("all")
        self.root.quit()
        self.root.destroy()


# ---------- Event loop ----------
def main(args: Iterable[str] | None = None) -> None:  # noqa: ARG001
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()
    chosen_font = _configure_korean_fonts()
    root.title(f"Waveform Toolkit ({chosen_font})")
    app = WaveformApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
