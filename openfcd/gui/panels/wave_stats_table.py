"""Wave-stats segment table panel (v2).

A table view of all wave segments in the active annotation.  Supports:

  * Multiple segments (any N)
  * Per-row visibility checkbox
  * Per-row color picker (QColorDialog)
  * Editable label, s_lo_mm, s_hi_mm cells
  * Per-row delete button
  * Read-only λ, k, mean height and peak count from results.h5 if available

Signals
-------
segmentClicked(int)
    Emitted with row index when a row is clicked (used by ProfileSceneView
    to highlight the 1D subplot segment overlay).
segmentEdited(int, object)
    Emitted with (idx, new_segment) after the user edits a cell.
segmentDeleted(int)
    Emitted with the row index after the user clicks delete.
visibilityToggled(int, bool)
    Emitted with (idx, new_visible) after the checkbox is toggled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# Column constants
COL_IDX = 0
COL_VISIBLE = 1
COL_COLOR = 2
COL_LABEL = 3
COL_S_LO = 4
COL_S_HI = 5
COL_LAMBDA = 6
COL_WAVENUM = 7
COL_HEIGHTS = 8
COL_N_PEAKS = 9
COL_S_GAMMA = 10
COL_S_G = 11
COL_S_CG = 12
COL_DELETE = 13
N_COLS = 14

# Legacy alias retained for any external imports — points at the same column.
COL_MEAN_HEIGHT = COL_HEIGHTS

COL_HEADERS = [
    "#",
    "Visible",
    "Color",
    "Label",
    "s_lo (mm)",
    "s_hi (mm)",
    "λ (mm)",
    "k (1/mm)",
    "Heights",
    "N peaks",
    "S_γ (μN/mm)",
    "S_g (μN/mm)",
    "S_cg (μN/mm)",
    "",
]


class WaveStatsTablePanel(QWidget):
    """Multi-segment wave statistics table."""

    segmentClicked = pyqtSignal(int)
    segmentEdited = pyqtSignal(int, object)        # (idx, WaveSegment)
    segmentDeleted = pyqtSignal(int)
    visibilityToggled = pyqtSignal(int, bool)
    heightsRequested = pyqtSignal(int)             # row idx; emitted on double-click of Heights cell
    copyPreviousFrameRequested = pyqtSignal()      # "copy previous frame" button clicked

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._annotation = None
        self._h5_path: Path | None = None
        self._batch: str = "default"
        self._live_stats = None  # FrameWaveStats | None — populated by ProfileSceneView
        self._suspend_signals = False
        # v3: per-frame wave-stats. The table renders the segments for the
        # currently active frame. None ⇒ no frame selected (renders empty).
        self._current_frame_idx: int | None = None
        self._previous_frame_has_segments: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        self._title = QLabel("Wave Segments")
        self._title.setStyleSheet("font-size: 11px; font-weight: 600;")
        header_layout.addWidget(self._title)
        header_layout.addStretch()
        self._btn_copy_prev = QPushButton("复制上帧")
        self._btn_copy_prev.setFixedHeight(22)
        self._btn_copy_prev.setEnabled(False)
        self._btn_copy_prev.clicked.connect(
            lambda: self.copyPreviousFrameRequested.emit()
        )
        header_layout.addWidget(self._btn_copy_prev)
        layout.addWidget(header)

        self._table = QTableWidget(0, N_COLS, self)
        self._table.setHorizontalHeaderLabels(COL_HEADERS)
        self._table.verticalHeader().setVisible(False)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(False)
        # Excel-like selection: arbitrary rectangular cell ranges.
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        # Read-only cells must still be navigable + part of selections; suppress
        # double-click edit-trigger so dragging selects rather than entering edit.
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        # Ctrl+C / Cmd+C → copy selected range as TSV (Excel-compatible).
        # Use an eventFilter on the table viewport rather than a QShortcut: the
        # shortcut path can be swallowed by the QTableWidget's internal handling
        # of Copy and lose all but the current cell, while a key-press filter
        # intercepts the raw event before Qt's default kicks in.
        self._table.installEventFilter(self)
        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.cellChanged.connect(self._on_cell_changed)
        self._table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        layout.addWidget(self._table)

    # ── Public API ────────────────────────────────────────────────────

    def set_annotation(self, annotation) -> None:
        """Bind annotation and rebuild table rows."""
        self._annotation = annotation
        self._rebuild_rows()

    def set_h5_path(self, path: Path | None, batch: str = "default") -> None:
        """Legacy hook retained for callers; live stats now drive the table.

        The path is stored only to keep older tests that exercise the setter
        happy.  λ / k / Heights / N peaks are populated via ``set_live_stats``.
        """
        self._h5_path = Path(path) if path is not None else None
        self._batch = batch
        self._rebuild_rows()

    def set_live_stats(self, stats) -> None:
        """Bind a freshly computed FrameWaveStats and refresh the stats columns."""
        self._live_stats = stats
        self._rebuild_rows()

    def live_stats(self):
        return self._live_stats

    def set_current_frame(self, frame_idx: int | None) -> None:
        """Bind the currently active frame index (v3 per-frame wave-stats)."""
        self._current_frame_idx = frame_idx
        self._rebuild_rows()

    def set_previous_frame_has_segments(self, has_segments: bool) -> None:
        """Enable/disable the 'copy previous frame' button."""
        self._previous_frame_has_segments = bool(has_segments)
        self._btn_copy_prev.setEnabled(self._previous_frame_has_segments)

    def eventFilter(self, source, event) -> bool:  # type: ignore[override]
        if source is self._table and event.type() == QEvent.Type.KeyPress:
            if event.matches(QKeySequence.StandardKey.Copy):
                self._copy_selection_as_tsv()
                return True
        return super().eventFilter(source, event)

    def _copy_selection_as_tsv(self) -> None:
        """Copy the currently selected rectangular cell range as TSV.

        Tab-separated columns, newline-separated rows — paste cleanly into
        Excel/Numbers/Google Sheets. Widget cells (checkbox/color/delete)
        have no QTableWidgetItem and emit an empty string. Multiple disjoint
        selections are concatenated row by row (Excel can't paste disjoint
        ranges anyway; the first contiguous range is the common case).
        """
        ranges = self._table.selectedRanges()
        if not ranges:
            return
        # Collect every selected cell as (row, col) → text, then emit row-by-row
        # in ascending order of row, then column.
        cell_text: dict[tuple[int, int], str] = {}
        for r in ranges:
            for row in range(r.topRow(), r.bottomRow() + 1):
                for col in range(r.leftColumn(), r.rightColumn() + 1):
                    item = self._table.item(row, col)
                    cell_text[(row, col)] = item.text() if item is not None else ""
        if not cell_text:
            return
        rows = sorted({r for r, _ in cell_text})
        cols = sorted({c for _, c in cell_text})
        lines = [
            "\t".join(cell_text.get((row, col), "") for col in cols) for row in rows
        ]
        QApplication.clipboard().setText("\n".join(lines))

    def select_row(self, idx: int) -> None:
        """Programmatically select a row (no signal emitted)."""
        if 0 <= idx < self._table.rowCount():
            self._table.selectRow(idx)

    # ── Internal ──────────────────────────────────────────────────────

    def _segments(self) -> list:
        if self._annotation is None or self._annotation.wave_stats is None:
            return []
        if self._current_frame_idx is None:
            return []
        return self._annotation.wave_stats.segments_for_frame(
            self._current_frame_idx
        )

    def _rebuild_rows(self) -> None:
        self._suspend_signals = True
        try:
            segments = self._segments()
            self._table.setRowCount(0)
            self._table.setRowCount(len(segments))

            stats = self._load_per_segment_stats()

            for row, seg in enumerate(segments):
                # 0: idx
                idx_item = QTableWidgetItem(str(row))
                idx_item.setFlags(idx_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row, COL_IDX, idx_item)

                # 1: visible checkbox
                cb_widget = QWidget()
                cb_layout = QHBoxLayout(cb_widget)
                cb_layout.setContentsMargins(0, 0, 0, 0)
                cb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cb = QCheckBox()
                cb.setChecked(bool(seg.visible))
                cb.toggled.connect(lambda checked, r=row: self._on_visibility_toggled(r, checked))
                cb_layout.addWidget(cb)
                self._table.setCellWidget(row, COL_VISIBLE, cb_widget)

                # 2: color button
                btn = QPushButton(seg.color)
                btn.setStyleSheet(
                    f"background:{seg.color}; color:white; font-family: monospace;"
                )
                btn.clicked.connect(lambda _checked, r=row: self._on_color_clicked(r))
                self._table.setCellWidget(row, COL_COLOR, btn)

                # 3: label (editable)
                label_item = QTableWidgetItem(seg.label or f"Segment {row + 1}")
                self._table.setItem(row, COL_LABEL, label_item)

                # 4: s_lo (editable)
                self._table.setItem(
                    row, COL_S_LO, QTableWidgetItem(f"{seg.s_lo_mm:.2f}")
                )
                # 5: s_hi (editable)
                self._table.setItem(
                    row, COL_S_HI, QTableWidgetItem(f"{seg.s_hi_mm:.2f}")
                )

                # 6-9: read-only stats from the live FrameWaveStats
                row_stats = stats.get(row, {})
                self._set_readonly(row, COL_LAMBDA, _fmt_float(row_stats.get("lambda")))
                self._set_readonly(row, COL_WAVENUM, _fmt_float(row_stats.get("k")))
                self._set_readonly(
                    row, COL_HEIGHTS, _fmt_count(row_stats.get("n_pairs"))
                )
                self._set_readonly(row, COL_N_PEAKS, _fmt_count(row_stats.get("n_peaks")))
                self._set_readonly(
                    row, COL_S_GAMMA, _fmt_stress(row_stats.get("S_gamma_N_per_m"))
                )
                self._set_readonly(
                    row, COL_S_G, _fmt_stress(row_stats.get("S_g_N_per_m"))
                )
                self._set_readonly(
                    row, COL_S_CG, _fmt_stress(row_stats.get("S_cg_N_per_m"))
                )

                # 13: delete button
                del_btn = QPushButton("✕")
                del_btn.setFixedWidth(28)
                del_btn.clicked.connect(lambda _checked, r=row: self._on_delete_clicked(r))
                self._table.setCellWidget(row, COL_DELETE, del_btn)
        finally:
            self._suspend_signals = False

    def _set_readonly(self, row: int, col: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, col, item)

    def _load_per_segment_stats(self) -> dict[int, dict]:
        """Build per-row stats dict from the live FrameWaveStats cache."""
        if self._live_stats is None:
            return {}
        out: dict[int, dict] = {}
        try:
            for seg_stats in self._live_stats.segments:
                idx = int(seg_stats.segment_idx)
                heights = np.asarray(seg_stats.peak_to_trough_heights, dtype=np.float64)
                out[idx] = {
                    "lambda": float(seg_stats.wavelength_mm),
                    "k": float(seg_stats.wavenumber_per_mm),
                    "n_peaks": int(seg_stats.n_peaks),
                    "n_pairs": int(heights.size),
                    "heights": heights,
                    "S_gamma_N_per_m": float(seg_stats.S_gamma_N_per_m),
                    "S_g_N_per_m": float(seg_stats.S_g_N_per_m),
                    "S_cg_N_per_m": float(seg_stats.S_cg_N_per_m),
                }
        except Exception:
            return {}
        return out

    def heights_for_row(self, row: int) -> np.ndarray:
        """Return the per-segment heights array for ``row`` (empty if missing)."""
        if self._live_stats is None:
            return np.empty(0, dtype=np.float64)
        for seg_stats in self._live_stats.segments:
            if int(seg_stats.segment_idx) == int(row):
                return np.asarray(seg_stats.peak_to_trough_heights, dtype=np.float64)
        return np.empty(0, dtype=np.float64)

    # ── Cell-event handlers ───────────────────────────────────────────

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if 0 <= row < self._table.rowCount():
            self.segmentClicked.emit(row)

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """Double-click on the Heights cell opens the per-segment detail dialog."""
        if col != COL_HEIGHTS:
            return
        if not (0 <= row < self._table.rowCount()):
            return
        segs = self._segments()
        if row >= len(segs):
            return
        self.heightsRequested.emit(row)
        seg = segs[row]
        heights = self.heights_for_row(row)
        try:
            from openfcd.gui.dialogs.wave_stats_detail import WaveStatsDetailDialog
        except Exception:
            return
        label = seg.label or f"Segment {row + 1}"
        dlg = WaveStatsDetailDialog(label, heights, color=seg.color, parent=self)
        dlg.exec()

    def _on_visibility_toggled(self, row: int, checked: bool) -> None:
        if self._suspend_signals:
            return
        segs = self._segments()
        if row >= len(segs):
            return
        new_seg = segs[row].model_copy(update={"visible": bool(checked)})
        self.visibilityToggled.emit(row, bool(checked))
        self.segmentEdited.emit(row, new_seg)

    def _on_color_clicked(self, row: int) -> None:
        if self._suspend_signals:
            return
        segs = self._segments()
        if row >= len(segs):
            return
        current = QColor(segs[row].color)
        new_color = QColorDialog.getColor(current, self, "Choose segment color")
        if not new_color.isValid():
            return
        hex_str = new_color.name()
        new_seg = segs[row].model_copy(update={"color": hex_str})
        self.segmentEdited.emit(row, new_seg)

    def _on_delete_clicked(self, row: int) -> None:
        if self._suspend_signals:
            return
        segs = self._segments()
        if row >= len(segs):
            return
        self.segmentDeleted.emit(row)

    def _on_cell_changed(self, row: int, col: int) -> None:
        if self._suspend_signals:
            return
        segs = self._segments()
        if row >= len(segs):
            return
        seg = segs[row]
        item = self._table.item(row, col)
        if item is None:
            return
        text = item.text().strip()
        try:
            if col == COL_LABEL:
                new_seg = seg.model_copy(update={"label": text})
            elif col == COL_S_LO:
                new_lo = float(text)
                hi = float(seg.s_hi_mm)
                if new_lo >= hi:
                    self._rebuild_rows()
                    return
                new_seg = seg.model_copy(update={"s_lo_mm": new_lo})
            elif col == COL_S_HI:
                new_hi = float(text)
                lo = float(seg.s_lo_mm)
                if new_hi <= lo:
                    self._rebuild_rows()
                    return
                new_seg = seg.model_copy(update={"s_hi_mm": new_hi})
            else:
                return
        except (ValueError, Exception):
            self._rebuild_rows()
            return
        self.segmentEdited.emit(row, new_seg)


# ── Helpers ──────────────────────────────────────────────────────────────


def _fmt_float(v) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f):
        return "—"
    return f"{f:.3f}"


def _fmt_int(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_stress(v) -> str:
    """Format radiation stress (S_γ / S_g / S_cg) for display in μN/mm.

    Internal storage is SI (N/m). The GUI rescales by 1000× so typical
    capillary-wave magnitudes show as values like 0.7 instead of 7e-4. HDF5
    and the SegmentWaveStats fields keep N/m for physics correctness.
    """
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f):
        return "—"
    return f"{f * 1000.0:.4g}"


def _fmt_count(v) -> str:
    """Format an integer count (Heights / N peaks). Empty → '—', 0 → '0'."""
    if v is None:
        return "—"
    try:
        n = int(v)
    except (TypeError, ValueError):
        return "—"
    return str(n)
