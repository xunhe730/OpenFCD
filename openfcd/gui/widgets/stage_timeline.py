"""StageTimeline — bottom progress bar showing pipeline stages."""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt

from openfcd.gui import tokens


class StageTimeline(QWidget):
    """Bottom bar showing Preprocess → Compute → Postprocess progress."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(52)  # chrome.jsx has height 52
        self._states: dict[str, dict] = {}
        self._build_ui()
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()
        self.reset()

    # ── build ──────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._stage_widgets: dict[str, dict] = {}
        self._separators = []

        stages = ("Preprocess", "Compute", "Postprocess")
        for i, name in enumerate(stages):
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.NoFrame)
            vlayout = QVBoxLayout(frame)
            vlayout.setContentsMargins(16, 6, 16, 6)
            vlayout.setSpacing(6)

            # Top row: icon + name + progress text
            row = QHBoxLayout()
            row.setSpacing(8)

            icon = QLabel("○")
            icon.setFixedWidth(14)
            icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

            label = QLabel(name)
            
            prog_text = QLabel("")
            prog_text.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            row.addWidget(icon)
            row.addWidget(label)
            row.addStretch()
            row.addWidget(prog_text)

            vlayout.addLayout(row)

            # Progress bar track
            from PyQt6.QtWidgets import QProgressBar
            bar = QProgressBar()
            bar.setTextVisible(False)
            bar.setFixedHeight(3)
            bar.setMinimum(0)
            bar.setMaximum(100)
            bar.setValue(0)
            vlayout.addWidget(bar)
            
            layout.addWidget(frame, 2 if name == "Compute" else 1)

            if i < len(stages) - 1:
                sep = QFrame()
                sep.setFixedWidth(1)
                self._separators.append(sep)
                layout.addWidget(sep)

            self._stage_widgets[name] = {
                "frame": frame,
                "icon": icon,
                "label": label,
                "prog_text": prog_text,
                "bar": bar
            }
            self._states[name] = {"status": "idle", "text": "", "progress": 0}

    # ── theme ──────────────────────────────────────────────────────
    def _apply_theme(self) -> None:
        self.setStyleSheet(f"""
            StageTimeline {{
                background: {tokens.BG_SECONDARY};
                border-top: 1px solid {tokens.BORDER_SUBTLE};
            }}
        """)
        for sep in self._separators:
            sep.setStyleSheet(f"background: {tokens.BORDER_SUBTLE}; border: none;")
            
        # Re-apply states to refresh colors
        for name, st in self._states.items():
            self.set_stage_state(name, st["status"], st["text"], st["progress"])

    # ── public API ─────────────────────────────────────────────────
    def set_stage_state(self, stage: str, status: str, text: str = "", progress: float = 0) -> None:
        """Update a stage's visual state.
        status: "idle" | "running" | "done" | "failed"
        """
        self._states[stage] = {"status": status, "text": text, "progress": progress}
        
        info = self._stage_widgets.get(stage)
        if not info:
            return
            
        frame = info["frame"]
        icon = info["icon"]
        label = info["label"]
        prog_text = info["prog_text"]
        bar = info["bar"]

        if status == "done":
            icon.setText("✓")
            icon.setStyleSheet(f"color: {tokens.SUCCESS}; font-size: 12px; font-weight: 700;")
            label.setStyleSheet(f"font-size: 12px; color: {tokens.TEXT_PRIMARY}; font-weight: 500;")
            frame.setStyleSheet("background: transparent;")
            bar_color = tokens.SUCCESS
            val = 100
        elif status == "running":
            icon.setText("●")
            icon.setStyleSheet(f"color: {tokens.ACCENT_CLAY}; font-size: 10px;")
            label.setStyleSheet(f"font-size: 12px; color: {tokens.TEXT_PRIMARY}; font-weight: 500;")
            frame.setStyleSheet(f"QFrame {{ background: {tokens.ACCENT_CLAY_BG}; }}")
            bar_color = tokens.ACCENT_CLAY
            val = int(progress)
        elif status == "failed":
            icon.setText("✕")
            icon.setStyleSheet(f"color: {tokens.ERROR}; font-size: 12px; font-weight: 700;")
            label.setStyleSheet(f"font-size: 12px; color: {tokens.TEXT_PRIMARY};")
            frame.setStyleSheet("background: transparent;")
            bar_color = tokens.ERROR
            val = max(int(progress), 34) # Show some progress on failure
        else:  # idle
            icon.setText("○")
            icon.setStyleSheet(f"color: {tokens.TEXT_MUTED}; font-size: 12px;")
            label.setStyleSheet(f"font-size: 12px; color: {tokens.TEXT_MUTED};")
            frame.setStyleSheet("background: transparent;")
            bar_color = "transparent"
            val = 0

        prog_text.setText(text)
        prog_text.setStyleSheet(f"font-size: 11px; color: {tokens.TEXT_MUTED}; font-family: {tokens.FONT_MONO};")

        bar.setStyleSheet(f"""
            QProgressBar {{
                background: {tokens.BORDER_SUBTLE};
                border: none;
                border-radius: 1px;
            }}
            QProgressBar::chunk {{
                background: {bar_color};
                border-radius: 1px;
            }}
        """)
        bar.setValue(val)

    def reset(self) -> None:
        """Reset all stages to idle."""
        for name in ("Preprocess", "Compute", "Postprocess"):
            self.set_stage_state(name, "idle")
