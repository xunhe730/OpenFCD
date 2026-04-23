#!/usr/bin/env python3
"""Icon Visual QA script for OpenFCD GUI.

Tests all icon replacements across light and dark themes, captures screenshots,
and generates an evidence report.
"""
from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QPixmap

from openfcd.gui import tokens
from openfcd.gui.widgets.toolbar import Toolbar
from openfcd.gui.panels.properties import ImagePropertiesPanel, ComputePanel
from openfcd.gui.panels.sim_tree import SimTree
from openfcd.gui.dialogs.new_project_wizard import NewProjectWizard
from openfcd.gui.scenes.annotation import _ToolRail


EVIDENCE_DIR = Path(__file__).resolve().parent.parent / ".sisyphus" / "evidence"


def grab_widget(widget: QWidget, name: str, theme: str) -> Path:
    """Capture a screenshot of a widget and return the saved path."""
    widget.show()
    widget.resize(widget.minimumSizeHint().expandedTo(QSize(400, 200)))
    QApplication.processEvents()

    pixmap = widget.grab()
    if pixmap.isNull():
        rect = widget.rect()
        pixmap = QPixmap(rect.size())
        widget.render(pixmap)

    out_path = EVIDENCE_DIR / f"task-9-{name}-{theme}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(out_path))
    widget.hide()
    return out_path


def check_button_icon(btn, name: str) -> tuple[bool, str]:
    """Check if a button has a non-null icon."""
    icon = btn.icon()
    if icon.isNull():
        return False, f"MISSING: {name}"
    return True, f"OK: {name}"


def check_toolbar(tb: Toolbar) -> tuple[int, int, list[str]]:
    """Check all toolbar buttons have non-null icons."""
    checks = [
        ("New", tb._btn_new_project),
        ("Open", tb._btn_open),
        ("Save", tb._btn_save),
        ("Run", tb._btn_run),
        ("Cancel", tb._btn_cancel),
        ("ThemeToggle", tb._theme_toggle),
    ]
    non_null = 0
    missing = []
    for name, btn in checks:
        ok, msg = check_button_icon(btn, name)
        if ok:
            non_null += 1
        else:
            missing.append(msg)
    return non_null, len(checks), missing


def check_image_properties(props: ImagePropertiesPanel) -> tuple[int, int, list[str]]:
    """Check ImagePropertiesPanel action button icons (5 buttons)."""
    checks = [
        ("DrawROI", props._btn_roi),
        ("DrawMask", props._btn_mask),
        ("Clear", props._btn_clear),
        ("SetRef", props._btn_ref),
        ("ComputeFrame", props._btn_compute),
    ]
    non_null = 0
    missing = []
    for name, btn in checks:
        ok, msg = check_button_icon(btn, name)
        if ok:
            non_null += 1
        else:
            missing.append(msg)
    return non_null, len(checks), missing


def check_compute_panel(panel: ComputePanel) -> tuple[int, int, list[str]]:
    """Check ComputePanel action button icons."""
    checks = [("RunAllFrames", panel._btn_run)]
    non_null = 0
    missing = []
    for name, btn in checks:
        ok, msg = check_button_icon(btn, name)
        if ok:
            non_null += 1
        else:
            missing.append(msg)
    return non_null, len(checks), missing


def check_sim_tree(tree: SimTree) -> tuple[int, int, list[str]]:
    """Check SimTree item icons."""
    non_null = 0
    total = 0
    missing = []

    def walk(item, depth=0):
        nonlocal non_null, total, missing
        total += 1
        icon = item.icon(0)
        name = item.text(0)[:30]
        if icon.isNull():
            missing.append(f"  [{depth}] {name}")
        else:
            non_null += 1
        for i in range(item.childCount()):
            walk(item.child(i), depth + 1)

    root = tree.invisibleRootItem()
    for i in range(root.childCount()):
        walk(root.child(i))

    return non_null, total, missing


def check_wizard(wizard: NewProjectWizard) -> tuple[int, int, list[str]]:
    """Check wizard navigation button icons."""
    checks = [
        ("Back", wizard._back_btn),
        ("Next", wizard._next_btn),
    ]
    non_null = 0
    missing = []
    for name, btn in checks:
        ok, msg = check_button_icon(btn, name)
        if ok:
            non_null += 1
        else:
            missing.append(msg)
    return non_null, len(checks), missing


def check_tool_rail(rail: _ToolRail) -> tuple[int, int, list[str]]:
    """Check _ToolRail button icons."""
    non_null = 0
    total = 0
    missing = []
    for tool, btn in rail._buttons.items():
        total += 1
        icon = btn.icon()
        if icon.isNull():
            missing.append(tool.name)
        else:
            non_null += 1
    return non_null, total, missing


def fmt_check(name: str, non_null: int, total: int, missing: list[str]) -> str:
    """Format a single check result."""
    status = "PASS" if non_null == total else "FAIL"
    marker = "✓" if status == "PASS" else "✗"
    lines = [f"  {name}: {non_null}/{total} non-null {marker}"]
    for m in missing:
        lines.append(f"    {m}")
    return "\n".join(lines)


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    app.setApplicationName("OpenFCD-QA")

    # ── Create all widgets ONCE ──────────────────────────────────────
    win = QMainWindow()
    win.setWindowTitle("OpenFCD Icon QA")
    win.resize(1000, 800)

    container = QWidget()
    layout = QVBoxLayout(container)

    # Toolbar
    tb = Toolbar(running=False)
    layout.addWidget(tb)

    # ImagePropertiesPanel
    props = ImagePropertiesPanel()
    props.set_frame_info("test_000.jpg", "1024x768", 0, 10)
    layout.addWidget(props)

    # ComputePanel
    compute = ComputePanel()
    layout.addWidget(compute)

    # SimTree with test data
    tree = SimTree()
    tree.populate_from_session(
        project_name="test_experiment",
        frames=[Path("img_000.jpg"), Path("img_001.jpg"), Path("img_002.jpg")],
        ref_index=1,
        roi_info="x=0 y=0 w=100 h=100",
        mask_info="4 points",
        runs=[{"id": "run_001", "status": "done", "detail": "3 frames"}],
        scenes=[{"id": "scene_0", "label": "eta_heatmap"}],
    )
    layout.addWidget(tree)

    # ToolRail
    rail = _ToolRail()
    layout.addWidget(rail)

    win.setCentralWidget(container)
    win.show()
    QApplication.processEvents()

    # ── LIGHT MODE ───────────────────────────────────────────────────
    tokens.set_dark_mode(False)
    QApplication.processEvents()

    light_results = {
        "toolbar": check_toolbar(tb),
        "properties": check_image_properties(props),
        "compute": check_compute_panel(compute),
        "simtree": check_sim_tree(tree),
        "toolrail": check_tool_rail(rail),
    }

    # Screenshots - individual widgets
    grab_widget(tb, "toolbar", "light")
    grab_widget(props, "properties", "light")
    grab_widget(compute, "compute", "light")
    grab_widget(tree, "simtree", "light")
    grab_widget(rail, "toolrail", "light")
    grab_widget(container, "full-window", "light")

    print("=== LIGHT MODE RESULTS ===")
    for name, (nn, tot, miss) in light_results.items():
        print(f"  {name}: {nn}/{tot} {'PASS' if nn == tot else 'FAIL'}")
        for m in miss:
            print(f"    {m}")

    # ── DARK MODE ────────────────────────────────────────────────────
    tokens.set_dark_mode(True)
    QApplication.processEvents()
    # Extra processing for icon re-render
    for _ in range(5):
        QApplication.processEvents()

    dark_results = {
        "toolbar": check_toolbar(tb),
        "properties": check_image_properties(props),
        "compute": check_compute_panel(compute),
        "simtree": check_sim_tree(tree),
        "toolrail": check_tool_rail(rail),
    }

    # Screenshots
    grab_widget(tb, "toolbar", "dark")
    grab_widget(props, "properties", "dark")
    grab_widget(compute, "compute", "dark")
    grab_widget(tree, "simtree", "dark")
    grab_widget(rail, "toolrail", "dark")
    grab_widget(container, "full-window", "dark")

    print("\n=== DARK MODE RESULTS ===")
    for name, (nn, tot, miss) in dark_results.items():
        print(f"  {name}: {nn}/{tot} {'PASS' if nn == tot else 'FAIL'}")
        for m in miss:
            print(f"    {m}")

    # ── Also test NewProjectWizard separately (it's a QDialog) ──────
    wizard = NewProjectWizard()
    wizard.show()
    QApplication.processEvents()

    # Light mode wizard check
    tokens.set_dark_mode(False)
    QApplication.processEvents()
    light_wiz = check_wizard(wizard)
    grab_widget(wizard, "wizard", "light")

    # Dark mode wizard check
    tokens.set_dark_mode(True)
    QApplication.processEvents()
    for _ in range(5):
        QApplication.processEvents()
    dark_wiz = check_wizard(wizard)
    grab_widget(wizard, "wizard", "dark")
    wizard.close()

    light_results["wizard"] = light_wiz
    dark_results["wizard"] = dark_wiz

    # ── Generate report ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    screenshot_count = len(list(EVIDENCE_DIR.glob("task-9-*.png")))

    def all_pass(results: dict) -> bool:
        return all(nn == tot for nn, tot, _ in results.values())

    overall = "PASS" if (all_pass(light_results) and all_pass(dark_results)) else "FAIL"

    report_lines = [
        f"Task 9: Icon Visual QA Report",
        f"Date: {timestamp}",
        "",
        "LIGHT MODE:",
    ]
    for name, (nn, tot, miss) in light_results.items():
        report_lines.append(fmt_check(name, nn, tot, miss))

    report_lines.append("")
    report_lines.append("DARK MODE (after theme toggle):")
    for name, (nn, tot, miss) in dark_results.items():
        report_lines.append(fmt_check(name, nn, tot, miss))

    report_lines.extend([
        "",
        f"Screenshots captured: {screenshot_count} files",
        "",
        f"OVERALL: {overall}",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = EVIDENCE_DIR / "task-9-qa-report.txt"
    report_path.write_text(report)

    print(f"\nReport saved: {report_path}")
    print(f"Screenshots: {screenshot_count} files in {EVIDENCE_DIR}")
    print(f"\n{report}")

    win.close()
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
