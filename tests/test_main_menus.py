"""Smoke tests for MainWindow menu wiring (no more empty shells)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QMenu

from openfcd.gui import tokens
from openfcd.gui.mainwindow import MainWindow
from openfcd.gui.preferences import UserPrefs
from openfcd.gui.widgets.title_bar import TitleBarWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def main_window(qapp):
    w = MainWindow()
    yield w


@pytest.fixture(autouse=True)
def _restore_prefs(main_window):
    """Ensure each test starts with a fresh in-memory prefs backend."""
    saved = main_window._prefs
    main_window._prefs = UserPrefs(backend={})
    yield
    main_window._prefs = saved


def _build(menu_name: str, w: MainWindow) -> QMenu:
    menu = QMenu(w)
    builders = {
        "File": w._build_file_menu,
        "Edit": w._build_edit_menu,
        "Run": w._build_run_menu,
        "Scenes": w._build_scenes_menu,
        "Tools": w._build_tools_menu,
        "View": w._build_view_menu,
        "Help": w._build_help_menu,
    }
    builders[menu_name](menu)
    return menu


@pytest.mark.parametrize("name", ["File", "Edit", "Run", "Scenes", "Tools", "View", "Help"])
def test_each_menu_has_at_least_one_real_action(main_window, name) -> None:
    menu = _build(name, main_window)
    actions = [a for a in menu.actions() if not a.isSeparator()]
    assert actions, f"{name} menu produced no actions"
    # Make sure none of them is the legacy "Not applicable" placeholder.
    for act in actions:
        assert "Not applicable" not in act.text()


def test_recent_projects_submenu_lists_prefs_entries(main_window) -> None:
    main_window._prefs = UserPrefs(backend={
        "recent_projects": ["/x/foo.ofcd", "/y/bar.ofcd"],
    })
    menu = _build("File", main_window)
    recent = next(a for a in menu.actions() if a.text() == "Recent Projects")
    sub = recent.menu()
    labels = [a.text() for a in sub.actions() if not a.isSeparator()]
    # Two recents + a Clear Recent action.
    assert any("/x/foo.ofcd" in lbl for lbl in labels)
    assert any("/y/bar.ofcd" in lbl for lbl in labels)
    assert any("Clear Recent" in lbl for lbl in labels)


def test_recent_projects_submenu_empty_state(main_window) -> None:
    main_window._prefs = UserPrefs(backend={})
    menu = _build("File", main_window)
    recent = next(a for a in menu.actions() if a.text() == "Recent Projects")
    sub = recent.menu()
    none_actions = [a for a in sub.actions() if a.text() == "(none)"]
    assert none_actions and not none_actions[0].isEnabled()


def test_recent_projects_marks_missing_paths(main_window, tmp_path) -> None:
    real = tmp_path / "real.ofcd"
    real.mkdir()
    main_window._prefs = UserPrefs(backend={
        "recent_projects": [str(real), "/does/not/exist.ofcd"],
    })
    menu = _build("File", main_window)
    sub = next(a for a in menu.actions() if a.text() == "Recent Projects").menu()
    labels = [a.text() for a in sub.actions() if not a.isSeparator()]
    assert any(str(real) == lbl for lbl in labels)
    assert any("(missing)" in lbl for lbl in labels)


def test_help_about_triggers_without_error(main_window, monkeypatch) -> None:
    """Make sure About handler runs end-to-end without raising."""
    from PyQt6.QtWidgets import QMessageBox

    called = {}

    def fake_about(parent, title, body):
        called["title"] = title
        called["body"] = body

    monkeypatch.setattr(QMessageBox, "about", staticmethod(fake_about))
    main_window._on_about()
    assert called["title"] == "About OpenFCD"
    assert "OpenFCD" in called["body"]


def test_title_bar_does_not_draw_fake_window_controls(qapp) -> None:
    bar = TitleBarWidget()
    assert bar._right_controls == []


def test_title_bar_menu_labels_are_transparent(qapp) -> None:
    bar = TitleBarWidget()
    assert bar._menus
    for label in bar._menus:
        style = label.styleSheet()
        assert "background:transparent" in style
        assert "border:none" in style


def test_main_window_global_style_covers_dark_menu_contrast(main_window) -> None:
    main_window._apply_theme()
    style = main_window.styleSheet()
    assert "QMenu::item" in style
    assert "background: transparent" in style
    assert "selection-color" in style


def test_dark_mode_applies_readable_qt_palette(main_window, qapp) -> None:
    try:
        tokens.set_dark_mode(True)
        main_window._apply_theme()
        palette = qapp.palette()
        assert palette.windowText().color().name().upper() == tokens.TEXT_PRIMARY
        assert palette.text().color().name().upper() == tokens.TEXT_PRIMARY
        assert palette.base().color().name().upper() == tokens.BG_TERTIARY
    finally:
        tokens.set_dark_mode(False)
        main_window._apply_theme()
