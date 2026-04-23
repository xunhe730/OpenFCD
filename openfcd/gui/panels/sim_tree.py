"""Simulation tree panel — QTreeWidget showing project hierarchy.

Data-driven: populated from SessionController state, not hardcoded.
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem, QMenu
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QAction, QIcon

from openfcd.gui import tokens
from openfcd.gui.icons import (
    get_icon, ICON_PHOTO_LIBRARY, ICON_PHOTO, ICON_STAR,
    ICON_SPEED, ICON_CHECK_CIRCLE, ICON_ERROR, ICON_SCHEDULE,
    ICON_PIE_CHART,
)


class NodeType(Enum):
    """Tree node type, stored in UserRole for routing."""
    ROOT = auto()
    IMAGES = auto()        # Images parent node
    IMAGE_FRAME = auto()   # Individual image file
    ANNOTATIONS = auto()   # Annotations parent node
    ANN_ROI = auto()       # ROI child
    ANN_MASK = auto()      # Mask child
    RUNS = auto()          # Runs parent node
    RUN_ITEM = auto()      # Individual run
    SCENES = auto()        # Scenes parent node
    SCENE_ITEM = auto()    # Individual scene/figure


# Role constants
ROLE_NODE_TYPE = Qt.ItemDataRole.UserRole
ROLE_DATA = Qt.ItemDataRole.UserRole + 1    # path, run_id, etc.
ROLE_MUTED = Qt.ItemDataRole.UserRole + 2


class SimTree(QTreeWidget):
    """Project simulation tree with folder hierarchy.

    Signals:
        node_selected(str): node type key emitted on selection.
        set_reference_requested(int): frame index for ref designation.
        compute_frame_requested(int): frame index for single-frame compute.
    """

    node_selected = pyqtSignal(str)
    set_reference_requested = pyqtSignal(int)
    compute_frame_requested = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frames: list[Path] = []
        self._ref_index: int = -1
        self._setup_ui()
        tokens.on_theme_changed(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(self._style())
        self._update_item_colors(self.invisibleRootItem())

    def _update_item_colors(self, parent_item: QTreeWidgetItem) -> None:
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            is_muted = child.data(0, ROLE_MUTED)
            if is_muted:
                child.setForeground(0, QColor(tokens.TEXT_MUTED))
            self._update_item_colors(child)

    def _setup_ui(self) -> None:
        self.setFixedWidth(280)
        self.setMinimumWidth(280)
        self.setHeaderHidden(True)
        self.setIndentation(14)
        self.setAnimated(True)
        self.setUniformRowHeights(True)
        self.setSelectionBehavior(self.SelectionBehavior.SelectRows)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        self.currentItemChanged.connect(self._on_selection_changed)

    def _style(self) -> str:
        t = tokens
        return f"""
            QTreeWidget {{
                background: {t.BG_SECONDARY};
                border: none;
                color: {t.TEXT_PRIMARY};
                font-family: {t.FONT_UI};
                font-size: 12px;
                outline: none;
            }}
            QTreeWidget::item {{
                height: 24px;
                padding: 0 4px;
                border-left: 2px solid transparent;
            }}
            QTreeWidget::item:selected {{
                background: {t.SELECTION_BG};
                color: {t.TEXT_PRIMARY};
                border-left: 2px solid {t.ACCENT_CLAY};
            }}
            QTreeWidget::item:hover:!selected {{
                background: {t.BORDER_SUBTLE};
            }}
        """

    # ── Population ──────────────────────────────────────────────────

    def populate(self, project_name: str = "my_experiment") -> None:
        """Populate tree with empty project skeleton."""
        self.clear()
        self._frames = []
        self._ref_index = -1
        self._build_skeleton(project_name)

    def populate_from_session(
        self,
        project_name: str,
        frames: list[Path],
        ref_index: int = -1,
        roi_info: str = "",
        mask_info: str = "",
        runs: list[dict] | None = None,
        scenes: list[dict] | None = None,
    ) -> None:
        """Data-driven tree population from session state."""
        self.clear()
        self._frames = frames
        self._ref_index = ref_index

        root = self._make_item(project_name, NodeType.ROOT)
        root.setExpanded(True)
        self.addTopLevelItem(root)

        # ── Images ──
        img_node = self._make_item(
            f"Images ({len(frames)})", NodeType.IMAGES,
            icon=get_icon(ICON_PHOTO_LIBRARY),
        )
        img_node.setExpanded(True)
        root.addChild(img_node)

        for i, fpath in enumerate(frames):
            name = fpath.name
            if i == ref_index:
                item = self._make_item(
                    f"{name}  [ref]", NodeType.IMAGE_FRAME, data=i,
                    icon=get_icon(ICON_STAR),
                )
            else:
                item = self._make_item(
                    name, NodeType.IMAGE_FRAME, data=i,
                    icon=get_icon(ICON_PHOTO),
                )
            if i == ref_index:
                item.setForeground(0, QColor(tokens.ACCENT_CLAY))
            img_node.addChild(item)

        # ── Runs ──
        run_list = runs or []
        runs_node = self._make_item(
            f"Runs ({len(run_list)})", NodeType.RUNS,
            icon=get_icon(ICON_SPEED),
        )
        runs_node.setExpanded(True)
        root.addChild(runs_node)

        for run in run_list:
            rid = run.get("id", "unknown")
            status = run.get("status", "")
            icon_map = {"done": ICON_CHECK_CIRCLE, "failed": ICON_ERROR, "pending": ICON_SCHEDULE}
            icon_name = icon_map.get(status, ICON_SCHEDULE)
            item = self._make_item(rid, NodeType.RUN_ITEM, data=rid, icon=get_icon(icon_name))
            detail = run.get("detail", "")
            if detail:
                sub = self._make_item(detail, NodeType.RUN_ITEM, muted=True, data=rid)
                item.addChild(sub)
            runs_node.addChild(item)

        # ── Scenes ──
        scene_list = scenes or []
        scenes_node = self._make_item(
            f"Scenes ({len(scene_list)})", NodeType.SCENES,
            icon=get_icon(ICON_PIE_CHART),
        )
        scenes_node.setExpanded(True)
        root.addChild(scenes_node)

        for scn in scene_list:
            sid = scn.get("id", "")
            label = scn.get("label", sid)
            item = self._make_item(label, NodeType.SCENE_ITEM, data=sid)
            scenes_node.addChild(item)
            
        self.expandAll()

    def _build_skeleton(self, project_name: str) -> None:
        """Build an empty project skeleton."""
        root = self._make_item(project_name, NodeType.ROOT)
        root.setExpanded(True)
        self.addTopLevelItem(root)

        icons_map = {
            NodeType.IMAGES: get_icon(ICON_PHOTO_LIBRARY),
            NodeType.RUNS: get_icon(ICON_SPEED),
            NodeType.SCENES: get_icon(ICON_PIE_CHART),
        }
        for label, ntype in [
            ("Images (0)", NodeType.IMAGES),
            ("Runs (0)", NodeType.RUNS),
            ("Scenes (0)", NodeType.SCENES),
        ]:
            node = self._make_item(label, ntype, icon=icons_map[ntype])
            
        self.expandAll()

    def _make_item(
        self,
        text: str,
        node_type: NodeType,
        *,
        data: object = None,
        muted: bool = False,
        icon: QIcon | None = None,
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([text])
        if icon is not None:
            item.setIcon(0, icon)
            item.setSizeHint(0, QSize(20, 20))
        item.setData(0, ROLE_NODE_TYPE, node_type)
        item.setData(0, ROLE_DATA, data)
        item.setData(0, ROLE_MUTED, muted)
        if muted:
            item.setForeground(0, QColor(tokens.TEXT_MUTED))
        return item

    # ── Context menu ────────────────────────────────────────────────

    def _on_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is None:
            return

        node_type = item.data(0, ROLE_NODE_TYPE)
        if node_type is None:
            return

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {tokens.BG_PRIMARY};
                color: {tokens.TEXT_PRIMARY};
                border: 1px solid {tokens.BORDER_SUBTLE};
                font-family: {tokens.FONT_UI};
                font-size: 12px;
            }}
            QMenu::item:selected {{
                background-color: {tokens.ACCENT_CLAY};
                color: white;
            }}
        """)

        if node_type == NodeType.IMAGE_FRAME:
            frame_idx = item.data(0, ROLE_DATA)
            if frame_idx is not None:
                act_ref = menu.addAction("Set as Reference")
                act_ref.triggered.connect(lambda: self.set_reference_requested.emit(frame_idx))
                act_compute = menu.addAction("Compute This Frame")
                act_compute.triggered.connect(lambda: self.compute_frame_requested.emit(frame_idx))

        elif node_type == NodeType.RUNS:
            act_run_all = menu.addAction("Run All Frames")
            # Will be connected by MainWindow
            act_run_all.setData("run_all")

        elif node_type == NodeType.SCENES:
            for label in ["New η Map", "New λ/h Profile", "New RMS Map"]:
                menu.addAction(label)

        elif node_type == NodeType.SCENE_ITEM:
            menu.addAction("Export as PNG")
            menu.addAction("Export as PDF")
            menu.addAction("Export as CSV")
            menu.addSeparator()
            menu.addAction("Delete")

        else:
            return  # No menu for other node types

        menu.exec(self.viewport().mapToGlobal(pos))

    # ── Selection ───────────────────────────────────────────────────

    def _on_selection_changed(
        self, current: QTreeWidgetItem | None, _prev: QTreeWidgetItem | None
    ) -> None:
        if current is None:
            return
        node_type = current.data(0, ROLE_NODE_TYPE)
        if node_type is None:
            key = current.text(0).lower()
        else:
            key = node_type.name.lower()
        self.node_selected.emit(str(key))

    # ── Ref update ──────────────────────────────────────────────────

    def update_ref_mark(self, new_ref_index: int) -> None:
        """Update the visual ref marker on images."""
        self._ref_index = new_ref_index
        # Find Images node and refresh labels
        root = self.topLevelItem(0)
        if root is None:
            return
        for i in range(root.childCount()):
            section = root.child(i)
            if section.data(0, ROLE_NODE_TYPE) == NodeType.IMAGES:
                for j in range(section.childCount()):
                    frame_item = section.child(j)
                    idx = frame_item.data(0, ROLE_DATA)
                    if idx is not None and idx < len(self._frames):
                        name = self._frames[idx].name
                        if idx == new_ref_index:
                            frame_item.setText(0, f"{name}  [ref]")
                            frame_item.setIcon(0, get_icon(ICON_STAR))
                            frame_item.setForeground(0, QColor(tokens.ACCENT_CLAY))
                        else:
                            frame_item.setText(0, name)
                            frame_item.setIcon(0, get_icon(ICON_PHOTO))
                            frame_item.setForeground(0, QColor(tokens.TEXT_PRIMARY))
                break
