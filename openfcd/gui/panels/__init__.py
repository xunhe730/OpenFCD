"""Panels __init__ — re-export panel widgets."""

from openfcd.gui.panels.sim_tree import SimTree
from openfcd.gui.panels.scene_tabs import SceneTabs
from openfcd.gui.panels.properties import PropertiesPanel

__all__ = ["SimTree", "SceneTabs", "PropertiesPanel"]
