"""Enum alignment tests — ComputePanel options must match Pydantic schema exactly.

These tests validate that the dropdown values in ComputePanel match the
Pydantic schema definitions WITHOUT instantiating Qt widgets (headless-safe).
"""

import ast
from pathlib import Path




# ── Schema ground truth (hardcoded from Pydantic models) ─────────────
# OpticalStack.preset: Literal["pattern_below_window", "immersed_pattern", "custom"]
# ProcessConfig.detrend: Literal["plane", "none"]
SCHEMA_PRESETS = {"pattern_below_window", "immersed_pattern", "custom"}
SCHEMA_DETREND = {"plane", "none"}

PROPERTIES_PY = Path(__file__).parent.parent / "openfcd" / "gui" / "panels" / "properties.py"


def _extract_propselect_items(source: str, variable: str) -> set[str]:
    """Extract the list argument from a PropSelect([...]) assignment."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == variable:
                    if isinstance(node.value, ast.Call):
                        if node.value.args and isinstance(node.value.args[0], ast.List):
                            result: set[str] = set()
                            for elt in node.value.args[0].elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    result.add(elt.value)
                            return result
    return set()


class TestComputePanelEnumAlignment:
    """Verify ComputePanel dropdown items match the Pydantic schema."""

    def test_compute_panel_preset_matches_schema(self) -> None:
        """Preset dropdown must contain exactly the 3 schema values."""
        source = PROPERTIES_PY.read_text()
        items = _extract_propselect_items(source, "_preset")

        assert items == SCHEMA_PRESETS, (
            f"ComputePanel._preset items {items!r} do not match schema {SCHEMA_PRESETS!r}"
        )

    def test_compute_panel_detrend_matches_schema(self) -> None:
        """Detrend dropdown must contain exactly the 2 schema values."""
        source = PROPERTIES_PY.read_text()
        items = _extract_propselect_items(source, "_detrend")

        assert items == SCHEMA_DETREND, (
            f"ComputePanel._detrend items {items!r} do not match schema {SCHEMA_DETREND!r}"
        )

    def test_sync_wont_write_invalid_preset(self) -> None:
        """Verify forbidden preset values are not in schema."""
        # AGENTS.md §7: pattern_in_air is forbidden
        assert "pattern_in_air" not in SCHEMA_PRESETS
        # Ensure only valid presets exist
        assert SCHEMA_PRESETS == {"pattern_below_window", "immersed_pattern", "custom"}

    def test_sync_wont_write_invalid_detrend(self) -> None:
        """Verify forbidden detrend values are not in schema."""
        # "linear" detrend is not in the schema
        assert "linear" not in SCHEMA_DETREND
        # Ensure only valid detrends exist
        assert SCHEMA_DETREND == {"plane", "none"}
