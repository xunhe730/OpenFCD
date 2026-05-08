from PyQt6.QtCore import QObject, pyqtSignal

class _ThemeSignals(QObject):
    changed = pyqtSignal()

_signals = _ThemeSignals()

def _theme_signals() -> _ThemeSignals:
    global _signals
    try:
        _signals.objectName()
    except RuntimeError:
        _signals = _ThemeSignals()
    return _signals

def on_theme_changed(slot):
    _theme_signals().changed.connect(slot)

is_dark = False

LIGHT_PALETTE = {
    "BG_PRIMARY": "#F5F1EB",
    "BG_SECONDARY": "#FAFAF7",
    "BG_TERTIARY": "#FFFFFF",
    "BG_DOCK": "#EFEAE2",
    "BORDER_SUBTLE": "#E6E0D6",
    "BORDER_STRONG": "#C9BFB0",
    "TEXT_PRIMARY": "#1A1A1A",
    "TEXT_SECONDARY": "#6B6558",
    "TEXT_MUTED": "#9A9689",
    "ACCENT_CLAY": "#CC785C",
    "ACCENT_CLAY_HOVER": "#B86547",
    "ACCENT_CLAY_BG": "#F4DFD5",
    "ACCENT_CLAY_BG_STRONG": "#EBC8B8",
    "SUCCESS": "#5B8A72",
    "SUCCESS_BG": "#E3ECE4",
    "WARNING": "#D4A24C",
    "WARNING_BG": "#F5EAD2",
    "ERROR": "#B4442E",
    "ERROR_BG": "#F0D9D2",
    "SELECTION_BG": "#F4DFD5",
}

DARK_PALETTE = {
    "BG_PRIMARY": "#1F1B16",
    "BG_SECONDARY": "#2A2520",
    "BG_TERTIARY": "#332D27",
    "BG_DOCK": "#252017",
    "BORDER_SUBTLE": "#3E3830",
    "BORDER_STRONG": "#5A5247",
    "TEXT_PRIMARY": "#EDE6DA",
    "TEXT_SECONDARY": "#B5AC9D",
    "TEXT_MUTED": "#8A8173",
    "ACCENT_CLAY": "#E89A7C",
    "ACCENT_CLAY_HOVER": "#F2B093",
    "ACCENT_CLAY_BG": "#3D2A22",
    "ACCENT_CLAY_BG_STRONG": "#5A3A2D",
    "SUCCESS": "#7FA894",
    "SUCCESS_BG": "#2A3630",
    "WARNING": "#D9B36A",
    "WARNING_BG": "#3A3324",
    "ERROR": "#D06A55",
    "ERROR_BG": "#3C2621",
    "SELECTION_BG": "#3D2A22",
}

# Initial flat module properties to allow backwards compatibility
BG_PRIMARY      = LIGHT_PALETTE["BG_PRIMARY"]
BG_SECONDARY    = LIGHT_PALETTE["BG_SECONDARY"]
BG_TERTIARY     = LIGHT_PALETTE["BG_TERTIARY"]
BG_DOCK         = LIGHT_PALETTE["BG_DOCK"]
BORDER_SUBTLE   = LIGHT_PALETTE["BORDER_SUBTLE"]
BORDER_STRONG   = LIGHT_PALETTE["BORDER_STRONG"]
TEXT_PRIMARY    = LIGHT_PALETTE["TEXT_PRIMARY"]
TEXT_SECONDARY  = LIGHT_PALETTE["TEXT_SECONDARY"]
TEXT_MUTED      = LIGHT_PALETTE["TEXT_MUTED"]
ACCENT_CLAY     = LIGHT_PALETTE["ACCENT_CLAY"]
ACCENT_CLAY_HOVER = LIGHT_PALETTE["ACCENT_CLAY_HOVER"]
ACCENT_CLAY_BG  = LIGHT_PALETTE["ACCENT_CLAY_BG"]
ACCENT_CLAY_BG_STRONG = LIGHT_PALETTE["ACCENT_CLAY_BG_STRONG"]
SUCCESS         = LIGHT_PALETTE["SUCCESS"]
SUCCESS_BG      = LIGHT_PALETTE["SUCCESS_BG"]
WARNING         = LIGHT_PALETTE["WARNING"]
WARNING_BG      = LIGHT_PALETTE["WARNING_BG"]
ERROR           = LIGHT_PALETTE["ERROR"]
ERROR_BG        = LIGHT_PALETTE["ERROR_BG"]
SELECTION_BG    = LIGHT_PALETTE["SELECTION_BG"]

FONT_UI    = "Inter, SF Pro Text, system-ui, sans-serif"
FONT_SERIF = "STIX Two Text, Times New Roman, serif"
FONT_MONO  = "JetBrains Mono, SF Mono, Consolas, monospace"

RADIUS_SM = 4
RADIUS_MD = 6
RADIUS_LG = 8

def set_dark_mode(dark: bool) -> None:
    global is_dark, BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BG_DOCK, BORDER_SUBTLE, BORDER_STRONG, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, ACCENT_CLAY, ACCENT_CLAY_HOVER, ACCENT_CLAY_BG, ACCENT_CLAY_BG_STRONG, SUCCESS, SUCCESS_BG, WARNING, WARNING_BG, ERROR, ERROR_BG, SELECTION_BG
    changed = bool(dark) != is_dark
    is_dark = dark
    palette = DARK_PALETTE if dark else LIGHT_PALETTE
    
    # Update module scope globals so tokens.BG_PRIMARY resolves dynamically
    for k, v in palette.items():
        globals()[k] = v
        
    if changed:
        _theme_signals().changed.emit()
