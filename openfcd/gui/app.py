from __future__ import annotations
import sys


def main() -> None:
    try:
        from PyQt6.QtWidgets import QApplication
        from openfcd.gui.mainwindow import MainWindow
        app = QApplication(sys.argv)
        app.setApplicationName("OpenFCD")
        win = MainWindow()
        win.show()
        sys.exit(app.exec())
    except ImportError:
        print("PyQt6 not installed. Run: pip install 'openfcd[gui]'")
        sys.exit(1)
