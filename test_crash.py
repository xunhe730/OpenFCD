import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from openfcd.gui.mainwindow import MainWindow

app = QApplication(sys.argv)
win = MainWindow()
win.show()

# Open the project
proj_path = Path("/Volumes/ZXD_PKU/MAC_mini/Research/XJ-robot/OpenFCD/tests/scratch/dummy.ofcd")
if not proj_path.exists():
    print("Project not found")
    sys.exit(1)

# Create a project if not exists
if not (proj_path / "project.yaml").exists():
    win._session.new_project("dummy", proj_path.parent, image_folder="images", file_pattern="Img*.jpg")
else:
    win._session.open_project(proj_path)

# Mock _frames
win._frames = [proj_path / "images" / "Img0.jpg", proj_path / "images" / "Img1.jpg"]

# Mock _sim_tree
class MockItem:
    def __init__(self, idx):
        self._idx = idx
    def data(self, role, val=None):
        return self._idx

import time
from PyQt6.QtCore import QTimer

def test():
    try:
        # Simulate selecting first frame
        print("Selecting frame 0")
        win._sim_tree.currentItem = lambda: MockItem(0)
        win._on_node_selected("image_frame")
        print("Selected frame 0 successfully")
        
        # Simulate selecting second frame
        print("Selecting frame 1")
        win._sim_tree.currentItem = lambda: MockItem(1)
        win._on_node_selected("image_frame")
        print("Selected frame 1 successfully")
        
        app.quit()
    except Exception as e:
        import traceback
        traceback.print_exc()
        app.quit()

QTimer.singleShot(100, test)
app.exec()
