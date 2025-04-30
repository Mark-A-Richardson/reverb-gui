import sys
from typing import NoReturn

from PySide6.QtWidgets import QApplication, QMainWindow

def launch_gui() -> NoReturn:
    """Initializes and runs the main application window."""
    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setWindowTitle("reverb-gui")
    window.resize(800, 600)  # Set a default size
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
