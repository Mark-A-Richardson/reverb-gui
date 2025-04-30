import sys
from typing import NoReturn

from PySide6.QtWidgets import QApplication
from .gui.mainwindow import MainWindow  

def launch_gui() -> NoReturn:
    """Initializes and runs the main application window."""
    app = QApplication(sys.argv)

    window = MainWindow() 
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
