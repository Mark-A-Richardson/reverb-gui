"""Main application window for reverb-gui.

Hosts the primary user interface components like drag-drop area, settings, progress.
"""

from PySide6.QtWidgets import QMainWindow, QLabel # Use QLabel as placeholder for now


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self) -> None:
        """Initializes the main window."""
        super().__init__()
        self.setWindowTitle("reverb-gui")
        self.resize(800, 600)

        # Placeholder central widget - will be replaced by drag-drop area etc.
        central_widget = QLabel("Main Application Window - Content Area")
        central_widget.setStyleSheet("QLabel { font-size: 16px; alignment: center; }")
        self.setCentralWidget(central_widget)

    # TODO: Add methods for drag-drop handling
    # TODO: Add methods for settings panel integration
    # TODO: Add methods for progress bar updates
