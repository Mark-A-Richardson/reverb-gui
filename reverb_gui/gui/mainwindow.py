"""Main application window for reverb-gui.

Hosts the primary user interface components like drag-drop area, settings, progress.
"""

from PySide6.QtWidgets import QMainWindow
from .widgets.drop_zone import DropZone # Import the custom widget
import pathlib


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self) -> None:
        """Initializes the main window."""
        super().__init__()
        self.setWindowTitle("reverb-gui")
        self.resize(800, 600)

        # Use the custom DropZone widget as the central widget
        self.drop_zone = DropZone()
        self.setCentralWidget(self.drop_zone)

        # Connect the signal from the drop zone to our handler slot
        self.drop_zone.fileDropped.connect(self._handle_file_drop)

    def _handle_file_drop(self, file_path: pathlib.Path) -> None:
        """Handles the fileDropped signal from the DropZone widget."""
        print(f"MainWindow received file drop: {file_path}")
        # TODO: Validate file type (audio/video)
        # TODO: Trigger transcription pipeline (Phase 3)

    # TODO: Add methods for settings panel integration
    # TODO: Add methods for progress bar updates
