"""Main application window for reverb-gui.

Hosts the primary user interface components like drag-drop area, settings, progress.
"""

from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import QThreadPool
from .widgets.drop_zone import DropZone
from .workers.transcription_worker import TranscriptionWorker, WorkerSignals
from ..pipeline.engine import AlignedLine
import pathlib
from typing import List, Tuple, Any


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self) -> None:
        """Initializes the main window."""
        super().__init__()
        self.setWindowTitle("reverb-gui")
        self.resize(800, 600)

        # Setup thread pool for background tasks
        self.thread_pool = QThreadPool()
        print(f"Multithreading with maximum {self.thread_pool.maxThreadCount()} threads")

        # Use the custom DropZone widget as the central widget
        self.drop_zone = DropZone()
        self.setCentralWidget(self.drop_zone)

        # Connect the signal from the drop zone to our handler slot
        self.drop_zone.fileDropped.connect(self._handle_file_drop)

    def _handle_file_drop(self, file_path: pathlib.Path) -> None:
        """Handles the fileDropped signal from the DropZone widget.

        Starts the transcription worker in a background thread.
        """
        print(f"MainWindow: File drop received: {file_path}")
        print("MainWindow: Starting transcription worker...")

        # Disable drop zone while processing
        self.drop_zone.setEnabled(False)
        self.drop_zone.setText("Processing... Please Wait")  # Update text

        # Create worker and signals
        signals = WorkerSignals()
        worker = TranscriptionWorker(file_path, signals)

        # Connect worker signals to slots
        signals.result.connect(self._on_transcription_result)
        signals.error.connect(self._on_transcription_error)
        signals.finished.connect(self._on_transcription_finished)
        # signals.progress.connect(self._update_progress) # Future connection

        # Execute the worker in the thread pool
        self.thread_pool.start(worker)

    def _on_transcription_result(self, result: List[AlignedLine]) -> None:
        """Handles the successful result from the transcription worker."""
        print("MainWindow: Transcription successful!")
        # For now, just print the result lines
        for line in result:
            print(f"  [{line.start_time:.2f}s - {line.end_time:.2f}s] {line.speaker or 'UNKNOWN'}: {line.text}")
        # TODO: Display results in the GUI (e.g., text area)

    def _on_transcription_error(self, error_details: Tuple[Any, Any, str]) -> None:
        """Handles errors reported by the transcription worker."""
        exc_type, exc_value, tb_str = error_details
        print(f"MainWindow: Transcription Error! Type: {exc_type.__name__}, Value: {exc_value}")
        print(f"Traceback:\n{tb_str}")
        # TODO: Show error message dialog to the user
        self.drop_zone.setText("Error during processing. Drop another file.")  # Update text on error

    def _on_transcription_finished(self) -> None:
        """Handles the finished signal from the transcription worker."""
        print("MainWindow: Transcription worker finished.")
        # Re-enable drop zone
        self.drop_zone.setEnabled(True)
        # Reset text only if there wasn't an error message set previously
        if "Error" not in self.drop_zone.text():
            self.drop_zone.setText("Drag and Drop Audio/Video File Here")

    # TODO: Add methods for settings panel integration
    # TODO: Add methods for progress bar updates
