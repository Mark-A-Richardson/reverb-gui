"""Main application window for reverb-gui.

Hosts the primary user interface components like drag-drop area, settings, progress.
"""

# --- Imports (Added QWidget, QVBoxLayout, QPlainTextEdit) ---
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPlainTextEdit
from PySide6.QtCore import QThreadPool
from .widgets.drop_zone import DropZone
from .workers.transcription_worker import TranscriptionWorker, WorkerSignals
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

        # --- Layout Setup (Modified) ---
        # Create a main container widget and layout
        main_container = QWidget()
        layout = QVBoxLayout(main_container)

        # Create and add the DropZone
        self.drop_zone = DropZone()
        layout.addWidget(self.drop_zone, stretch=1) # Give it some stretch factor

        # Create and add the Transcript Display (New)
        self.transcript_display = QPlainTextEdit()
        self.transcript_display.setPlaceholderText("Full transcript will appear here...")
        self.transcript_display.setReadOnly(True)
        layout.addWidget(self.transcript_display, stretch=2) # More stretch factor

        # Create and add the Diarization Display (New)
        self.diarization_display = QPlainTextEdit()
        self.diarization_display.setPlaceholderText("Speaker segments will appear here...")
        self.diarization_display.setReadOnly(True)
        layout.addWidget(self.diarization_display, stretch=1) # Less stretch factor

        # Set the main container as the central widget
        self.setCentralWidget(main_container)
        # --- End Layout Setup ---

        # Connect the signal from the drop zone to our handler slot
        self.drop_zone.fileDropped.connect(self._handle_file_drop)

    def _handle_file_drop(self, file_path: pathlib.Path) -> None:
        """Handles the fileDropped signal from the DropZone widget.

        Starts the transcription worker in a background thread.
        """
        print(f"MainWindow: File drop received: {file_path}")
        print("MainWindow: Starting transcription worker...")

        # --- Clear previous results (Added) ---
        self.transcript_display.clear()
        self.diarization_display.clear()
        # --- End Clear ---

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

    def _on_transcription_result(self, result_data: Tuple[str, List[Tuple[float, float, str]]]) -> None:
        """Handles the successful result from the transcription worker. (Modified)

        Args:
            result_data: A tuple containing (full_text, diarization_segments).
                         diarization_segments is List[Tuple[start, end, speaker]].
        """
        full_text, diarization_segments = result_data
        print("MainWindow: Received transcription result.")
        print(f"  Full Text Length: {len(full_text)}")
        print(f"  Diarization Segments: {len(diarization_segments)}")

        # --- Update GUI elements (Modified) ---
        self.transcript_display.setPlainText(full_text)

        # Format diarization segments for display
        diarization_text_lines = []
        if diarization_segments:
            for start, end, speaker in diarization_segments:
                diarization_text_lines.append(f"[{start:.2f}s -> {end:.2f}s] {speaker}")
        else:
            diarization_text_lines.append("No speaker segments identified.")

        self.diarization_display.setPlainText("\n".join(diarization_text_lines))
        # --- End Update GUI ---

    def _on_transcription_error(self, error_details: Tuple[Any, Any, str]) -> None:
        """Handles errors reported by the transcription worker."""
        exc_type, exc_value, tb_str = error_details
        print(f"MainWindow: Transcription Error! Type: {exc_type.__name__}, Value: {exc_value}")
        print(f"Traceback:\n{tb_str}")
        # Show error message dialog to the user (Consider using QMessageBox)
        self.transcript_display.setPlaceholderText("An error occurred during transcription.")
        self.diarization_display.setPlaceholderText(f"Error: {exc_value}")
        self.drop_zone.setText("Error during processing. Drop another file.") # Update text on error

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
    