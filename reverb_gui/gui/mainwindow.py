"""Main application window for reverb-gui.

Hosts the primary user interface components like drag-drop area, settings, progress.
"""

# --- Imports (Added QWidget, QVBoxLayout, QPlainTextEdit) ---
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPlainTextEdit
from PySide6.QtCore import QThreadPool
from .widgets.drop_zone import DropZone
from .workers.transcription_worker import TranscriptionWorker, WorkerSignals
from ..utils.formatting import format_timestamp_ms
import pathlib
from typing import List, Tuple, Any


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self, models_dir: pathlib.Path) -> None:
        """Initializes the main window.

        Args:
            models_dir: Path to the directory containing downloaded models.
        """
        super().__init__()
        self.setWindowTitle("reverb-gui")
        self.resize(800, 600)
        self.models_dir = models_dir  # Store models_dir

        # Setup thread pool for background tasks
        self.thread_pool = QThreadPool()
        print(f"Multithreading with maximum {self.thread_pool.maxThreadCount()} threads")

        # --- Layout Setup (Modified) ---
        # Create a main container widget and layout
        main_container = QWidget()
        layout = QVBoxLayout(main_container)

        # Create and add the DropZone
        self.drop_zone = DropZone()
        layout.addWidget(self.drop_zone, stretch=1)  # Give it some stretch factor

        # Create and add the Transcript Display (Repurposed for unified output)
        self.transcript_display = QPlainTextEdit()
        self.transcript_display.setPlaceholderText("Speaker-assigned transcript will appear here...")  # Updated placeholder
        self.transcript_display.setReadOnly(True)
        layout.addWidget(self.transcript_display, stretch=3)  # Increased stretch factor

        # Create and add the Diarization Display (Now hidden)
        self.diarization_display = QPlainTextEdit()
        self.diarization_display.setPlaceholderText("Speaker segments will appear here...")
        self.diarization_display.setReadOnly(True)
        self.diarization_display.setVisible(False)  # Hide this widget
        layout.addWidget(self.diarization_display, stretch=0)  # No stretch

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
        worker = TranscriptionWorker(
            input_path=file_path,
            models_dir=self.models_dir,  # Pass stored models_dir
            signals=signals
        )

        # Connect worker signals to slots
        signals.result.connect(self._on_transcription_result)
        signals.error.connect(self._on_transcription_error)
        signals.finished.connect(self._on_transcription_finished)
        # signals.progress.connect(self._update_progress) # Future connection

        # Execute the worker in the thread pool
        self.thread_pool.start(worker)

    def _on_transcription_result(self, result_data: List[Tuple[float, float, str, str]]) -> None:  # Updated type hint
        """Handles the successful result from the transcription worker. (Modified)

        Args:
            result_data: A list of tuples, where each tuple is
                         (start_time, end_time, speaker_label, word_text).
        """
        unified_transcript = result_data  # Rename for clarity
        print("MainWindow: Received unified transcription result.")
        print(f"  Total Word Segments: {len(unified_transcript)}")

        # --- Format and Update GUI elements (Modified) ---
        self.transcript_display.clear()
        self.diarization_display.clear()  # Clear even if hidden

        if not unified_transcript:
            self.transcript_display.setPlainText("No transcription results returned.")
            return

        # Group consecutive words by the same speaker
        formatted_lines = []
        current_speaker = None
        current_line = ""
        line_start_time = -1.0
        line_end_time = -1.0

        for start, end, speaker, word in unified_transcript:
            # Strip potential whitespace/special tokens if needed
            word = word.strip()
            if not word:  # Skip empty tokens if any
                continue

            if speaker != current_speaker:
                # Finalize previous line if it exists
                if current_line:
                    formatted_start = format_timestamp_ms(line_start_time)
                    formatted_end = format_timestamp_ms(line_end_time)
                    header = f"[{formatted_start} - {formatted_end}] {current_speaker}:"
                    # Add blank line separator if not the first entry
                    prefix = "\n" if formatted_lines else ""
                    formatted_lines.append(f"{prefix}{header}\n{current_line}")

                # Start new line
                current_speaker = speaker
                current_line = word
                line_start_time = start
                line_end_time = end
            else:
                # Append word to current line
                current_line += f" {word}"
                line_end_time = max(line_end_time, end)  # Update end time

        # Add the last accumulated line
        if current_line:
            formatted_start = format_timestamp_ms(line_start_time)
            formatted_end = format_timestamp_ms(line_end_time)
            header = f"[{formatted_start} - {formatted_end}] {current_speaker}:"
            # Add blank line separator if not the first entry
            prefix = "\n" if formatted_lines else ""
            formatted_lines.append(f"{prefix}{header}\n{current_line}")

        self.transcript_display.setPlainText("\n".join(formatted_lines))
        # --- End Update GUI ---

    def _on_transcription_error(self, error_details: Tuple[Any, Any, str]) -> None:
        """Handles errors reported by the transcription worker."""
        exc_type, exc_value, tb_str = error_details
        print(f"MainWindow: Transcription Error! Type: {exc_type.__name__}, Value: {exc_value}")
        print(f"Traceback:\n{tb_str}")
        # Show error message dialog to the user (Consider using QMessageBox)
        error_message = f"An error occurred during transcription:\nType: {exc_type.__name__}\nDetails: {exc_value}"
        self.transcript_display.setPlainText(error_message)  # Display error in main area
        self.diarization_display.clear()  # Clear hidden area too
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