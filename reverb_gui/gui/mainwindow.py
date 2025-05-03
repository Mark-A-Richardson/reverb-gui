"""Main application window for reverb-gui.

Hosts the primary user interface components like drag-drop area, settings, progress.
"""

# --- Imports (Added QWidget, QVBoxLayout, QPlainTextEdit) ---
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPlainTextEdit,
    QGroupBox, QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox, QLabel
)
from PySide6.QtCore import QThreadPool
from .widgets.drop_zone import DropZone
from .workers.transcription_worker import TranscriptionWorker, WorkerSignals
from ..utils.formatting import format_transcript_lines
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

        # --- ASR Settings GroupBox (Added) ---
        self._setup_asr_settings_widgets(layout)
        # --- End ASR Settings --- 

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

    def _setup_asr_settings_widgets(self, parent_layout: QVBoxLayout) -> None:
        """Creates and adds the ASR settings group box and widgets."""
        asr_group_box = QGroupBox("ASR Settings")
        form_layout = QFormLayout()

        # Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["ctc_prefix_beam_search", "attention_rescoring"])
        form_layout.addRow(QLabel("Mode:"), self.mode_combo)

        # Beam Size
        self.beam_size_spin = QSpinBox()
        self.beam_size_spin.setRange(1, 50) # Reasonable range
        self.beam_size_spin.setValue(10) # Default from engine
        form_layout.addRow(QLabel("Beam Size:"), self.beam_size_spin)

        # Length Penalty
        self.length_penalty_spin = QDoubleSpinBox()
        self.length_penalty_spin.setRange(-10.0, 10.0)
        self.length_penalty_spin.setDecimals(2)
        self.length_penalty_spin.setSingleStep(0.1)
        self.length_penalty_spin.setValue(0.0) # Default from engine
        form_layout.addRow(QLabel("Length Penalty:"), self.length_penalty_spin)

        # CTC Weight (Relevant for attention_rescoring)
        self.ctc_weight_label = QLabel("CTC Weight:")
        self.ctc_weight_spin = QDoubleSpinBox()
        self.ctc_weight_spin.setRange(0.0, 1.0)
        self.ctc_weight_spin.setDecimals(2)
        self.ctc_weight_spin.setSingleStep(0.05)
        self.ctc_weight_spin.setValue(0.1) # Default from engine
        form_layout.addRow(self.ctc_weight_label, self.ctc_weight_spin)

        # Reverse Weight (Relevant for attention_rescoring)
        self.reverse_weight_label = QLabel("Reverse Weight:")
        self.reverse_weight_spin = QDoubleSpinBox()
        self.reverse_weight_spin.setRange(0.0, 1.0)
        self.reverse_weight_spin.setDecimals(2)
        self.reverse_weight_spin.setSingleStep(0.05)
        self.reverse_weight_spin.setValue(0.0) # Default from engine
        form_layout.addRow(self.reverse_weight_label, self.reverse_weight_spin)

        # Blank Penalty
        self.blank_penalty_spin = QDoubleSpinBox()
        self.blank_penalty_spin.setRange(0.0, 10.0)
        self.blank_penalty_spin.setDecimals(2)
        self.blank_penalty_spin.setSingleStep(0.1)
        self.blank_penalty_spin.setValue(0.0) # Default from engine
        form_layout.addRow(QLabel("Blank Penalty:"), self.blank_penalty_spin)

        # Verbatimicity
        self.verbatimicity_spin = QDoubleSpinBox()
        self.verbatimicity_spin.setRange(0.0, 1.0)
        self.verbatimicity_spin.setDecimals(2)
        self.verbatimicity_spin.setSingleStep(0.1)
        self.verbatimicity_spin.setValue(1.0) # Default from engine was 0.5, but 1.0 seems more standard
        form_layout.addRow(QLabel("Verbatimicity:"), self.verbatimicity_spin)

        # Set layout for group box
        asr_group_box.setLayout(form_layout)
        parent_layout.addWidget(asr_group_box, stretch=0) # No stretch for settings

        # Connect mode change signal
        self.mode_combo.currentIndexChanged.connect(self._update_asr_param_widgets)
        # Initial update
        self._update_asr_param_widgets()

    def _update_asr_param_widgets(self) -> None:
        """Enables/disables widgets based on the selected ASR mode."""
        selected_mode = self.mode_combo.currentText()
        is_rescoring_mode = (selected_mode == "attention_rescoring")

        self.ctc_weight_label.setEnabled(is_rescoring_mode)
        self.ctc_weight_spin.setEnabled(is_rescoring_mode)
        self.reverse_weight_label.setEnabled(is_rescoring_mode)
        self.reverse_weight_spin.setEnabled(is_rescoring_mode)

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

        # --- Get ASR parameters from GUI (Added) ---
        asr_params = {
            "mode": self.mode_combo.currentText(),
            "beam_size": self.beam_size_spin.value(),
            "length_penalty": self.length_penalty_spin.value(),
            "ctc_weight": self.ctc_weight_spin.value(),
            "reverse_weight": self.reverse_weight_spin.value(),
            "blank_penalty": self.blank_penalty_spin.value(),
            "verbatimicity": self.verbatimicity_spin.value(),
        }
        # --- End Get ASR parameters ---

        # Create worker and signals
        signals = WorkerSignals()
        worker = TranscriptionWorker(
            input_path=file_path,
            models_dir=self.models_dir,  # Pass stored models_dir
            asr_params=asr_params, # Pass ASR parameters
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
        if not unified_transcript:
            self.transcript_display.setPlainText("(No transcription results)")
            return

        # Call the utility function to format the entire transcript
        formatted_text = format_transcript_lines(unified_transcript)
        self.transcript_display.setPlainText(formatted_text)
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