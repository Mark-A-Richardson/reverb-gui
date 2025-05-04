"""Main application window for reverb-gui.

Hosts the primary user interface components like drag-drop area, settings, progress.
"""

# --- Imports (Added QWidget, QVBoxLayout, QPlainTextEdit) ---
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPlainTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QLabel,
    QToolButton, QFrame, QSizePolicy, QGridLayout
)
from PySide6.QtCore import Qt, Slot, QThreadPool
from .widgets.drop_zone import DropZone
from .workers.transcription_worker import TranscriptionWorker, WorkerSignals
from ..utils.formatting import format_transcript_lines
import pathlib
from typing import List, Dict, Tuple, Any


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
        self.drop_zone.fileDropped.connect(self._handle_file_drop)
        # Set fixed height and expanding horizontal policy for DropZone
        self.drop_zone.setFixedHeight(150) 
        self.drop_zone.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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

    def _setup_asr_settings_widgets(self, parent_layout: QVBoxLayout) -> None:
        """Creates and adds the ASR settings collapsible section and widgets."""
        # Tooltip texts (ensure newlines are escaped for JSON: \\n)
        tooltips: Dict[str, str] = {
            "mode": "Selects the ASR decoding algorithm.",
            "beam_size": "Max hypotheses kept during beam search.\nLarger = potentially better accuracy but slower processing.",
            "length_penalty": "Adjusts preference for longer/shorter sentences.\nPositive values favor longer, negative values favor shorter.\n(Used in 'attention' and 'joint_decoding' modes).",
            "blank_penalty": "Penalty applied to the CTC blank symbol to discourage silence/stalls.\n(Used in modes involving CTC).",
            "ctc_weight": "Weight given to the CTC score component.\n(Used in 'attention_rescoring' and 'joint_decoding' modes).",
            "reverse_weight": "Weight given to the right-to-left decoder component.\nHelps improve punctuation/end-of-sentence accuracy.\n(Used in 'attention_rescoring' mode).",
            "verbatimicity": "Controls output strictness (e.g., number formatting, punctuation).\n1.0 = most verbatim/raw, lower values allow more normalization."
        }

        # --- Collapsible Section Setup ---
        self.asr_settings_toggle_button = QToolButton()
        self.asr_settings_toggle_button.setText("ASR Settings")
        self.asr_settings_toggle_button.setCheckable(True)
        self.asr_settings_toggle_button.setChecked(False) # Start collapsed
        self.asr_settings_toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.asr_settings_toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.asr_settings_toggle_button.setArrowType(Qt.ArrowType.RightArrow) # Start with right arrow
        # Connect the 'toggled' signal instead of 'pressed'
        self.asr_settings_toggle_button.toggled.connect(self._toggle_asr_settings_visibility)
        parent_layout.addWidget(self.asr_settings_toggle_button)

        # Frame to hold the settings, initially hidden
        self.asr_settings_container = QFrame()
        self.asr_settings_container.setFrameShape(QFrame.Shape.StyledPanel)
        self.asr_settings_container.setVisible(False)
        self.asr_settings_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Use QGridLayout for 2-column layout
        grid_layout = QGridLayout()
        # Set stretch factors: Give columns 1 & 3 priority for horizontal space
        grid_layout.setColumnStretch(1, 1)
        grid_layout.setColumnStretch(3, 1)
        self.asr_settings_container.setLayout(grid_layout) # Set grid layout for the container
        parent_layout.addWidget(self.asr_settings_container)
        # --- End Collapsible Section Setup ---

        # Mode (Spans 3 columns)
        self.mode_label = QLabel("Mode:")
        self.mode_label.setToolTip(tooltips["mode"])
        self.mode_combo = QComboBox() 
        modes = [
            "ctc_greedy_search",
            "ctc_prefix_beam_search",
            "attention",
            "attention_rescoring",
            "joint_decoding"
        ]
        for display_name, internal_id in [(mode, mode) for mode in modes]: 
            self.mode_combo.addItem(display_name, internal_id) # Store internal ID as data
        self.mode_combo.setCurrentIndex(3) # Default to 'Attention Rescoring'
        self.mode_combo.setToolTip(tooltips["mode"])
        self.mode_combo.currentIndexChanged.connect(self._update_asr_param_state)
        grid_layout.addWidget(self.mode_label, 0, 0)       # Row 0, Col 0
        grid_layout.addWidget(self.mode_combo, 0, 1, 1, 3) # Row 0, starting Col 1, span 1 row, span 3 columns

        # --- Column 1 (Left) ---
        
        # Verbatimicity (Moved to Row 1, Col 0/1)
        self.verbatimicity_label = QLabel("Verbatimicity:")
        self.verbatimicity_label.setToolTip(tooltips["verbatimicity"])
        self.verbatimicity_spin = QDoubleSpinBox() 
        self.verbatimicity_spin.setRange(0.0, 1.0) 
        self.verbatimicity_spin.setDecimals(2)
        self.verbatimicity_spin.setSingleStep(0.05) 
        self.verbatimicity_spin.setValue(1.0) 
        self.verbatimicity_spin.setToolTip(tooltips["verbatimicity"])
        grid_layout.addWidget(self.verbatimicity_label, 1, 0) # Row 1, Col 0
        grid_layout.addWidget(self.verbatimicity_spin, 1, 1)   # Row 1, Col 1

        # Beam Size
        self.beam_size_label = QLabel("Beam Size:")
        self.beam_size_label.setToolTip(tooltips["beam_size"])
        self.beam_size_spin = QSpinBox() 
        self.beam_size_spin.setRange(1, 30) # Adjusted range
        self.beam_size_spin.setValue(10)
        self.beam_size_spin.setToolTip(tooltips["beam_size"])
        grid_layout.addWidget(self.beam_size_label, 2, 0) # Row 2, Col 0
        grid_layout.addWidget(self.beam_size_spin, 2, 1)   # Row 2, Col 1

        # Length Penalty
        self.length_penalty_label = QLabel("Length Penalty:")
        self.length_penalty_label.setToolTip(tooltips["length_penalty"])
        self.length_penalty_spin = QDoubleSpinBox() 
        self.length_penalty_spin.setRange(-2.0, 2.0) # Adjusted range
        self.length_penalty_spin.setDecimals(2)
        self.length_penalty_spin.setSingleStep(0.05) 
        self.length_penalty_spin.setValue(0.0) 
        self.length_penalty_spin.setToolTip(tooltips["length_penalty"])
        grid_layout.addWidget(self.length_penalty_label, 3, 0) # Row 3, Col 0
        grid_layout.addWidget(self.length_penalty_spin, 3, 1)   # Row 3, Col 1

        # --- Column 2 (Right) --- 

        # CTC Weight
        self.ctc_weight_label = QLabel("CTC Weight:")
        self.ctc_weight_label.setToolTip(tooltips["ctc_weight"])
        self.ctc_weight_spin = QDoubleSpinBox() 
        self.ctc_weight_spin.setRange(0.0, 1.0) 
        self.ctc_weight_spin.setDecimals(2) 
        self.ctc_weight_spin.setSingleStep(0.05)
        self.ctc_weight_spin.setValue(0.1) 
        self.ctc_weight_spin.setToolTip(tooltips["ctc_weight"])
        grid_layout.addWidget(self.ctc_weight_label, 1, 2) # Row 1, Col 2
        grid_layout.addWidget(self.ctc_weight_spin, 1, 3)   # Row 1, Col 3

        # Reverse Weight
        self.reverse_weight_label = QLabel("Reverse Weight:")
        self.reverse_weight_label.setToolTip(tooltips["reverse_weight"])
        self.reverse_weight_spin = QDoubleSpinBox() 
        self.reverse_weight_spin.setRange(0.0, 0.5) 
        self.reverse_weight_spin.setDecimals(2) 
        self.reverse_weight_spin.setSingleStep(0.05)
        self.reverse_weight_spin.setValue(0.0) 
        self.reverse_weight_spin.setToolTip(tooltips["reverse_weight"])
        grid_layout.addWidget(self.reverse_weight_label, 2, 2) # Row 2, Col 2
        grid_layout.addWidget(self.reverse_weight_spin, 2, 3)   # Row 2, Col 3

        # Blank Penalty
        self.blank_penalty_label = QLabel("Blank Penalty:")
        self.blank_penalty_label.setToolTip(tooltips["blank_penalty"])
        self.blank_penalty_spin = QDoubleSpinBox() 
        self.blank_penalty_spin.setRange(0.0, 2.0) # Adjusted range
        self.blank_penalty_spin.setDecimals(2) 
        self.blank_penalty_spin.setSingleStep(0.05) 
        self.blank_penalty_spin.setValue(0.0) 
        self.blank_penalty_spin.setToolTip(tooltips["blank_penalty"])
        grid_layout.addWidget(self.blank_penalty_label, 3, 2) # Row 3, Col 2
        grid_layout.addWidget(self.blank_penalty_spin, 3, 3)   # Row 3, Col 3

        # Initial setup of enabled/disabled state based on default mode
        self._update_asr_param_state()

    def _get_asr_params(self) -> Dict[str, Any]:
        """Retrieves the current ASR parameters from the GUI widgets."""
        # Determine device based on checkbox state
        device = "cpu" # Default to CPU

        # --- Get ASR parameters ---
        return {
            "mode": self.mode_combo.currentText(),
            "device": device, # Map checkbox to device string
            "beam_size": self.beam_size_spin.value(),
            "length_penalty": self.length_penalty_spin.value(),
            "ctc_weight": self.ctc_weight_spin.value(),
            "reverse_weight": self.reverse_weight_spin.value(),
            "blank_penalty": self.blank_penalty_spin.value(),
            "verbatimicity": self.verbatimicity_spin.value(),
        }
        # --- End Get ASR parameters ---

    @Slot(bool) # Connected to the 'toggled' signal
    def _toggle_asr_settings_visibility(self, is_checked: bool) -> None:
        """Shows/hides the ASR settings container when the toggle button is clicked."""
        # Set visibility based on the button's checked state
        self.asr_settings_container.setVisible(is_checked)
        # Update arrow direction
        arrow = Qt.ArrowType.DownArrow if is_checked else Qt.ArrowType.RightArrow
        self.asr_settings_toggle_button.setArrowType(arrow)

    def _update_asr_param_state(self) -> None:
        """Enables/disables ASR parameter widgets and labels based on the selected mode, following the matrix in reverb-gui_asr_parameters.md."""
        selected_mode = self.mode_combo.currentText()
        beam_enabled = selected_mode in ["ctc_prefix_beam_search", "attention", "attention_rescoring", "joint_decoding"]
        length_penalty_enabled = selected_mode in ["attention", "joint_decoding"]
        blank_penalty_enabled = selected_mode in ["ctc_prefix_beam_search", "attention_rescoring", "joint_decoding"]
        ctc_weight_enabled = selected_mode in ["attention_rescoring", "joint_decoding"]
        reverse_weight_enabled = selected_mode == "attention_rescoring"

        self.beam_size_label.setEnabled(beam_enabled)
        self.beam_size_spin.setEnabled(beam_enabled)

        self.length_penalty_label.setEnabled(length_penalty_enabled)
        self.length_penalty_spin.setEnabled(length_penalty_enabled)

        self.blank_penalty_label.setEnabled(blank_penalty_enabled)
        self.blank_penalty_spin.setEnabled(blank_penalty_enabled)

        self.ctc_weight_label.setEnabled(ctc_weight_enabled)
        self.ctc_weight_spin.setEnabled(ctc_weight_enabled)

        self.reverse_weight_label.setEnabled(reverse_weight_enabled)
        self.reverse_weight_spin.setEnabled(reverse_weight_enabled)

        # Mode, Verbatimicity are always enabled
        # (No need to explicitly setEnabled(True) unless they might be disabled elsewhere)

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
        asr_params = self._get_asr_params()
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