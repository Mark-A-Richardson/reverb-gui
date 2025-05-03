"""Worker for running the transcription process in a background thread."""

import pathlib
import traceback
import sys
from typing import List, Tuple, Any, Dict

from PySide6.QtCore import QObject, QRunnable, Signal

# Assuming transcribe is defined correctly in engine.py
# Import the transcribe function directly
from ...pipeline.engine import transcribe


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc())

    result
        `tuple` data returned from processing, Tuple[str, List[Tuple[float, float, str]]]

    progress
        `int` indicating % progress (optional, for future use)
    """
    finished = Signal()
    error = Signal(tuple)
    result = Signal(tuple) # Emits Tuple[str, List[Tuple[float, float, str]]]
    # progress = Signal(int) # Placeholder for future progress reporting


class TranscriptionWorker(QRunnable):
    """Worker thread for executing the transcription task."""

    def __init__(self, input_path: pathlib.Path, models_dir: pathlib.Path, asr_params: Dict, signals: WorkerSignals) -> None:
        """Initializes the worker.

        Args:
            input_path: The path to the audio/video file to transcribe.
            models_dir: The path to the directory containing downloaded models.
            asr_params: Dictionary containing ASR transcription parameters from the GUI.
            signals: An instance of WorkerSignals to communicate back.
        """
        super().__init__()
        self.input_path: pathlib.Path = input_path
        self.models_dir: pathlib.Path = models_dir
        self.asr_params: Dict = asr_params
        self.signals: WorkerSignals = signals

    def run(self) -> None:
        """Execute the transcription task."""
        try:
            # Call the transcribe function directly, passing models_dir and asr_params
            result_data: List[Tuple[float, float, str, str]] = transcribe(
                self.input_path,
                models_dir=self.models_dir,
                asr_params=self.asr_params
            )

            # --- Emit results --- 
            self.signals.result.emit(result_data)
        except Exception as e:
            # Capture traceback details
            exc_type, exc_value, tb = sys.exc_info()
            if exc_type and exc_value and tb:
                formatted_traceback = "".join(traceback.format_exception(exc_type, exc_value, tb))
                error_details: Tuple[Any, Any, str] = (exc_type, exc_value, formatted_traceback)
                self.signals.error.emit(error_details)
            else:
                # Fallback if sys.exc_info() fails somehow
                fallback_error = (type(e), e, traceback.format_exc())
                self.signals.error.emit(fallback_error)
        finally:
            # Always emit finished signal
            self.signals.finished.emit()
