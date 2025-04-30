"""Worker for running the transcription process in a background thread."""

import pathlib
import traceback
import sys
from typing import List, Tuple, Any

from PySide6.QtCore import QObject, QRunnable, Signal

# Assuming AlignedLine is defined correctly in engine.py
# Adjust the import path based on your final structure if needed
from ...pipeline.engine import transcribe, AlignedLine


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc())

    result
        `list` data returned from processing, List[AlignedLine]

    progress
        `int` indicating % progress (optional, for future use)
    """
    finished = Signal()
    error = Signal(tuple)
    result = Signal(list) # Emits List[AlignedLine]
    # progress = Signal(int) # Placeholder for future progress reporting


class TranscriptionWorker(QRunnable):
    """Worker thread for executing the transcription task."""

    def __init__(self, input_path: pathlib.Path, signals: WorkerSignals) -> None:
        """Initializes the worker.

        Args:
            input_path: The path to the audio/video file to transcribe.
            signals: An instance of WorkerSignals to communicate back.
        """
        super().__init__()
        self.input_path: pathlib.Path = input_path
        self.signals: WorkerSignals = signals

    def run(self) -> None:
        """Execute the transcription task."""
        try:
            result_data: List[AlignedLine] = transcribe(self.input_path)
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
        else:
            # Emit result if no exception occurred
            self.signals.result.emit(result_data)
        finally:
            # Always emit finished signal
            self.signals.finished.emit()
