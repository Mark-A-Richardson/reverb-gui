"""Core transcription pipeline engine.

Wraps FFmpeg conversion and Reverb inference.
"""

import dataclasses
import pathlib
import tempfile
from typing import List, Optional

# Import the conversion utility
from ..utils.ffmpeg import convert_to_wav


@dataclasses.dataclass
class AlignedLine:
    """Represents a single line/segment in the final transcript.

    Attributes:
        speaker: Optional speaker identifier (e.g., 'SPEAKER_01').
        start_time: Start time of the segment in seconds.
        end_time: End time of the segment in seconds.
        text: The transcribed text content of the segment.
    """
    speaker: Optional[str]
    start_time: float
    end_time: float
    text: str


def transcribe(input_path: pathlib.Path) -> List[AlignedLine]:
    """Processes an audio/video file through FFmpeg and Reverb.

    First converts the input to a 16kHz/16bit mono WAV file using FFmpeg,
    then (currently) processes the WAV file with placeholder logic.

    Args:
        input_path: Path to the input audio or video file.

    Returns:
        A list of AlignedLine objects representing the speaker-aware transcript.

    Raises:
        FileNotFoundError: If the input_path or the ffmpeg command does not exist.
        RuntimeError: If the FFmpeg conversion process fails.
        ValueError: If FFmpeg conversion arguments are invalid.
        # TODO: Add exceptions for Reverb errors, etc.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    wav_path: Optional[pathlib.Path] = None
    temp_dir_path: Optional[pathlib.Path] = None
    was_temporary: bool = False

    try:
        print(f"Engine: Converting {input_path.name} to WAV...")
        # Convert using default settings (16kHz, 16bit, mono, temp dir)
        wav_path = convert_to_wav(input_path)
        # Check if the output path is in a standard temporary location
        try:
            # Use same logic as in ffmpeg.py to identify temp dir
            temp_dir = tempfile.gettempdir()
            if str(wav_path.parent).startswith(temp_dir) and \
               wav_path.parent.name.startswith("reverb_gui_ffmpeg_"):
                 was_temporary = True
                 temp_dir_path = wav_path.parent
                 print(f"Engine: Temporary WAV created at {wav_path} in {temp_dir_path}")
            else:
                 print(f"Engine: WAV created at {wav_path} (not temporary)")
        except Exception as path_err: # Catch potential issues with path checks
            print(f"Warning: Could not determine if path {wav_path} is temporary: {path_err}")
            was_temporary = False

        print(f"Engine: Processing WAV file {wav_path.name}...")
        # TODO: Implement Reverb ASR + Diarization using wav_path
        # TODO: Implement word/speaker alignment

        # Placeholder return value (using info from the converted wav)
        return [
            AlignedLine(speaker="SPEAKER_00", start_time=0.5, end_time=2.1, text=f"Processed {wav_path.name}"),
            AlignedLine(speaker="SPEAKER_01", start_time=2.5, end_time=4.0, text="This is a test."),
        ]

    finally:
        # Cleanup temporary file and directory if created
        if was_temporary and wav_path and wav_path.exists():
            print(f"Engine: Cleaning up temporary WAV file: {wav_path}")
            try:
                wav_path.unlink()
            except OSError as e:
                print(f"Engine: Warning - could not delete temp file {wav_path}: {e}")

        if was_temporary and temp_dir_path and temp_dir_path.exists():
             print(f"Engine: Cleaning up temporary directory: {temp_dir_path}")
             try:
                 temp_dir_path.rmdir() # Only removes if empty
             except OSError as e:
                 print(f"Engine: Warning - could not remove temp dir {temp_dir_path} (might not be empty): {e}")
