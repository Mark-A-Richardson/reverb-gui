"""Core transcription pipeline engine.

Wraps FFmpeg conversion and Reverb inference.
"""

import dataclasses
import pathlib
from typing import List, Optional


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

    Args:
        input_path: Path to the input audio or video file.

    Returns:
        A list of AlignedLine objects representing the speaker-aware transcript.

    Raises:
        FileNotFoundError: If the input_path does not exist.
        # TODO: Add exceptions for FFmpeg errors, Reverb errors, etc.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Placeholder: Processing {input_path.name}...")
    # TODO: Implement FFmpeg conversion
    # TODO: Implement Reverb ASR + Diarization
    # TODO: Implement word/speaker alignment

    # Placeholder return value
    return [
        AlignedLine(speaker="SPEAKER_00", start_time=0.5, end_time=2.1, text="Hello world."),
        AlignedLine(speaker="SPEAKER_01", start_time=2.5, end_time=4.0, text="This is a test."),
    ]
