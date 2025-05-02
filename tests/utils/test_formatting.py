from typing import List, Tuple
import pytest

from reverb_gui.utils.formatting import format_timestamp_ms, format_transcript_lines

# Test cases for format_timestamp_ms
@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "00:00:00.000"),
        (5.569, "00:00:05.569"),
        (62.22, "00:01:02.220"),
        (3600, "01:00:00.000"),
        (3661.1234, "01:01:01.123"),
        (-1, "00:00:00.000"), # Handle negative input gracefully
        (None, "00:00:00.000"), # Handle None input gracefully
        ("invalid", "00:00:00.000"), # Handle non-numeric input
    ]
)
def test_format_timestamp_ms(seconds, expected):
    """Tests the format_timestamp_ms function with various inputs."""
    assert format_timestamp_ms(seconds) == expected

# Test case for format_transcript_lines
def test_format_transcript_lines():
    """Tests the format_transcript_lines function for correct output format."""
    sample_data: List[Tuple[float, float, str, str]] = [
        (0.99, 5.569, "SPEAKER_02", "Hello"),
        (5.569, 5.57, "SPEAKER_02", "there."), # This word continues the previous segment
        (8.55, 37.36, "SPEAKER_00", "This is a test sentence."),
        (39.06, 44.95, "SPEAKER_02", "Another segment."),
    ]

    # Expected output based on the logic in format_transcript_lines
    expected_output = (
        "[00:00:00.990 - 00:00:05.570] SPEAKER_02:\n"  # Start/end times consolidate for the speaker block
        "Hello there.\n"                             # Text concatenates
        "\n"                                         # Blank line before next speaker
        "[00:00:08.550 - 00:00:37.360] SPEAKER_00:\n"
        "This is a test sentence.\n"
        "\n"
        "[00:00:39.060 - 00:00:44.950] SPEAKER_02:\n"
        "Another segment."
    )

    actual_output = format_transcript_lines(sample_data)
    # Use pytest's multiline diff feature by asserting equality directly
    assert actual_output == expected_output

# Test case for empty input to format_transcript_lines
def test_format_transcript_lines_empty():
    """Tests format_transcript_lines with empty input data."""
    assert format_transcript_lines([]) == ""
