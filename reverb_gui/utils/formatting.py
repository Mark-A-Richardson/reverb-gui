import math
from typing import Union, List, Tuple

def format_timestamp_ms(seconds: Union[float, int]) -> str:
    """Formats a duration in seconds into HH:MM:SS.ms string.

    Args:
        seconds: The duration in seconds.

    Returns:
        A string formatted as HH:MM:SS.ms.
    """
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "00:00:00.000"

    total_seconds = seconds
    # Round before casting to int to handle precision issues
    milliseconds = int(round(math.modf(total_seconds)[0] * 1000))
    total_seconds = int(total_seconds)

    hrs = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hrs:02}:{mins:02}:{secs:02}.{milliseconds:03}"

def format_transcript_lines(result_data: List[Tuple[float, float, str, str]]) -> str:
    """Formats the unified transcription data into a display string.

    Args:
        result_data: A list of tuples, where each tuple is
                     (start_time, end_time, speaker_label, word_text).

    Returns:
        A formatted string ready for display, with timestamps,
        speaker labels, newlines, and spacing.
    """
    formatted_lines = []
    current_speaker = None
    current_line = ""
    line_start_time = -1.0
    line_end_time = -1.0

    for start, end, speaker, word in result_data:
        # Strip potential whitespace/special tokens if needed
        word = word.strip()
        if not word:  # Skip empty tokens if any
            continue

        if speaker != current_speaker:
            # Finalize previous line if it exists
            if current_line:
                # Use the fixed format_timestamp_ms here
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
        # Use the fixed format_timestamp_ms here
        formatted_start = format_timestamp_ms(line_start_time)
        formatted_end = format_timestamp_ms(line_end_time)
        header = f"[{formatted_start} - {formatted_end}] {current_speaker}:"
        # Add blank line separator if not the first entry
        prefix = "\n" if formatted_lines else ""
        formatted_lines.append(f"{prefix}{header}\n{current_line}")

    # Join all formatted speaker segments into a single string
    return "\n".join(formatted_lines)
