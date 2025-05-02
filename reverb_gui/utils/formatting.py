import math
from typing import Union

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
    milliseconds = int(math.modf(total_seconds)[0] * 1000)
    total_seconds = int(total_seconds)

    hrs = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hrs:02}:{mins:02}:{secs:02}.{milliseconds:03}"
