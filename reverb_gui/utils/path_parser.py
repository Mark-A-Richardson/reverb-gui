import os
import codecs
from pathlib import Path
from typing import Optional

def parse_env_path(raw: Optional[str]) -> Optional[Path]:
    """
    Given a raw string (e.g. from os.environ or a .env file),
    strip quotes, decode escape-sequences, expand ~ and $VARS,
    normalize separators, and return a pathlib.Path.

    Returns None if the input is None or empty after stripping.
    """
    if not raw:
        return None

    # 1. strip outer whitespace, then quotes, then inner whitespace
    s = raw.strip()      # Strip outer whitespace
    s = s.strip('\'"')  # Strip quotes
    s = s.strip()      # Strip inner whitespace (that might have been inside quotes)
    if not s:
        return None # Return None if string becomes empty after stripping

    # 2. expand ~ and any embedded environment variables
    s = os.path.expanduser(os.path.expandvars(s))

    # 3. collapse redundant separators/up-levels (e.g. "foo/../bar", mix of slashes)
    s = os.path.normpath(s)

    return Path(s)

# Example usage (for manual testing if needed)
if __name__ == "__main__":
    examples = [
        r'"C:/Program Files/ffmpeg/bin/ffmpeg.exe"',
        r'"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"',
        r'"C:\Program Files\ffmpeg\bin\ffmpeg.exe"', # Test the problematic case
        r'~/myapp/data.txt',
        r'$HOME\other\data.log', # Assumes HOME is set
        r'   "/var/log/syslog"  ',
        r'""', # Empty quotes
        r'   ',
        None,
        r'relative/path',
        r'ffmpeg' # Simple command name
    ]

    print("--- Parsing Examples ---")
    for raw in examples:
        try:
            p = parse_env_path(raw)
            if p:
                print(f"{raw!r:<45}  →  {p!s:<45} (exists? {p.exists()})")
            else:
                print(f"{raw!r:<45}  →  None")
        except Exception as e:
            print(f"{raw!r:<45}  →  ERROR: {e}")

    print("\n--- Checking FFMPEG_PATH from environment ---")
    ffmpeg_env = os.getenv("FFMPEG_PATH")
    if ffmpeg_env:
        try:
            parsed = parse_env_path(ffmpeg_env)
            print(f"FFMPEG_PATH={ffmpeg_env!r}")
            print(f"Parsed as: {parsed!s}")
        except Exception as e:
            print(f"Error parsing FFMPEG_PATH={ffmpeg_env!r}: {e}")
    else:
        print("FFMPEG_PATH environment variable not set.")
