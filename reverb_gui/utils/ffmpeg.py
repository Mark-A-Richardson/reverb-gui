"""Utilities for interacting with the FFmpeg executable."""

import shutil
import subprocess
import pathlib
import tempfile
import os
from typing import Optional
import shlex # Import shlex
from .path_parser import parse_env_path # Import the new parser

# Read FFmpeg command path from environment variable, default to 'ffmpeg'
raw_ffmpeg_path = os.getenv("FFMPEG_PATH")
parsed_path = parse_env_path(raw_ffmpeg_path)

if parsed_path:
    # Use the string representation of the parsed path
    FFMPEG_CMD = str(parsed_path)
else:
    # Default if env var is not set or empty, or parsing failed
    FFMPEG_CMD = "ffmpeg"

def check_ffmpeg_availability() -> bool:
    """Checks if the ffmpeg command is available and executable."""
    cmd_path = shutil.which(FFMPEG_CMD)
    if cmd_path:
        print(f"FFmpeg found at: {cmd_path}")
        return True
    else:
        print(f"Error: Command '{FFMPEG_CMD}' not found using shutil.which.")
        print("Please ensure the path is correct in FFMPEG_PATH (if set) or that 'ffmpeg' is in your system's PATH.")
        return False

def convert_to_wav(
    input_path: pathlib.Path,
    output_dir: Optional[pathlib.Path] = None,
    sample_rate: int = 16000,
    bit_depth: int = 16,
    channels: int = 1
) -> pathlib.Path:
    """Converts an input audio/video file to a mono WAV file using FFmpeg.

    Args:
        input_path: Path to the input file.
        output_dir: Directory to save the output WAV file. If None, a system
                    temporary directory is used.
        sample_rate: Target sample rate in Hz (default: 16000).
        bit_depth: Target bit depth (default: 16).
        channels: Target number of channels (default: 1 for mono).

    Returns:
        Path to the generated WAV file.

    Raises:
        FileNotFoundError: If FFmpeg command is not found.
        RuntimeError: If the FFmpeg conversion process fails.
        ValueError: If input arguments are invalid (e.g., bit_depth).
    """
    if not check_ffmpeg_availability():
        raise FileNotFoundError(f"Command '{FFMPEG_CMD}' not found or not executable. Cannot convert audio.")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if bit_depth not in [16, 24, 32]: # Common PCM bit depths
         raise ValueError(f"Unsupported bit depth: {bit_depth}. Must be 16, 24, or 32.")
    pcm_codec = f"pcm_s{bit_depth}le" # e.g., pcm_s16le

    # Create output directory if needed
    if output_dir is None:
        # Use a temporary directory that gets cleaned up automatically
        # Note: We need to manage the lifecycle or pass the directory handle out
        # For simplicity now, let's create a named temporary file directly.
        # Using a directory might be better if FFmpeg needs intermediate files.
        temp_dir = tempfile.mkdtemp(prefix="reverb_gui_ffmpeg_")
        output_dir_path = pathlib.Path(temp_dir)
        print(f"Using temporary directory: {output_dir_path}")
    else:
        output_dir_path = pathlib.Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    # Construct output path (ensure unique name to avoid collisions)
    output_filename = f"{input_path.stem}_converted_{sample_rate}hz_{bit_depth}bit_mono.wav"
    output_wav_path = output_dir_path / output_filename

    # Construct FFmpeg command
    # -i: input file
    # -vn: disable video recording
    # -acodec: audio codec (pcm_s16le for 16-bit PCM)
    # -ar: audio sample rate
    # -ac: audio channels
    # -y: overwrite output files without asking
    # -loglevel error: Only show errors
    command = [
        FFMPEG_CMD,
        '-i', str(input_path),
        '-vn',
        '-acodec', pcm_codec,
        '-ar', str(sample_rate),
        '-ac', str(channels),
        '-y',
        '-loglevel', 'error', # Keep output clean unless error
        # TODO: Add '-progress pipe:1' for progress reporting later
        str(output_wav_path)
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=False)
        # print(f"DEBUG: FFmpeg stdout:\n{result.stdout}") # Often empty with -loglevel error
        # print(f"DEBUG: FFmpeg stderr:\n{result.stderr}") # Can contain info even on success
    except subprocess.CalledProcessError as e:
        print("FFmpeg conversion failed with CalledProcessError!")
        print(f"Command executed: {' '.join(shlex.quote(c) for c in command)}") # Use shlex.quote for safe display
        print(f"Return code: {e.returncode}")
        print(f"Stderr:\n{e.stderr}")
        raise RuntimeError(f"FFmpeg conversion failed for {input_path}. Command: '{FFMPEG_CMD}'. Error: {e.stderr}") from e
    except FileNotFoundError as e:
        print(f"Error executing FFmpeg command (FileNotFoundError): {e}")
        raise FileNotFoundError(f"Failed to execute command '{FFMPEG_CMD}'. Is it a valid executable and accessible? Original error: {e}") from e

    # Check if the output file was actually created (belt-and-suspenders)
    if not output_wav_path.is_file() or output_wav_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg conversion failed to produce output file: {output_wav_path}")

    print(f"FFmpeg conversion successful: {output_wav_path}")
    return output_wav_path

# Example usage (for testing)
if __name__ == "__main__":
    if check_ffmpeg_availability():
        # Create a dummy input file for testing
        dummy_input = pathlib.Path("dummy_input.txt")
        dummy_input.touch()
        print("\nTesting conversion (will fail as input is not audio):")
        try:
            # Test with temporary directory
            wav_path_temp = convert_to_wav(dummy_input)
            print(f" -> Temporary WAV path: {wav_path_temp}")
            if wav_path_temp.exists():
                 # Clean up the temp file/dir if needed (depends on implementation)
                 # For mkdtemp, the dir needs manual cleanup unless using TemporaryDirectory context manager
                 # For tempfile.mkstemp or NamedTemporaryFile, the file is usually deleted on close
                 # Since we used mkdtemp, let's clean the parent dir of the file path
                 # Careful: only delete if it's truly temporary!
                 if tempfile.gettempdir() in str(wav_path_temp.parent):
                      print(f" -> Cleaning up temp file: {wav_path_temp}")
                      wav_path_temp.unlink()
                      try:
                           wav_path_temp.parent.rmdir()
                           print(f" -> Cleaned up temp dir: {wav_path_temp.parent}")
                      except OSError:
                           print(f" -> Temp dir {wav_path_temp.parent} not empty, skipping rmdir")
                 else:
                      print(" -> Output was not in system temp dir, skipping cleanup.")

            # Test with specific output directory
            test_output_dir = pathlib.Path("./ffmpeg_test_output")
            wav_path_specific = convert_to_wav(dummy_input, output_dir=test_output_dir)
            print(f" -> Specific dir WAV path: {wav_path_specific}")
            if wav_path_specific.exists():
                 print(f" -> Cleaning up specific file: {wav_path_specific}")
                 wav_path_specific.unlink()
            if test_output_dir.exists():
                 print(f" -> Cleaning up specific dir: {test_output_dir}")
                 test_output_dir.rmdir()

        except (RuntimeError, FileNotFoundError, ValueError) as e:
            print(f"Conversion test expectedly failed or encountered error: {e}")
        finally:
            dummy_input.unlink() # Clean up dummy input
            print("\nFFmpeg util test finished.")
