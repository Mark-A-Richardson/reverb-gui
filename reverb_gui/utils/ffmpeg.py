"""Utilities for interacting with the FFmpeg executable."""

import shutil
import subprocess
import pathlib
import tempfile
from typing import Optional

FFMPEG_CMD = "ffmpeg" # Default command name, assumes it's in PATH
# TODO: Make this configurable via settings/env

def check_ffmpeg_availability() -> bool:
    """Checks if the ffmpeg command is available in the system PATH."""
    if shutil.which(FFMPEG_CMD):
        print(f"FFmpeg found at: {shutil.which(FFMPEG_CMD)}")
        return True
    else:
        print(f"Error: '{FFMPEG_CMD}' command not found in PATH.")
        print("Please ensure FFmpeg is installed and added to your system's PATH.")
        # In the future, we might check a bundled location too.
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
        raise FileNotFoundError(f"'{FFMPEG_CMD}' command not found. Cannot convert audio.")

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
    output_path = output_dir_path / output_filename

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
        "-i", str(input_path),
        "-vn",
        "-acodec", pcm_codec,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-y",
        "-loglevel", "error", # Keep output clean unless error
        # TODO: Add '-progress pipe:1' for progress reporting later
        str(output_path)
    ]

    print(f"Running FFmpeg command: {' '.join(command)}")
    try:
        # Using subprocess.run for simplicity; use Popen for progress later
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"FFmpeg conversion successful: {output_path}")
        # print(f"FFmpeg stdout:\n{result.stdout}") # Usually empty with -loglevel error
        # print(f"FFmpeg stderr:\n{result.stderr}") # Contains errors if any
        return output_path
    except subprocess.CalledProcessError as e:
        error_message = (
            f"FFmpeg conversion failed with exit code {e.returncode}.\n"
            f"Command: {' '.join(e.cmd)}\n"
            f"Stderr:\n{e.stderr}"
        )
        print(error_message)
        # Clean up failed output file if it exists
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError as unlink_err:
                 print(f"Warning: Could not delete incomplete output file {output_path}: {unlink_err}")
        raise RuntimeError(error_message) from e
    except FileNotFoundError as e:
         # This typically means FFMPEG_CMD wasn't found despite check_ffmpeg_availability
         # Should not happen if check passes, but handle defensively.
         print(f"Error running subprocess: {e}")
         raise FileNotFoundError(f"'{FFMPEG_CMD}' command failed to execute. Is it installed correctly?") from e

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
