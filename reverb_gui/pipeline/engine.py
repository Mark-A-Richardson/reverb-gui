"""Core pipeline logic combining ASR and Diarization."""

import pathlib
import sys
import tempfile
import time # For basic timing
import traceback # For error reporting in worker
from typing import List, Optional, Dict, Any, Tuple

# --- Model & Pipeline Imports ---
import torch
# Assume 'wenet' module is available via PYTHONPATH or similar mechanism
import wenet # Local ASR module
from pyannote.audio import Pipeline as DiarizationPipeline
# from pyannote.database.util import load_rttm # Not needed if we use pipeline directly

# --- Utility Imports ---
from ..utils.ffmpeg import convert_to_wav
from ..utils.model_downloader import get_models_dir, _get_hf_token
# --- Hugging Face Hub utility for snapshot path ---
from huggingface_hub import snapshot_download

# --- Constants ---
ASR_MODEL_ID = "Revai/reverb-asr"
DIARIZATION_MODEL_ID = "Revai/reverb-diarization-v2"

# Add project root to sys.path to find local 'wenet'
# Assumes engine.py is at reverb-gui/reverb_gui/pipeline/engine.py
PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === Global Model Cache (Load Once) ===
# Basic caching to avoid reloading models on every call within the same app run.
# A more robust solution might use a class structure.
_cached_models: Dict[str, Any] = {}
_models_loaded = False

def _load_models_if_needed() -> None:
    """Loads ASR and Diarization models into cache if not already loaded."""
    global _models_loaded, _cached_models
    if _models_loaded:
        # Even if loaded, confirm device status in case it changed?
        print(f"Engine: Models already loaded. Device check: torch.cuda.is_available() -> {torch.cuda.is_available()}")
        return

    print("Engine: Loading models...")
    start_time = time.time()
    try:
        models_dir = get_models_dir()
        hf_token = _get_hf_token()
        cuda_available = torch.cuda.is_available()
        print(f"Engine: torch.cuda.is_available()? -> {cuda_available}")
        if cuda_available:
            try:
                print(f"Engine: CUDA Device Name: {torch.cuda.get_device_name(0)}") # Assumes device 0
            except Exception as e:
                print(f"Engine: Warning - Could not get CUDA device name: {e}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Engine: Using device: {device}")

        # 1. Load Diarization Pipeline
        print(f"Engine: Loading diarization model '{DIARIZATION_MODEL_ID}'...")
        # Use models_dir as cache_dir for pyannote if possible, or let it use default HF cache
        # Pyannote uses HF_HOME or default ~/.cache/huggingface/hub
        # Setting HF_HOME might be better if we want it within our models_dir
        # For now, let pyannote manage its cache based on HF defaults/HF_HOME
        diarization_pipeline = DiarizationPipeline.from_pretrained(
            DIARIZATION_MODEL_ID,
            use_auth_token=hf_token
        )
        diarization_pipeline.to(device)
        _cached_models['diarization'] = diarization_pipeline
        print("Engine: Diarization model loaded.")

        # 2. Load ASR Model (Wenet)
        print(f"Engine: Loading ASR model '{ASR_MODEL_ID}'...")
        # We need the specific path within models_dir where wenet expects its files.
        # Use snapshot_download to get the path to the downloaded model files.
        asr_gpu_arg: Optional[int] = 0 if device.type == 'cuda' else None
        print(f"Engine: Passing gpu={asr_gpu_arg} to ASR model loader.")
        asr_model_path = snapshot_download(
            ASR_MODEL_ID,
            cache_dir=models_dir, # Use our designated models dir
            token=hf_token,
            local_files_only=True, # Assume already downloaded by main.py check
            # repo_type="model" # Default is model
        )
        print(f"Engine: Found ASR model snapshot at: {asr_model_path}")

        # Load Wenet model - it doesn't seem to accept device arguments directly.
        # It might respect the global device context or default to CPU if CUDA not compiled in?
        # For now, load with default arguments.
        asr_model = wenet.load_model(asr_model_path, gpu=asr_gpu_arg)

        _cached_models['asr'] = asr_model
        print("Engine: ASR model loaded.")

        _models_loaded = True
        end_time = time.time()
        print(f"Engine: Models loaded in {end_time - start_time:.2f} seconds.")

    except ImportError as e:
         print(f"FATAL: Engine failed to import required model library: {e}")
         print("Ensure 'wenet' directory is accessible (added project root to path) and dependencies (torch, pyannote) are installed.")
         traceback.print_exc()
         # Cannot recover if imports fail
         raise
    except Exception as e:
        print(f"FATAL: Engine failed to load models: {e}")
        traceback.print_exc()
        _models_loaded = False # Ensure we retry if loading failed
        # Also check for huggingface_hub errors specifically during snapshot download
        if "snapshot_download" in str(e):
            print("Engine: Hint - Check if models were downloaded correctly to the cache directory.")
        # Propagate error to let the worker signal it
        raise RuntimeError(f"Failed to load models: {e}") from e

# === Main Transcription Function ===

def transcribe(input_path: pathlib.Path) -> Tuple[str, List[Tuple[float, float, str]]]:
    """Processes an audio/video file through FFmpeg, Diarization, and ASR.

    Converts input to WAV, runs diarization, runs ASR.
    Returns the full ASR transcript and speaker segments from diarization.

    Args:
        input_path: Path to the input audio or video file.

    Returns:
        A tuple containing:
            - str: The full transcript text from ASR.
            - List[Tuple[float, float, str]]: Speaker segments as
              (start_time, end_time, speaker_label).

    Raises:
        FileNotFoundError: If input_path, ffmpeg, or required models don't exist.
        RuntimeError: If FFmpeg, model loading, or inference fails.
        ValueError: If FFmpeg arguments are invalid.
        ImportError: If model libraries cannot be imported.
    """
    start_process_time = time.time()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Ensure models are loaded (call handles caching check)
    _load_models_if_needed()
    if not _models_loaded or 'asr' not in _cached_models or 'diarization' not in _cached_models:
         # This should ideally be caught by _load_models_if_needed raising an error
         raise RuntimeError("Models were not loaded successfully.")

    asr_model = _cached_models['asr']
    diarization_pipeline = _cached_models['diarization']

    wav_path: Optional[pathlib.Path] = None
    temp_dir_path: Optional[pathlib.Path] = None
    was_temporary: bool = False

    try:
        # 1. Convert to WAV
        print(f"Engine: Converting {input_path.name} to WAV...")
        conv_start = time.time()
        wav_path = convert_to_wav(input_path)
        conv_end = time.time()
        print(f"Engine: Conversion took {conv_end - conv_start:.2f}s. Output: {wav_path}")

        # Check if temporary (for cleanup)
        try:
            temp_dir = tempfile.gettempdir()
            if str(wav_path.parent).startswith(temp_dir) and \
               wav_path.parent.name.startswith("reverb_gui_ffmpeg_"):
                 was_temporary = True
                 temp_dir_path = wav_path.parent
        except Exception as path_err:
            print(f"Warning: Could not determine if path {wav_path} is temporary: {path_err}")
            was_temporary = False

        # 2. Run Diarization
        print(f"Engine: Running diarization on {wav_path.name}...")
        diar_start = time.time()
        # Pyannote pipeline expects path or dict {'uri': '...', 'audio': path}
        annotation = diarization_pipeline(str(wav_path))
        diar_end = time.time()
        print(f"Engine: Diarization took {diar_end - diar_start:.2f}s.")

        # 3. Run ASR (Wenet)
        print(f"Engine: Running ASR on {wav_path.name}...")
        asr_start = time.time()
        # *** IMPORTANT: Assuming wenet transcribe just returns a string ***
        # Check wenet docs/code for exact output format if different
        asr_result: str = asr_model.transcribe(str(wav_path))
        asr_end = time.time()
        print(f"Engine: ASR took {asr_end - asr_start:.2f}s.")

        # 4. Extract Diarization Segments
        print("Engine: Extracting diarization segments...")
        extract_start = time.time()
        diarization_segments: List[Tuple[float, float, str]] = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            diarization_segments.append((segment.start, segment.end, label))
        extract_end = time.time()
        print(f"Engine: Segment extraction took {extract_end - extract_start:.2f}s.")

        end_process_time = time.time()
        print(f"Engine: Total processing time: {end_process_time - start_process_time:.2f}s")

        # Return the raw ASR text and the list of speaker segments
        return asr_result, diarization_segments

    except Exception as e:
         # Catch errors during inference or processing
         print(f"ERROR during transcription pipeline: {e}")
         traceback.print_exc()
         # Re-raise to allow the worker to catch and signal the error
         raise RuntimeError(f"Pipeline error: {e}") from e

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
