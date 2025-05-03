"""Core pipeline logic combining ASR and Diarization."""

import sys
import time
import traceback
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict

# --- Model & Pipeline Imports ---
import torch
from intervaltree import Interval, IntervalTree
# Assume 'wenet' module is available via PYTHONPATH or similar mechanism
# We need the ReverbASR class and its loader
# from asr.wenet.cli.reverb import ReverbASR, load_model as load_asr_model
from pyannote.audio import Pipeline as DiarizationPipeline
# from pyannote.database.util import load_rttm # Not needed if we use pipeline directly

# --- Utility Imports ---
from ..utils.ffmpeg import convert_to_wav
from ..utils.model_downloader import _get_hf_token
# --- Hugging Face Hub utility for snapshot path ---

# --- Constants ---
ASR_MODEL_ID = "Revai/reverb-asr"
DIARIZATION_MODEL_ID = "Revai/reverb-diarization-v2"

# Add project root to sys.path to find local 'asr' module
# Assumes engine.py is at reverb-gui/reverb_gui/pipeline/engine.py
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
ASR_MODULE_PATH = str(Path(PROJECT_ROOT) / 'asr') # Path to the 'asr' directory
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if ASR_MODULE_PATH not in sys.path:
     sys.path.insert(0, ASR_MODULE_PATH) # Ensure asr module is findable

# Now we can import from the asr module
from wenet.cli.reverb import ReverbASR, load_model as load_asr_model # noqa: E402

# === Global Model Cache (Load Once) ===
# Basic caching to avoid reloading models on every call within the same app run.
# A more robust solution might use a class structure.
_cached_models: Dict[str, Any] = {}
_models_loaded = False
# _models_dir: Optional[pathlib.Path] = None # Let's get it dynamically

def _load_models_if_needed(models_dir: Path) -> None:
    """Loads ASR and Diarization models into cache if not already loaded.

    Args:
        models_dir: The verified path to the directory containing downloaded models.
    """
    global _models_loaded, _cached_models

    if _models_loaded:
        # Even if loaded, confirm device status in case it changed?
        print(f"Engine: Models already loaded (using dir: {models_dir}). Skipping load.")
        return

    print("Engine: Loading models...")
    start_time = time.time()
    try:
        hf_token = _get_hf_token()
        cuda_available = torch.cuda.is_available()
        print(f"Engine: torch.cuda.is_available()? -> {cuda_available}")
        gpu_index = 0 if cuda_available else None
        if cuda_available:
            try:
                print(f"Engine: CUDA Device Name: {torch.cuda.get_device_name(gpu_index)}") # Assumes device 0
            except Exception as e:
                print(f"Engine: Warning - Could not get CUDA device name: {e}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Engine: Using device: {device}")

        # 1. Load Diarization Pipeline
        print(f"Engine: Loading diarization model '{DIARIZATION_MODEL_ID}' from {models_dir}...")
        diar_pipeline = DiarizationPipeline.from_pretrained(
            DIARIZATION_MODEL_ID, use_auth_token=hf_token, cache_dir=str(models_dir)
        )
        diar_pipeline.to(device)
        _cached_models['diarization'] = diar_pipeline
        print("Engine: Diarization model loaded.")

        # 2. Load ASR Model
        print(f"Engine: Loading ASR model '{ASR_MODEL_ID}' from {models_dir}...")
        # Construct path using HF cache naming convention (replace '/' with '--')
        hf_cache_model_dir_name = f"models--{ASR_MODEL_ID.replace('/', '--')}"
        asr_model_path = models_dir / hf_cache_model_dir_name
        print(f"Engine: Passing gpu={gpu_index} and model_path='{asr_model_path}' to ASR model loader.")
        asr_model = load_asr_model(model=str(asr_model_path), gpu=gpu_index)
        print("Engine: ASR model loaded.")

        _cached_models['asr'] = asr_model

        _models_loaded = True
        end_time = time.time()
        print(f"Engine: Models loaded successfully in {end_time - start_time:.2f} seconds.")

    except Exception as e:
        print(f"FATAL: Engine failed to load models: {e}", file=sys.stderr)
        traceback.print_exc()
        # Indicate failure so subsequent calls don't assume models are loaded
        _models_loaded = False
        _cached_models = {}
        raise RuntimeError(f"Failed to load models: {e}") from e


# Helper function adapted from assign_words2speakers.py
def speaker_for_word(start_time: float,
                     duration: float,
                     tree: IntervalTree) -> str:
    """Given a word's start and duration in seconds, and an interval tree representing
    speaker segments, return the speaker label.

    If there are overlapping speakers, return the speaker who spoke most of the
    time. If there are no speakers, return the nearest one.
    """
    intervals = tree[start_time : start_time + duration]

    # Easy case, only one possible interval
    if len(intervals) == 1:
        return intervals.pop().data

    # First special case, no match
    # so we need to find the nearest interval
    elif len(intervals) == 0:
        seg = Interval(start_time, start_time + duration)
        distances = {interval: seg.distance_to(interval)
                     for interval in tree}
        if not distances:
            return "UNKNOWN" # Handle case with no speaker segments at all
        nearest_interval = min(distances, key=distances.get)
        # Optional: Add a distance threshold? If nearest is too far, return UNKNOWN?
        # distance_threshold = 1.0 # seconds
        # if distances[nearest_interval] > distance_threshold:
        #    return "UNKNOWN_FAR"
        return nearest_interval.data

    # Second special case, overlapping speakers
    # so we return whichever speaker has majority
    else:
        seg = Interval(start_time, start_time + duration)
        overlap_sizes = defaultdict(float) # Use float for precision
        for interval in intervals:
            i0 = max(seg.begin, interval.begin)
            i1 = min(seg.end, interval.end)
            overlap = i1 - i0
            if overlap > 0: # Ensure there is actual overlap
                 overlap_sizes[interval.data] += overlap
        if not overlap_sizes:
             # This can happen if the word interval is entirely within a gap
             # between speaker segments. Fallback to nearest.
             # TODO: Maybe log this case?
             distances = {interval: seg.distance_to(interval)
                         for interval in tree}
             if not distances:
                 return "UNKNOWN_GAP"
             return min(distances, key=distances.get).data
        return max(overlap_sizes, key=overlap_sizes.get)


def transcribe(
    input_path: Path,
    models_dir: Path,
    asr_params: dict # Added parameter to accept GUI settings
) -> List[Tuple[float, float, str, str]]:
    """
    Processes an audio/video file using cached Reverb models.
    1. Converts input to WAV.
    2. Runs diarization.
    3. Runs ASR (requesting CTM format).
    4. Parses CTM and assigns speakers based on diarization.
    5. Returns a list of (start_time, end_time, speaker_label, word_text).

    Args:
        input_path: Path to the input audio or video file.
        models_dir: Path to the directory containing downloaded models.
        asr_params: Dictionary containing ASR parameters from the GUI.

    Returns:
        A list of tuples, each containing (start_time, end_time, speaker_label, word_text).
    """
    temp_dir_obj = None # Use TemporaryDirectory for better cleanup
    try:
        # Ensure models are loaded into the cache
        _load_models_if_needed(models_dir=models_dir)
        if not _models_loaded:
             # This shouldn't happen if _load_models_if_needed worked, but belt-and-suspenders
             raise RuntimeError("Models could not be loaded. Cannot proceed.")

        asr_model: ReverbASR = _cached_models['asr']
        diar_pipeline: DiarizationPipeline = _cached_models['diarization']

        # 1. Convert to WAV
        print(f"Engine: Converting {input_path.name} to WAV...")
        conv_start = time.time()
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="reverb_gui_ffmpeg_")
        temp_dir_path = Path(temp_dir_obj.name)
        wav_path = convert_to_wav(input_path, output_dir=temp_dir_path)
        conv_end = time.time()
        print(f"Engine: Conversion took {conv_end - conv_start:.2f}s. Output: {wav_path}")

        # 2. Run Diarization
        print(f"Engine: Running diarization on {wav_path.name}...")
        diar_start = time.time()
        # The diarization pipeline likely returns a pyannote.core.Annotation object
        diarization_result = diar_pipeline(str(wav_path))
        diar_end = time.time()
        print(f"Engine: Diarization took {diar_end - diar_start:.2f}s.")

        # --- Build Speaker Interval Tree ---
        speaker_tree = IntervalTree()
        if diarization_result:
            # Convert pyannote.core.Annotation to IntervalTree
            # Assuming diarization_result.itertracks yields (segment, track_id, speaker_label)
            for segment, _, speaker_label in diarization_result.itertracks(yield_label=True):
                speaker_tree.add(Interval(segment.start, segment.end, speaker_label))
        else:
             print("Warning: Diarization returned no segments.")
             # Assign all words to UNKNOWN if no diarization? Or handle differently?

        # 3. Run ASR (Wenet) - Requesting CTM format
        print(f"Engine: Running ASR on {wav_path.name}...")
        asr_start = time.time()
        # *** IMPORTANT: Calling transcribe with format='ctm' and beam search parameters ***
        asr_result_ctm: str = asr_model.transcribe(
            str(wav_path),
            # --- Parameters passed from GUI --- 
            mode=asr_params.get("mode", "ctc_prefix_beam_search"), # Use provided or default
            beam_size=asr_params.get("beam_size", 10),
            length_penalty=asr_params.get("length_penalty", 0.0),
            ctc_weight=asr_params.get("ctc_weight", 0.1),
            reverse_weight=asr_params.get("reverse_weight", 0.0),
            blank_penalty=asr_params.get("blank_penalty", 0.0),
            verbatimicity=asr_params.get("verbatimicity", 1.0), # Use 1.0 as default if missing
            # --- Fixed parameter for this pipeline --- 
            format="ctm",  # Essential for word timing extraction
            # --- Parameters not exposed in GUI (using defaults) ---
            # chunk_size=2051,
            # batch_size=1,
            # decoding_chunk_size=-1,
            # num_decoding_left_chunks=-1,
            # simulate_streaming=False,
        )
        asr_end = time.time()
        print(f"Engine: ASR took {asr_end - asr_start:.2f}s.")

        # 4. Parse CTM and Assign Speakers
        print("Engine: Assigning words to speakers...")
        combined_transcript: List[Tuple[float, float, str, str]] = []
        if asr_result_ctm:
            # CTM format: <filename> <channel> <start_time> <duration> <word> <confidence?>
            # Example: video_converted_16000hz_16bit_mono 1 0.530 0.250 hello 1.00
            for line in asr_result_ctm.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        start_time = float(parts[2])
                        duration = float(parts[3])
                        word_text = parts[4]
                        end_time = start_time + duration

                        # Find speaker for this word's time interval
                        speaker_label = speaker_for_word(start_time, duration, speaker_tree)

                        combined_transcript.append((start_time, end_time, speaker_label, word_text))
                    except ValueError as e:
                        print(f"Warning: Skipping CTM line due to parsing error: '{line}' - {e}")
                        continue # Skip malformed lines
                    except IndexError as e:
                         print(f"Warning: Skipping CTM line due to index error (likely malformed): '{line}' - {e}")
                         continue
                else:
                     # Log lines that don't have enough parts
                     if line.strip(): # Avoid logging blank lines if any
                         print(f"Warning: Skipping malformed CTM line (too few parts): '{line}'")

        else:
            print("Warning: ASR returned empty CTM.")

        print("Engine: Word assignment complete.")
        return combined_transcript

    except Exception as e:
        print(f"ERROR during transcription pipeline: {type(e).__name__} - {e}", file=sys.stderr)
        traceback.print_exc()
        # Re-raise as a RuntimeError to be caught by the worker/GUI
        raise RuntimeError(f"Pipeline error: {e}") from e
    finally:
        # Cleanup temporary file/directory using the TemporaryDirectory object
        if temp_dir_obj:
            try:
                print(f"Engine: Cleaning up temporary directory: {temp_dir_obj.name}")
                temp_dir_obj.cleanup()
            except Exception as cleanup_error:
                print(f"Error cleaning up temp directory {temp_dir_obj.name}: {cleanup_error}", file=sys.stderr)
