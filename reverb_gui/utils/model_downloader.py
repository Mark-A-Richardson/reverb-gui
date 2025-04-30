"""Handles checking for and downloading required Hugging Face models."""

import os
import pathlib
from typing import Optional

import dotenv
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

# Environment variable names
ENV_MODELS_DIR = "REVERB_MODELS_DIR"
ENV_HF_TOKEN = "HUGGING_FACE_HUB_TOKEN"

# Default models directory relative to project root
DEFAULT_MODELS_DIR_NAME = "models"

# Required models (Corrected based on user feedback and project code)
REQUIRED_MODELS = [
    "Revai/reverb-asr",
    "Revai/reverb-diarization-v2",
    # Add other models here if needed
]

# --- Helper Functions ---

def _get_project_root() -> pathlib.Path:
    """Finds the project root directory (assuming it contains pyproject.toml)."""
    current_path = pathlib.Path(__file__).resolve()
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists():
            return current_path
        current_path = current_path.parent
    # Fallback if pyproject.toml is not found (should not happen in normal setup)
    return pathlib.Path.cwd()

def get_models_dir() -> pathlib.Path:
    """Gets the target directory for storing models.

    Reads from the REVERB_MODELS_DIR environment variable.
    If not set or empty, defaults to './models' relative to the project root.
    Creates the directory if it doesn't exist.

    Returns:
        An absolute pathlib.Path to the models directory.
    """
    dotenv.load_dotenv() # Load .env file if present
    models_dir_env = os.getenv(ENV_MODELS_DIR)

    if models_dir_env:
        models_path = pathlib.Path(models_dir_env)
    else:
        project_root = _get_project_root()
        models_path = project_root / DEFAULT_MODELS_DIR_NAME

    # Ensure the path is absolute
    if not models_path.is_absolute():
        project_root = _get_project_root()
        models_path = (project_root / models_path).resolve()

    # Create the directory if it doesn't exist
    models_path.mkdir(parents=True, exist_ok=True)
    print(f"Using models directory: {models_path}")
    return models_path

def _get_hf_token() -> Optional[str]:
    """Gets the Hugging Face Hub token from the environment variable."""
    dotenv.load_dotenv()
    token = os.getenv(ENV_HF_TOKEN)
    return token if token else None

def check_model_exists(model_id: str, models_dir: pathlib.Path, hf_token: Optional[str]) -> bool:
    """Checks if a model exists locally using huggingface_hub.

    Args:
        model_id: The Hugging Face model ID (e.g., 'openai/whisper-large-v3').
        models_dir: The local directory where models are stored.
        hf_token: Optional Hugging Face API token.

    Returns:
        True if the model seems complete locally, False otherwise.
    """
    try:
        # snapshot_download with local_files_only=True will check the cache
        # integrity and raise LocalEntryNotFoundError if incomplete or missing.
        snapshot_download(
            repo_id=model_id,
            cache_dir=models_dir, # Use our specific models dir as cache
            local_files_only=True,
            token=hf_token,
            repo_type="model", # Explicitly state repo type
        )
        print(f"Model '{model_id}' found locally.")
        return True
    except LocalEntryNotFoundError:
        print(f"Model '{model_id}' not found locally or incomplete.")
        return False
    except Exception as e:
        # Catch other potential issues during the check
        print(f"Error checking local model '{model_id}': {e}")
        return False

def download_model(
    model_id: str,
    models_dir: pathlib.Path,
    hf_token: Optional[str]
) -> bool:
    """Downloads a model from Hugging Face Hub.

    Args:
        model_id: The Hugging Face model ID.
        models_dir: The directory to download the model to.
        hf_token: Optional Hugging Face API token.

    Returns:
        True if download was successful, False otherwise.
    """
    print(f"Attempting to download model '{model_id}'...")
    # TODO: Integrate progress reporting with GUI later
    try:
        # Use ignore_patterns common for Whisper to potentially avoid issues if needed
        # Might not be necessary for RevAI models but keep as reference
        ignore_patterns = []
        # if "whisper" in model_id.lower():
        #     ignore_patterns = ["*.safetensors", "*.fp16.bin"] # Example patterns

        snapshot_download(
            repo_id=model_id,
            cache_dir=models_dir, # Use our specific models dir as cache
            resume_download=True,
            token=hf_token,
            repo_type="model",
            ignore_patterns=ignore_patterns if ignore_patterns else None,
        )
        print(f"Model '{model_id}' downloaded successfully to {models_dir}.")
        return True
    except HfHubHTTPError as e:
        print(f"HTTP Error downloading '{model_id}': {e}")
        if "authentication is required" in str(e).lower():
            print("  -> This model may require a Hugging Face Hub token.")
            print("  -> Please ensure HUGGING_FACE_HUB_TOKEN is set correctly in your .env file.")
        elif "Invalid username or password" in str(e):
             print("  -> Invalid Hugging Face Hub token provided.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred downloading '{model_id}': {e}")
        return False

# --- Main Function ---

def ensure_models_are_downloaded() -> Optional[pathlib.Path]:
    """Checks for all required models and downloads them if missing.

    Returns:
        The absolute path to the models directory if all models are present
        or were successfully downloaded, None otherwise.
    """
    print("\n--- Checking/Downloading Required Models ---")
    models_dir = get_models_dir()
    hf_token = _get_hf_token()
    all_models_present = True

    for model_id in REQUIRED_MODELS:
        print(f"\nChecking for model: {model_id}")
        if not check_model_exists(model_id, models_dir, hf_token):
            if not download_model(model_id, models_dir, hf_token):
                all_models_present = False
                print(f"Failed to download required model '{model_id}'. Cannot continue.")
                # Optionally, break here if one failure is critical
                # break

    print("\n--- Model Check Complete ---")
    if all_models_present:
        print("All required models are present.")
        return models_dir
    else:
        print("One or more required models could not be downloaded or verified.")
        return None

# Example usage (for testing)
if __name__ == "__main__":
    print("Running model downloader check...")
    final_models_dir = ensure_models_are_downloaded()
    if final_models_dir:
        print(f"\nModel check/download successful. Models are in: {final_models_dir}")
    else:
        print("\nModel check/download failed.")
