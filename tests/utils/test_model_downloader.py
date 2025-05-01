import os
from unittest import mock
import pathlib # Import pathlib to mock Path

import pytest # Import pytest for testing
from huggingface_hub.utils import LocalEntryNotFoundError, HfHubHTTPError # Import exceptions

# Module to test
from reverb_gui.utils import model_downloader


@pytest.mark.parametrize(
    "env_token, expected_token",
    [
        ("test_token_123", "test_token_123"), # Token exists
        (None, None),                      # Token does not exist
        ("", None),                       # Token is empty string (should be treated as None)
    ]
)
@mock.patch.dict(os.environ, {}, clear=True) # Start with clean environment for each test case
@mock.patch('dotenv.load_dotenv') # Mock load_dotenv to prevent loading .env files
def test_get_hf_token(mock_load_dotenv, env_token: str | None, expected_token: str | None, monkeypatch) -> None:
    """Test the _get_hf_token function retrieves token from env var."""
    # mock_load_dotenv is the MagicMock object for the patched function, but we don't need to interact with it here.
    if env_token is not None:
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", env_token)
    else:
        # Ensure it's not set if the test case expects None
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    token = model_downloader._get_hf_token()
    assert token == expected_token


@mock.patch('pathlib.Path.mkdir')
@mock.patch('reverb_gui.utils.model_downloader._get_project_root')
@mock.patch('dotenv.load_dotenv')
def test_get_models_dir_default(mock_load_dotenv: mock.MagicMock,
                                mock_get_project_root: mock.MagicMock,
                                mock_mkdir: mock.MagicMock) -> None:
    """Test get_models_dir default behavior (REVERB_MODELS_DIR unset)."""
    # Arrange: Configure mock for _get_project_root to return a Windows-absolute path
    fake_project_root = pathlib.Path('C:/fake/project/root')
    mock_get_project_root.return_value = fake_project_root

    # Arrange: Ensure environment variable is unset (although load_dotenv is mocked anyway)
    # We can use monkeypatch here if needed, but mocking load_dotenv should be sufficient.

    # Act: Call the function under test
    models_dir = model_downloader.get_models_dir()

    # Assert: Check the returned path is correct (project_root / 'models')
    expected_path = fake_project_root / model_downloader.DEFAULT_MODELS_DIR_NAME
    assert models_dir == expected_path

    # Assert: Check that mkdir was called correctly on the expected path object
    models_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    # Assert: Check the specific mock instance was called
    # Note: Since models_dir is the path object returned, mock_mkdir might not be
    # directly comparable if the path object is recreated internally. Checking
    # models_dir.mkdir is the more reliable check here.
    # mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@mock.patch('pathlib.Path.mkdir')
@mock.patch('dotenv.load_dotenv')
def test_get_models_dir_env_var_set(mock_load_dotenv: mock.MagicMock,
                                    mock_mkdir: mock.MagicMock,
                                    monkeypatch) -> None:
    """Test get_models_dir when REVERB_MODELS_DIR environment variable is set."""
    # Arrange: Set the environment variable
    test_env_path_str = "C:/custom/model/dir/from/env"
    monkeypatch.setenv(model_downloader.ENV_MODELS_DIR, test_env_path_str)
    expected_path = pathlib.Path(test_env_path_str)

    # Act: Call the function under test
    models_dir = model_downloader.get_models_dir()

    # Assert: Check the returned path matches the environment variable path
    assert models_dir == expected_path

    # Assert: Check that mkdir was called correctly on the expected path object
    models_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    # Assert: load_dotenv was called (to ensure it tries reading .env)
    mock_load_dotenv.assert_called_once()


# --- Tests for ensure_models_are_downloaded ---

@mock.patch('huggingface_hub.snapshot_download') # Mock the actual download function
@mock.patch('reverb_gui.utils.model_downloader.check_model_exists') # Mock the helper function
@mock.patch('reverb_gui.utils.model_downloader._get_hf_token') # Mock getting token
@mock.patch('reverb_gui.utils.model_downloader.get_models_dir')  # Mock getting models dir path
def test_ensure_models_downloaded_all_exist(mock_get_models_dir: mock.MagicMock,
                                           mock_get_hf_token: mock.MagicMock,
                                           mock_check_model_exists: mock.MagicMock, # Renamed arg
                                           mock_snapshot_download: mock.MagicMock) -> None:
    """Test ensure_models_are_downloaded when all models exist locally."""
    # Arrange: Configure mocks
    fake_models_path = pathlib.Path('C:/fake/models/dir')
    mock_get_models_dir.return_value = fake_models_path
    mock_get_hf_token.return_value = 'fake_token' # Token needed by check_model_exists signature
    mock_check_model_exists.return_value = True # Simulate the check always passes

    # Act: Call the function under test
    result_path = model_downloader.ensure_models_are_downloaded()

    # Assert: Check the returned path is correct
    assert result_path == fake_models_path

    # Assert: Check required functions were called
    mock_get_models_dir.assert_called_once()
    mock_get_hf_token.assert_called_once() # Called once before loop

    # Assert: Check check_model_exists was called for each required model
    assert mock_check_model_exists.call_count == len(model_downloader.REQUIRED_MODELS)
    for model_id in model_downloader.REQUIRED_MODELS:
        # Assert check_model_exists was called with the correct arguments
        mock_check_model_exists.assert_any_call(model_id, fake_models_path, 'fake_token')

    # Assert: snapshot_download should NOT have been called
    mock_snapshot_download.assert_not_called()


@mock.patch('huggingface_hub.snapshot_download')
@mock.patch('reverb_gui.utils.model_downloader.download_model') # Mock the download helper
@mock.patch('reverb_gui.utils.model_downloader.check_model_exists') # Mock the check helper
@mock.patch('reverb_gui.utils.model_downloader._get_hf_token')
@mock.patch('reverb_gui.utils.model_downloader.get_models_dir')
def test_ensure_models_downloaded_one_missing(
    mock_get_models_dir: mock.MagicMock,
    mock_get_hf_token: mock.MagicMock,
    mock_check_model_exists: mock.MagicMock,
    mock_download_model: mock.MagicMock,
    mock_snapshot_download: mock.MagicMock
) -> None:
    """Test ensure_models_are_downloaded when one model is missing and downloads ok."""
    # Arrange: Setup mocks
    fake_models_path = pathlib.Path('C:/fake/models/dir')
    mock_get_models_dir.return_value = fake_models_path
    fake_token = 'fake_token'
    mock_get_hf_token.return_value = fake_token

    # Simulate first model missing, rest exist
    num_models = len(model_downloader.REQUIRED_MODELS)
    check_side_effects = [False] + [True] * (num_models - 1)
    mock_check_model_exists.side_effect = check_side_effects

    # Simulate successful download
    mock_download_model.return_value = True

    # Act: Call the function
    result_path = model_downloader.ensure_models_are_downloaded()

    # Assert: Return value is correct
    assert result_path == fake_models_path

    # Assert: Correct functions called
    mock_get_models_dir.assert_called_once()
    mock_get_hf_token.assert_called_once()
    assert mock_check_model_exists.call_count == num_models

    # Assert: download_model called once for the first model
    mock_download_model.assert_called_once_with(
        model_downloader.REQUIRED_MODELS[0],
        fake_models_path,
        fake_token
    )

    # Assert: snapshot_download (the low-level func) was not called directly
    mock_snapshot_download.assert_not_called()


@mock.patch('reverb_gui.utils.model_downloader.download_model') # Mock the download helper
@mock.patch('reverb_gui.utils.model_downloader.check_model_exists') # Mock the check helper
@mock.patch('reverb_gui.utils.model_downloader._get_hf_token')
@mock.patch('reverb_gui.utils.model_downloader.get_models_dir')
def test_ensure_models_downloaded_download_fails(
    mock_get_models_dir: mock.MagicMock,
    mock_get_hf_token: mock.MagicMock,
    mock_check_model_exists: mock.MagicMock,
    mock_download_model: mock.MagicMock
) -> None:
    """Test ensure_models_are_downloaded when a model download fails."""
    # Arrange: Setup mocks
    fake_models_path = pathlib.Path('C:/fake/models/dir')
    mock_get_models_dir.return_value = fake_models_path
    fake_token = 'fake_token'
    mock_get_hf_token.return_value = fake_token

    # Simulate first model missing
    mock_check_model_exists.return_value = False

    # Simulate download failure
    mock_download_model.return_value = False

    # Act: Call the function
    result_path = model_downloader.ensure_models_are_downloaded()

    # Assert: Return value is None (failure)
    assert result_path is None

    # Assert: Correct functions called
    mock_get_models_dir.assert_called_once()
    mock_get_hf_token.assert_called_once()

    # Assert: check_model_exists called for the first model (at least)
    mock_check_model_exists.assert_any_call(
        model_downloader.REQUIRED_MODELS[0],
        fake_models_path,
        fake_token
    )
    # Assert check_model_exists was called for all models (since loop doesn't break)
    num_models = len(model_downloader.REQUIRED_MODELS)
    assert mock_check_model_exists.call_count == num_models

    # Assert: download_model called for the first model (at least)
    mock_download_model.assert_any_call(
        model_downloader.REQUIRED_MODELS[0],
        fake_models_path,
        fake_token
    )
    # Assert download_model was called for all models (since check returns False for all)
    assert mock_download_model.call_count == num_models


# --- Tests for check_model_exists --- #

@mock.patch('reverb_gui.utils.model_downloader.snapshot_download') # Corrected target
def test_check_model_exists_success(mock_snapshot_download: mock.MagicMock) -> None:
    """Test check_model_exists when the model is found locally."""
    # Arrange
    test_model_id = "org/model-name"
    fake_models_path = pathlib.Path('C:/fake/models/dir')
    fake_token = 'fake_token'
    # No exception raised by mock means success

    # Act
    result = model_downloader.check_model_exists(test_model_id, fake_models_path, fake_token)

    # Assert
    assert result is True
    mock_snapshot_download.assert_called_once_with(
        repo_id=test_model_id,
        cache_dir=fake_models_path,
        local_files_only=True,
        token=fake_token,
        repo_type="model"
    )

@mock.patch('reverb_gui.utils.model_downloader.snapshot_download') # Corrected target
def test_check_model_exists_not_found(mock_snapshot_download: mock.MagicMock) -> None:
    """Test check_model_exists when the model is not found locally."""
    # Arrange
    test_model_id = "org/model-name"
    fake_models_path = pathlib.Path('C:/fake/models/dir')
    fake_token = 'fake_token'
    mock_snapshot_download.side_effect = LocalEntryNotFoundError("Model not found")

    # Act
    result = model_downloader.check_model_exists(test_model_id, fake_models_path, fake_token)

    # Assert
    assert result is False
    mock_snapshot_download.assert_called_once_with(
        repo_id=test_model_id,
        cache_dir=fake_models_path,
        local_files_only=True,
        token=fake_token,
        repo_type="model"
    )

@mock.patch('reverb_gui.utils.model_downloader.snapshot_download') # Corrected target
def test_check_model_exists_other_error(mock_snapshot_download: mock.MagicMock) -> None:
    """Test check_model_exists when some other error occurs during check."""
    # Arrange
    test_model_id = "org/model-name"
    fake_models_path = pathlib.Path('C:/fake/models/dir')
    fake_token = 'fake_token'
    mock_snapshot_download.side_effect = Exception("Some other error")

    # Act
    result = model_downloader.check_model_exists(test_model_id, fake_models_path, fake_token)

    # Assert
    assert result is False
    mock_snapshot_download.assert_called_once_with(
        repo_id=test_model_id,
        cache_dir=fake_models_path,
        local_files_only=True,
        token=fake_token,
        repo_type="model"
    )

# --- Tests for download_model --- #

@mock.patch('reverb_gui.utils.model_downloader.snapshot_download')
def test_download_model_success(mock_snapshot_download: mock.MagicMock) -> None:
    """Test download_model successful execution."""
    # Arrange
    test_model_id = "org/dl-model-success"
    fake_models_path = pathlib.Path('C:/fake/models/dir')
    fake_token = 'fake_token'
    # No exception means success

    # Act
    result = model_downloader.download_model(test_model_id, fake_models_path, fake_token)

    # Assert
    assert result is True
    mock_snapshot_download.assert_called_once_with(
        repo_id=test_model_id,
        cache_dir=fake_models_path,
        resume_download=True,
        token=fake_token,
        repo_type="model",
        ignore_patterns=None # Default when no specific patterns set
    )

@mock.patch('reverb_gui.utils.model_downloader.snapshot_download')
def test_download_model_http_error(mock_snapshot_download: mock.MagicMock) -> None:
    """Test download_model handling HfHubHTTPError."""
    # Arrange
    test_model_id = "org/dl-model-http-error"
    fake_models_path = pathlib.Path('C:/fake/models/dir')
    fake_token = 'fake_token'
    # Configure the mock response
    mock_response = mock.Mock()
    mock_response.json.return_value = {} # Return an empty dict for .json()
    mock_response.headers = {} # Set headers to an empty dict
    mock_snapshot_download.side_effect = HfHubHTTPError("Auth required", response=mock_response)

    # Act
    result = model_downloader.download_model(test_model_id, fake_models_path, fake_token)

    # Assert
    assert result is False
    mock_snapshot_download.assert_called_once_with(
        repo_id=test_model_id,
        cache_dir=fake_models_path,
        resume_download=True,
        token=fake_token,
        repo_type="model",
        ignore_patterns=None
    )

@mock.patch('reverb_gui.utils.model_downloader.snapshot_download')
def test_download_model_other_error(mock_snapshot_download: mock.MagicMock) -> None:
    """Test download_model handling generic Exception."""
    # Arrange
    test_model_id = "org/dl-model-other-error"
    fake_models_path = pathlib.Path('C:/fake/models/dir')
    fake_token = 'fake_token'
    mock_snapshot_download.side_effect = Exception("Unexpected error")

    # Act
    result = model_downloader.download_model(test_model_id, fake_models_path, fake_token)

    # Assert
    assert result is False
    mock_snapshot_download.assert_called_once_with(
        repo_id=test_model_id,
        cache_dir=fake_models_path,
        resume_download=True,
        token=fake_token,
        repo_type="model",
        ignore_patterns=None
    )


# End of tests for model_downloader.py
