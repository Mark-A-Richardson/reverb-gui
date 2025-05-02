import pytest
import pathlib
import subprocess
from unittest import mock
import os

# Module to test
from reverb_gui.utils import ffmpeg


# --- Tests for check_ffmpeg_availability --- #

@mock.patch('reverb_gui.utils.ffmpeg.shutil.which') # Patch within the module namespace
def test_check_ffmpeg_availability_found(mock_which: mock.MagicMock) -> None:
    """Test check_ffmpeg_availability when ffmpeg is found."""
    # Arrange
    mock_which.return_value = '/usr/bin/ffmpeg' # Simulate ffmpeg found

    # Act
    result = ffmpeg.check_ffmpeg_availability()

    # Assert
    assert result is True
    assert mock_which.call_count == 1 # Only called once to check existence
    mock_which.assert_called_with(ffmpeg.FFMPEG_CMD) # Check args of the last call (same as first)


@mock.patch('reverb_gui.utils.ffmpeg.shutil.which') # Patch within the module namespace
def test_check_ffmpeg_availability_not_found(mock_which: mock.MagicMock) -> None:
    """Test check_ffmpeg_availability when ffmpeg is NOT found."""
    # Arrange
    mock_which.return_value = None # Simulate ffmpeg not found

    # Act
    result = ffmpeg.check_ffmpeg_availability()

    # Assert
    assert result is False
    mock_which.assert_called_once_with(ffmpeg.FFMPEG_CMD)


# --- Tests for convert_to_wav --- #

@mock.patch('reverb_gui.utils.ffmpeg.subprocess.run')
@mock.patch('reverb_gui.utils.ffmpeg.tempfile.mkdtemp')
@mock.patch('reverb_gui.utils.ffmpeg.check_ffmpeg_availability')
def test_convert_to_wav_success_temp_dir(
    mock_check_ffmpeg: mock.MagicMock,
    mock_mkdtemp: mock.MagicMock,
    mock_subprocess_run: mock.MagicMock
) -> None:
    """Test convert_to_wav successful execution using a temporary directory."""
    # Arrange
    mock_check_ffmpeg.return_value = True # FFmpeg is available

    # Mock input path object manually
    mock_input_path_instance = mock.MagicMock(spec=pathlib.Path)
    mock_input_path_instance.is_file.return_value = True
    mock_input_path_instance.stem = "test_audio"

    # Mock tempfile.mkdtemp
    fake_temp_dir_str = "/fake/temp/dir"
    mock_mkdtemp.return_value = fake_temp_dir_str

    # Mock the Path objects that will be created inside the function
    mock_output_dir_instance = mock.MagicMock(spec=pathlib.Path)
    mock_output_wav_instance = mock.MagicMock(spec=pathlib.Path)

    # Mock subprocess success
    mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    # Use context manager to mock Path constructor specifically for the temp dir path
    with mock.patch('reverb_gui.utils.ffmpeg.pathlib.Path') as mock_path_cls_local:

        def path_constructor_side_effect(*args, **kwargs):
            # This will be called like Path("/fake/temp/dir")
            if args == (fake_temp_dir_str,):
                return mock_output_dir_instance
            # We pass the mocked input instance directly, so Path() shouldn't be called for it.
            # Raise error for any other unexpected Path() calls
            raise ValueError(f"Unexpected Path constructor call: {args}")

        mock_path_cls_local.side_effect = path_constructor_side_effect
        # Configure the mock returned by Path(fake_temp_dir_str)
        mock_output_dir_instance.__truediv__.return_value = mock_output_wav_instance

        # Act
        result_path = ffmpeg.convert_to_wav(mock_input_path_instance) # Pass the mocked input instance

    # Assert
    mock_check_ffmpeg.assert_called_once()
    mock_input_path_instance.is_file.assert_called_once()
    mock_mkdtemp.assert_called_once_with(prefix="reverb_gui_ffmpeg_")

    # Assert that Path("/fake/temp/dir") was called
    mock_path_cls_local.assert_called_once_with(fake_temp_dir_str)
    # Assert that the '/' operator was used to create the output wav path
    mock_output_dir_instance.__truediv__.assert_called_once_with(
        "test_audio_converted_16000hz_16bit_mono.wav"
    )
    assert result_path == mock_output_wav_instance # Ensure the final mocked path is returned

    # Assert subprocess call
    expected_cmd = [
        ffmpeg.FFMPEG_CMD,
        "-i", str(mock_input_path_instance), # Use the mocked instance
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        "-loglevel", "error",
        str(mock_output_wav_instance) # Final output path mock
    ]
    mock_subprocess_run.assert_called_once_with(
        expected_cmd, check=True, capture_output=True, text=True, shell=False
    )


@mock.patch('reverb_gui.utils.ffmpeg.subprocess.run')
# No mkdtemp mock needed here
@mock.patch('reverb_gui.utils.ffmpeg.check_ffmpeg_availability')
def test_convert_to_wav_success_specific_dir(
    mock_check_ffmpeg: mock.MagicMock,
    mock_subprocess_run: mock.MagicMock
) -> None:
    """Test convert_to_wav successful execution using a specific output directory."""
    # Arrange
    mock_check_ffmpeg.return_value = True # FFmpeg is available

    # Mock input path object manually
    mock_input_path_instance = mock.MagicMock(spec=pathlib.Path)
    mock_input_path_instance.is_file.return_value = True
    mock_input_path_instance.stem = "test_specific"

    # Mock the specific output directory path object passed as argument
    mock_specific_output_dir_arg = mock.MagicMock(spec=pathlib.Path)

    # Mock the Path objects that will be created inside the function
    mock_output_dir_instance_internal = mock.MagicMock(spec=pathlib.Path)
    mock_output_wav_instance = mock.MagicMock(spec=pathlib.Path)

    # Mock subprocess success
    mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    # Use context manager to mock Path constructor
    with mock.patch('reverb_gui.utils.ffmpeg.pathlib.Path') as mock_path_cls_local:

        def path_side_effect(*args, **kwargs):
            # Path(output_dir) where output_dir is the argument
            if args == (mock_specific_output_dir_arg,):
                return mock_output_dir_instance_internal
            # Other calls are not expected
            raise ValueError(f"Unexpected Path constructor call: {args}")

        mock_path_cls_local.side_effect = path_side_effect

        # Configure the mock created internally from the specific dir argument
        mock_output_dir_instance_internal.mkdir.return_value = None
        mock_output_dir_instance_internal.__truediv__.return_value = mock_output_wav_instance

        # Act
        result_path = ffmpeg.convert_to_wav(mock_input_path_instance, output_dir=mock_specific_output_dir_arg)

    # Assert
    mock_check_ffmpeg.assert_called_once()
    mock_input_path_instance.is_file.assert_called_once()
    mock_path_cls_local.assert_called_once_with(mock_specific_output_dir_arg) # Path(output_dir) called
    mock_output_dir_instance_internal.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_output_dir_instance_internal.__truediv__.assert_called_once_with(
        "test_specific_converted_16000hz_16bit_mono.wav"
    )
    assert result_path == mock_output_wav_instance

    # Assert subprocess call (uses the final mock wav path)
    expected_cmd = [
        ffmpeg.FFMPEG_CMD,
        "-i", str(mock_input_path_instance),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        "-loglevel", "error",
        str(mock_output_wav_instance)
    ]
    mock_subprocess_run.assert_called_once_with(
        expected_cmd, check=True, capture_output=True, text=True, shell=False
    )


@mock.patch('reverb_gui.utils.ffmpeg.check_ffmpeg_availability')
def test_convert_to_wav_fail_ffmpeg_not_found(
    mock_check_ffmpeg: mock.MagicMock,
) -> None:
    """Test convert_to_wav raises FileNotFoundError when FFmpeg is not available."""
    # Arrange
    mock_check_ffmpeg.return_value = False # FFmpeg is NOT available
    mock_input_path = mock.MagicMock(spec=pathlib.Path) # Dummy input path

    # Act & Assert
    with pytest.raises(FileNotFoundError) as excinfo: # Expect FileNotFoundError
        ffmpeg.convert_to_wav(mock_input_path)

    # Assert correct exception message
    assert "not found or not executable" in str(excinfo.value) # Check new message substring

    mock_check_ffmpeg.assert_called_once() # Ensure the check was performed


@mock.patch('reverb_gui.utils.ffmpeg.check_ffmpeg_availability')
def test_convert_to_wav_fail_input_not_found(
    mock_check_ffmpeg: mock.MagicMock,
) -> None:
    """Test convert_to_wav raises FileNotFoundError when input file does not exist."""
    # Arrange
    mock_check_ffmpeg.return_value = True # FFmpeg is available

    mock_input_path = mock.MagicMock(spec=pathlib.Path)
    mock_input_path.is_file.return_value = False # Input file does *not* exist
    mock_input_path.__str__.return_value = "/fake/nonexistent/input.mp3" # For error message

    # Act & Assert
    with pytest.raises(FileNotFoundError) as excinfo:
        ffmpeg.convert_to_wav(mock_input_path)

    # Assert correct exception message
    assert f"Input file not found: {mock_input_path}" in str(excinfo.value)
    mock_check_ffmpeg.assert_called_once() # Ensure ffmpeg check happened
    mock_input_path.is_file.assert_called_once() # Ensure file check happened


@mock.patch('reverb_gui.utils.ffmpeg.subprocess.run')
@mock.patch('reverb_gui.utils.ffmpeg.tempfile.mkdtemp')
@mock.patch('reverb_gui.utils.ffmpeg.check_ffmpeg_availability')
def test_convert_to_wav_fail_ffmpeg_error(
    mock_check_ffmpeg: mock.MagicMock,
    mock_mkdtemp: mock.MagicMock,
    mock_subprocess_run: mock.MagicMock
) -> None:
    """Test convert_to_wav raises RuntimeError when the FFmpeg command fails."""
    # Arrange
    mock_check_ffmpeg.return_value = True # FFmpeg is available

    # Mock input path object manually
    mock_input_path_instance = mock.MagicMock(spec=pathlib.Path)
    mock_input_path_instance.is_file.return_value = True
    mock_input_path_instance.stem = "test_error"
    mock_input_path_instance.__str__.return_value = "/fake/input/test_error.mp3"

    # Mock tempfile.mkdtemp
    fake_temp_dir_str = "/fake/temp/error_dir"
    mock_mkdtemp.return_value = fake_temp_dir_str

    # Mock the Path objects that will be created inside the function
    mock_output_dir_instance = mock.MagicMock(spec=pathlib.Path)
    mock_output_wav_instance = mock.MagicMock(spec=pathlib.Path)
    mock_output_wav_instance.__str__.return_value = "/fake/temp/error_dir/output.wav" # For error message

    # Mock subprocess to raise CalledProcessError
    error_message = "ffmpeg failed with error"
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd=["ffmpeg", "..."], stderr=error_message
    )

    # Use context manager to mock Path constructor
    with mock.patch('reverb_gui.utils.ffmpeg.pathlib.Path') as mock_path_cls_local:
        def path_constructor_side_effect(*args, **kwargs):
            if args == (fake_temp_dir_str,):
                return mock_output_dir_instance
            raise ValueError(f"Unexpected Path constructor call: {args}")

        mock_path_cls_local.side_effect = path_constructor_side_effect
        mock_output_dir_instance.__truediv__.return_value = mock_output_wav_instance

        # Act & Assert
        with pytest.raises(RuntimeError) as excinfo:
            ffmpeg.convert_to_wav(mock_input_path_instance)

    # Assert correct exception message
    assert "FFmpeg conversion failed" in str(excinfo.value) # Check for new message start
    assert error_message in str(excinfo.value) # Ensure original stderr is included

    # Assert mocks were called as expected up to the failure point
    mock_check_ffmpeg.assert_called_once()
    mock_input_path_instance.is_file.assert_called_once()
    mock_mkdtemp.assert_called_once()
    mock_path_cls_local.assert_called_once_with(fake_temp_dir_str)
    mock_output_dir_instance.__truediv__.assert_called_once()
    mock_subprocess_run.assert_called_once() # Ensure ffmpeg was attempted


@mock.patch('reverb_gui.utils.ffmpeg.check_ffmpeg_availability')
def test_convert_to_wav_fail_invalid_bit_depth(
    mock_check_ffmpeg: mock.MagicMock,
) -> None:
    """Test convert_to_wav raises ValueError for invalid bit_depth."""
    # Arrange
    mock_check_ffmpeg.return_value = True # FFmpeg is available

    mock_input_path = mock.MagicMock(spec=pathlib.Path)
    mock_input_path.is_file.return_value = True # Input file exists
    mock_input_path.stem = "test_bit_depth"

    invalid_bit_depth = 15

    # Act & Assert
    with pytest.raises(ValueError) as excinfo:
        ffmpeg.convert_to_wav(mock_input_path, bit_depth=invalid_bit_depth)

    # Assert correct exception message
    assert f"Unsupported bit depth: {invalid_bit_depth}" in str(excinfo.value) # Check actual message
    assert "Must be 16, 24, or 32" in str(excinfo.value) # Check supported depths
    mock_check_ffmpeg.assert_called_once() # Ensure ffmpeg check happened
    mock_input_path.is_file.assert_called_once() # Ensure file check happened


# Test that FFMPEG_CMD is correctly determined from environment variable
@pytest.mark.parametrize(
    "env_path, expected_cmd",
    [
        (None, "ffmpeg"), # Default when not set
        ("", "ffmpeg"), # Default when empty
        ("/usr/bin/ffmpeg", os.path.normpath("/usr/bin/ffmpeg")), # Unix-style path
        ("C:/ffmpeg/bin/ffmpeg.exe", os.path.normpath('C:/ffmpeg/bin/ffmpeg.exe')), # Normalized
        ("C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe", os.path.normpath('C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe')), # Normalized
        ('"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"', os.path.normpath('C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe')), # Normalized from quoted
        ("ffmpeg_custom_name", "ffmpeg_custom_name"), # Just a name, expect passthrough (not a path)
    ]
)
def test_ffmpeg_cmd_from_env(monkeypatch, env_path, expected_cmd):
    """Verify FFMPEG_CMD uses env var FFMPEG_PATH correctly, normalizing paths."""
    if env_path is not None:
        monkeypatch.setenv("FFMPEG_PATH", env_path)
    else:
        monkeypatch.delenv("FFMPEG_PATH", raising=False) # Ensure it's not set

    # Reload the ffmpeg module to re-evaluate FFMPEG_CMD with the new env var
    import importlib
    importlib.reload(ffmpeg)

    assert ffmpeg.FFMPEG_CMD == expected_cmd
