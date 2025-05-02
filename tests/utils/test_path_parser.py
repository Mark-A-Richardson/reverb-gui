import os
import pytest
from pathlib import Path

from reverb_gui.utils.path_parser import parse_env_path

# Define platform-specific expected outputs where needed
IS_WINDOWS = os.name == 'nt'
HOME_DIR = Path.home()

# Test cases
# Format: (input_raw_string, expected_path_string_or_none, test_env_vars)
# expected_path_string uses os.sep for platform independence where appropriate
TEST_CASES = [
    # --- Basic cases ---
    (None, None, {}),
    ("", None, {}),
    ("   ", None, {}),
    ("ffmpeg", "ffmpeg", {}),
    ("/usr/bin/ffmpeg", os.path.normpath("/usr/bin/ffmpeg"), {}),
    ("relative/path", os.path.normpath("relative/path"), {}),
    # --- Quoting ---
    ('"ffmpeg"', "ffmpeg", {}),
    ("'ffmpeg'", "ffmpeg", {}),
    ('  "  /usr/bin/ffmpeg "  ', os.path.normpath("/usr/bin/ffmpeg"), {}),
    ('""', None, {}),
    ("''", None, {}),
    # --- Windows Paths ---
    ("C:/ffmpeg/bin/ffmpeg.exe", os.path.normpath("C:/ffmpeg/bin/ffmpeg.exe"), {}),
    ("C:\\ffmpeg\\bin\\ffmpeg.exe", os.path.normpath('C:\\ffmpeg\\bin\\ffmpeg.exe'), {}),
    # Test the problematic case directly (raw string in env)
    ('C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe', os.path.normpath('C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'), {}),
    # With quotes
    ('"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"', os.path.normpath('C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'), {}),
    # Mixed slashes
    ("C:/Program Files\\ffmpeg/bin\\ffmpeg.exe", os.path.normpath('C:/Program Files\\ffmpeg/bin/ffmpeg.exe'), {}),
    # --- Escape Sequences ---
    (r'C:\Users\test\t\n\r', os.path.normpath(r'C:\Users\test\t\n\r'), {}), # Raw string input -> escapes decoded
    # --- Expansion ---
    ("~/myapp/data.txt", os.path.normpath(f"{HOME_DIR}/myapp/data.txt"), {}),
    ("$MY_APP_VAR/data", os.path.normpath("/test/app/path/data"), {"MY_APP_VAR": "/test/app/path"}),
    ("~/$OTHER_VAR/log", os.path.normpath(f"{HOME_DIR}/other_dir/log"), {"OTHER_VAR": "other_dir"}),
    # Windows-specific expansion
    ("%USERPROFILE%\\Documents", os.path.normpath(f"{os.environ.get('USERPROFILE', HOME_DIR)}\\Documents") if IS_WINDOWS else None, {}),
    ("$UNDEFINED_VAR/path", os.path.normpath("$UNDEFINED_VAR/path"), {}), # Undefined vars are left as-is
]

@pytest.mark.parametrize("raw_input, expected_str, test_env", TEST_CASES)
def test_parse_env_path(monkeypatch, raw_input, expected_str, test_env):
    """Verify parse_env_path handles various inputs correctly."""
    # Skip Windows-specific tests on non-Windows platforms
    if not IS_WINDOWS and os.environ.get('USERPROFILE') and expected_str is None and raw_input and '%USERPROFILE%' in raw_input:
        pytest.skip("Skipping Windows USERPROFILE test on non-Windows platform")

    # Set environment variables for the test
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)

    # Call the function
    result_path = parse_env_path(raw_input)

    # Assertions
    if expected_str is None:
        assert result_path is None
    else:
        assert result_path is not None
        # Compare string representations after normalization
        assert str(result_path) == expected_str
        # Ensure it returns a Path object
        assert isinstance(result_path, Path)

# Specific test for the problematic backslash sequence
def test_parse_env_path_problematic_backslash():
    """Test the exact string causing issues with backslashes."""
    # This raw string is what os.getenv would return if the .env file had:
    # FFMPEG_PATH="C:\Program Files\ffmpeg\bin\ffmpeg.exe"
    # Note: The backslashes are *already* escaped by the time Python sees the string
    raw = 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe' # Literal: C:\Program Files\ffmpeg\bin\ffmpeg.exe
    expected = os.path.normpath(raw) # normpath leaves it as is
    result = parse_env_path(raw)
    assert result is not None
    assert str(result) == expected
