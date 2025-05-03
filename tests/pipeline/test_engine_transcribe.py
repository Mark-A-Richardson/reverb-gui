import pytest
import pathlib
from unittest import mock

from pyannote.core import Annotation, Segment

# Module to test
from reverb_gui.pipeline import engine
# Import helpers from conftest (needed for explicit calls, fixtures are auto-used)
from .conftest import create_empty_mock_annotation # Import from sibling conftest
# Import shared fixtures/helpers implicitly from conftest.py
# Explicit imports are not needed for fixtures like mock_models, MOCK_CTM_OUTPUT,
# create_mock_annotation, create_empty_mock_annotation


# === Tests for transcribe (originally in test_engine.py) ===

# Mock external dependencies used directly by transcribe
@mock.patch('builtins.print')
@mock.patch('tempfile.TemporaryDirectory')
@mock.patch('reverb_gui.pipeline.engine.convert_to_wav')
@mock.patch('reverb_gui.pipeline.engine.IntervalTree')
@mock.patch('collections.defaultdict')
@mock.patch('reverb_gui.pipeline.engine.speaker_for_word')
@pytest.mark.usefixtures("mock_models") # Explicitly use the model mocking fixture
def test_transcribe_success(
    # Correct parameter order
    mock_speaker_helper: mock.MagicMock, # Decorator 6: speaker_for_word
    mock_defaultdict: mock.MagicMock, # Decorator 5: defaultdict
    mock_interval_tree: mock.MagicMock, # Decorator 4: IntervalTree
    mock_convert: mock.MagicMock, # Decorator 3: convert_to_wav
    mock_tempfile: mock.MagicMock, # Decorator 2: tempfile module
    mock_print: mock.MagicMock, # Decorator 1: print
    tmp_path, # Add tmp_path fixture
    # monkeypatch # Auto-used fixtures from conftest handle state/model setup
    # Explicitly use mock_models fixture from conftest
) -> None:
    """Test successful transcription pipeline.
    Originally from test_engine.py line 431.
    Uses mock_models fixture implicitly.
    """
    # --- Arrange ---
    input_path = pathlib.Path("test_input.mp4")
    temp_dir_path = pathlib.Path("/fake/temp/dir")
    wav_path = temp_dir_path / f"{input_path.stem}.wav"

    # Setup mock return values
    # Mock the .name attribute of the TemporaryDirectory instance
    mock_temp_dir_instance = mock.MagicMock()
    mock_temp_dir_instance.name = str(temp_dir_path) # Set the name attribute
    mock_tempfile.return_value = mock_temp_dir_instance

    mock_convert.return_value = wav_path

    # Mock model interactions (using the mocks set up by mock_models fixture)
    mock_diar_model = engine._cached_models['diarization'] # Get mock from cache
    mock_asr_model = engine._cached_models['asr']         # Get mock from cache

    # Mock diarization result (pyannote.core.Annotation structure)
    mock_annotation = mock.MagicMock(spec=Annotation)
    # Simulate segments for pyannote.core.Annotation
    mock_segments = [
        (Segment(0.0, 1.0), 'track1', 'SPK_0'),
        (Segment(1.1, 2.5), 'track2', 'SPK_1'),
    ]
    mock_annotation.itertracks.return_value = mock_segments
    mock_diar_model.return_value = mock_annotation

    # Simulate ASR CTM output string
    sample_ctm_output = (
        f"{wav_path.stem} 1 0.50 0.30 hello 1.00\n" # Speaker 0
        f"{wav_path.stem} 1 1.20 0.40 world 0.95\n" # Speaker 1
        f"{wav_path.stem} 1 2.10 0.25 reverb 0.88\n" # Speaker 1
    )
    mock_asr_model.transcribe.return_value = sample_ctm_output

    # Mock speaker assignment helper
    # Define side effect based on expected calls
    def speaker_side_effect(start, duration, tree):
        if start == 0.50:
            return "SPK_0"
        elif start == 1.20 or start == 2.10:
            return "SPK_1"
        return "UNKNOWN" # Fallback
    mock_speaker_helper.side_effect = speaker_side_effect

    # --- Act ---
    result = engine.transcribe(input_path, models_dir=tmp_path, asr_params={})

    # --- Assert ---
    # Verify setup calls
    mock_tempfile.assert_called_once()
    mock_convert.assert_called_once_with(input_path, output_dir=temp_dir_path) # Expect the Path object

    # Verify model calls
    mock_diar_model.assert_called_once_with(str(wav_path))

    # Verify IntervalTree creation and population based on mock_segments
    mock_interval_tree.assert_called_once()
    expected_tree_calls = [
        mock.call(engine.Interval(0.0, 1.0, 'SPK_0')),
        mock.call(engine.Interval(1.1, 2.5, 'SPK_1')),
    ]
    mock_interval_tree.return_value.add.assert_has_calls(expected_tree_calls, any_order=True)

    # Verify ASR call
    mock_asr_model.transcribe.assert_called_once_with(
        str(wav_path),
        mode="ctc_prefix_beam_search",
        beam_size=10,
        length_penalty=0.0,
        ctc_weight=0.1,
        reverse_weight=0.0,
        blank_penalty=0.0,
        verbatimicity=1.0,
        format="ctm"
    )

    # Verify speaker_for_word calls based on sample CTM
    expected_speaker_calls = [
        mock.call(0.50, 0.30, mock_interval_tree.return_value), # hello -> SPK_0
        mock.call(1.20, 0.40, mock_interval_tree.return_value), # world -> SPK_1
        mock.call(2.10, 0.25, mock_interval_tree.return_value), # reverb -> SPK_1
    ]
    mock_speaker_helper.assert_has_calls(expected_speaker_calls)

    # Verify final output format
    expected_output = [
        (0.50, 0.80, "SPK_0", "hello"), # start, start + duration, speaker, word
        (1.20, 1.60, "SPK_1", "world"),
        (2.10, 2.35, "SPK_1", "reverb"),
    ]
    assert result == expected_output

    # Verify temp dir cleanup
    # Check cleanup() was called on the specific instance we created
    mock_temp_dir_instance.cleanup.assert_called_once()
    mock_print.assert_called() # Check for some logging

@mock.patch('builtins.print')
@mock.patch('tempfile.TemporaryDirectory')
@mock.patch('reverb_gui.pipeline.engine.convert_to_wav', side_effect=RuntimeError("Conversion failed!"))
@pytest.mark.usefixtures("mock_models") # Models not used, but fixture provides state reset
def test_transcribe_conversion_failure(
    mock_convert: mock.MagicMock, # Decorator 3
    mock_tempfile: mock.MagicMock, # Decorator 2
    mock_print: mock.MagicMock, # Decorator 1
    tmp_path, # Add tmp_path fixture
    # monkeypatch # Autouse fixture from conftest handles reset
) -> None:
    """Test transcribe handles failure during the convert_to_wav step.
    Originally from test_engine.py line 526.
    """
    # --- Arrange ---
    input_path = pathlib.Path("input.avi")
    temp_dir_path = pathlib.Path("/fake/temp/dir")
    # Mock the .name attribute of the TemporaryDirectory instance
    mock_temp_dir_instance = mock.MagicMock()
    mock_temp_dir_instance.name = str(temp_dir_path)
    mock_tempfile.return_value = mock_temp_dir_instance
    # The cleanup won't be called in this failure case, so no need to mock it

    mock_diar_model = engine._cached_models.get('diarization') # Should exist from fixture
    mock_asr_model = engine._cached_models.get('asr')         # Should exist from fixture

    # --- Act & Assert ---
    with pytest.raises(RuntimeError, match="Conversion failed!"):
        engine.transcribe(input_path, models_dir=tmp_path, asr_params={})

    # Verify calls up to failure point
    mock_tempfile.assert_called_once()
    mock_convert.assert_called_once_with(input_path, output_dir=temp_dir_path) # Expect the Path object

    # Verify models were NOT called
    assert mock_diar_model is not None and not mock_diar_model.called
    assert mock_asr_model is not None and not mock_asr_model.transcribe.called

    # Verify temp dir cleanup still happened
    mock_temp_dir_instance.cleanup.assert_called_once() # Check for cleanup call
    mock_print.assert_called()

@mock.patch('builtins.print')
@mock.patch('tempfile.TemporaryDirectory')
@mock.patch('reverb_gui.pipeline.engine.convert_to_wav')
@pytest.mark.usefixtures("mock_models") # Needs mocked diarization model
def test_transcribe_diarization_failure(
    mock_convert: mock.MagicMock, # Decorator 3
    mock_tempfile: mock.MagicMock, # Decorator 2
    mock_print: mock.MagicMock, # Decorator 1
    tmp_path, # Add tmp_path fixture
    # monkeypatch # Autouse fixture from conftest handles reset and models
) -> None:
    """Test transcribe handles failure during the diarization step.
    Originally from test_engine.py line 565.
    """
    # --- Arrange ---
    input_path = pathlib.Path("input.flac")
    temp_dir_path = pathlib.Path("/fake/temp/dir")
    wav_path = temp_dir_path / f"{input_path.stem}.wav"

    # Mock the .name attribute of the TemporaryDirectory instance
    mock_temp_dir_instance = mock.MagicMock()
    mock_temp_dir_instance.name = str(temp_dir_path)
    mock_tempfile.return_value = mock_temp_dir_instance

    mock_convert.return_value = wav_path

    # Get mock models from cache (populated by fixture)
    mock_diar_model = engine._cached_models['diarization']
    mock_asr_model = engine._cached_models['asr']
    # Setup diarization to fail
    mock_diar_model.side_effect = RuntimeError("Diarization crashed!")

    # --- Act & Assert ---
    with pytest.raises(RuntimeError, match="Diarization crashed!"):
        engine.transcribe(input_path, models_dir=tmp_path, asr_params={})

    # Verify calls up to failure point
    mock_tempfile.assert_called_once()
    mock_convert.assert_called_once_with(input_path, output_dir=temp_dir_path) # Expect the Path object
    mock_diar_model.assert_called_once_with(str(wav_path))

    # Verify ASR was NOT called
    assert mock_asr_model is not None and not mock_asr_model.transcribe.called

    # Verify temp dir cleanup still happened
    mock_temp_dir_instance.cleanup.assert_called_once()
    mock_print.assert_called()

@mock.patch('builtins.print')
@mock.patch('tempfile.TemporaryDirectory')
@mock.patch('reverb_gui.pipeline.engine.convert_to_wav')
@pytest.mark.usefixtures("mock_models") # Needs mocked ASR model
def test_transcribe_asr_failure(
    mock_convert: mock.MagicMock, # Decorator 3
    mock_tempfile: mock.MagicMock, # Decorator 2
    mock_print: mock.MagicMock, # Decorator 1
    tmp_path, # Add tmp_path fixture
    # monkeypatch # Autouse fixture from conftest handles reset and models
) -> None:
    """Test transcribe handles failure during the ASR transcribe step.
    Originally from test_engine.py line 613.
    """
    # --- Arrange ---
    input_path = pathlib.Path("input.mkv")
    temp_dir_path = pathlib.Path("/fake/temp/dir")
    wav_path = temp_dir_path / f"{input_path.stem}.wav"

    # Mock the .name attribute of the TemporaryDirectory instance
    mock_temp_dir_instance = mock.MagicMock()
    mock_temp_dir_instance.name = str(temp_dir_path)
    mock_tempfile.return_value = mock_temp_dir_instance

    mock_convert.return_value = wav_path

    # Get mock models from cache (populated by fixture)
    mock_diar_model = engine._cached_models['diarization']
    mock_asr_model = engine._cached_models['asr']

    # Setup successful diarization (create mock annotation directly)
    mock_annotation = mock.MagicMock(spec=Annotation)
    mock_segments = [
        (Segment(0.0, 1.0), 'track1', 'SPK_0'),
        (Segment(1.5, 2.5), 'track2', 'SPK_1'),
    ]
    mock_annotation.itertracks.return_value = mock_segments
    mock_diar_model.return_value = mock_annotation

    # Setup ASR to fail
    mock_asr_model.transcribe.side_effect = RuntimeError("ASR crashed!")

    # --- Act & Assert ---
    with pytest.raises(RuntimeError, match="ASR crashed!"):
        engine.transcribe(input_path, models_dir=tmp_path, asr_params={})

    # Verify calls up to failure point
    mock_tempfile.assert_called_once()
    mock_convert.assert_called_once_with(input_path, output_dir=temp_dir_path) # Expect Path object
    mock_diar_model.assert_called_once_with(str(wav_path))
    mock_asr_model.transcribe.assert_called_once_with(
        str(wav_path),
        mode="ctc_prefix_beam_search",
        beam_size=10,
        length_penalty=0.0,
        ctc_weight=0.1,
        reverse_weight=0.0,
        blank_penalty=0.0,
        verbatimicity=1.0,
        format="ctm"
    )

    # Verify temp dir cleanup still happened
    mock_temp_dir_instance.cleanup.assert_called_once()
    mock_print.assert_called()

@mock.patch('builtins.print')
@mock.patch('tempfile.TemporaryDirectory')
@mock.patch('reverb_gui.pipeline.engine.convert_to_wav')
@mock.patch('reverb_gui.pipeline.engine.IntervalTree')
@mock.patch('collections.defaultdict')
@mock.patch('reverb_gui.pipeline.engine.speaker_for_word')
@pytest.mark.usefixtures("mock_models") # Needs mocked models
def test_transcribe_empty_ctm(
    # Correct parameter order
    mock_speaker_helper: mock.MagicMock,
    mock_defaultdict: mock.MagicMock,
    mock_interval_tree: mock.MagicMock,
    mock_convert: mock.MagicMock,
    mock_tempfile: mock.MagicMock,
    mock_print: mock.MagicMock,
    tmp_path, # Add tmp_path fixture
    # monkeypatch # Autouse fixture handles reset/models
) -> None:
    """Test transcribe handles an empty CTM result from ASR.
    Originally from test_engine.py line 667.
    """
    # --- Arrange ---
    input_path = pathlib.Path("input.mp3")
    temp_dir_path = pathlib.Path("/fake/temp/dir")
    wav_path = temp_dir_path / f"{input_path.stem}.wav"

    # Mock the .name attribute of the TemporaryDirectory instance
    mock_temp_dir_instance = mock.MagicMock()
    mock_temp_dir_instance.name = str(temp_dir_path)
    mock_tempfile.return_value = mock_temp_dir_instance

    mock_convert.return_value = wav_path

    mock_diar_model = engine._cached_models['diarization']
    mock_asr_model = engine._cached_models['asr']

    # Successful diarization (create mock annotation directly)
    mock_annotation = mock.MagicMock(spec=Annotation)
    mock_segments = [
        (Segment(0.0, 5.0), 'track1', 'SPK_0'),
    ]
    mock_annotation.itertracks.return_value = mock_segments
    mock_diar_model.return_value = mock_annotation

    # ASR returns empty CTM
    mock_asr_model.transcribe.return_value = ""

    # Mock speaker helper (should not be called if CTM is empty)
    # Define side effect based on expected calls
    def speaker_side_effect(start, duration, tree):
        return "UNKNOWN" # Fallback
    mock_speaker_helper.side_effect = speaker_side_effect

    # --- Act ---
    result = engine.transcribe(input_path, models_dir=tmp_path, asr_params={})

    # --- Assert ---
    mock_tempfile.assert_called_once()
    mock_convert.assert_called_once_with(input_path, output_dir=temp_dir_path) # Expect Path object
    mock_diar_model.assert_called_once_with(str(wav_path))
    mock_asr_model.transcribe.assert_called_once_with(
        str(wav_path),
        mode="ctc_prefix_beam_search",
        beam_size=10,
        length_penalty=0.0,
        ctc_weight=0.1,
        reverse_weight=0.0,
        blank_penalty=0.0,
        verbatimicity=1.0,
        format="ctm"
    )

    # Speaker helper should not be called for empty CTM
    mock_speaker_helper.assert_not_called()

    # Result should be empty list
    assert result == []

    # Verify temp dir cleanup still happened
    mock_temp_dir_instance.cleanup.assert_called_once()
    # Check for warning print about empty CTM
    mock_print.assert_any_call("Warning: ASR returned empty CTM.")

@mock.patch('builtins.print')
@mock.patch('tempfile.TemporaryDirectory')
@mock.patch('reverb_gui.pipeline.engine.convert_to_wav')
@mock.patch('reverb_gui.pipeline.engine.IntervalTree')
@mock.patch('collections.defaultdict')
@mock.patch('reverb_gui.pipeline.engine.speaker_for_word')
@pytest.mark.usefixtures("mock_models") # Needs mocked models
def test_transcribe_empty_diarization(
    # Correct parameter order
    mock_speaker_helper: mock.MagicMock,
    mock_defaultdict: mock.MagicMock,
    mock_interval_tree: mock.MagicMock,
    mock_convert: mock.MagicMock,
    mock_tempfile: mock.MagicMock,
    mock_print: mock.MagicMock,
    tmp_path, # Add tmp_path fixture
    # monkeypatch # Autouse fixture handles reset/models
) -> None:
    """Test transcribe handles empty results from diarization.
    Originally from test_engine.py line 734.
    """
    # --- Arrange ---
    input_path = pathlib.Path("input.ogg")
    temp_dir_path = pathlib.Path("/fake/temp/dir")
    wav_path = temp_dir_path / f"{input_path.stem}.wav"

    # Mock the .name attribute of the TemporaryDirectory instance
    mock_temp_dir_instance = mock.MagicMock()
    mock_temp_dir_instance.name = str(temp_dir_path)
    mock_tempfile.return_value = mock_temp_dir_instance

    mock_convert.return_value = wav_path

    mock_diar_model = engine._cached_models['diarization']
    mock_asr_model = engine._cached_models['asr']

    # Empty diarization result (mock Annotation with empty itertracks)
    mock_annotation = mock.MagicMock()
    mock_annotation.itertracks.return_value = [] # Return empty list
    mock_annotation.__bool__ = lambda _: False # Accept self argument (ignored)
    mock_diar_model.return_value = mock_annotation

    # Successful ASR with some words
    sample_ctm_output = (
        f"{wav_path.stem} 1 0.50 0.30 hello 1.00\n"
        f"{wav_path.stem} 1 1.20 0.40 world 0.95\n"
    )
    mock_asr_model.transcribe.return_value = sample_ctm_output

    # Speaker helper should return UNKNOWN because tree is empty
    mock_speaker_helper.return_value = "UNKNOWN"

    # --- Act ---
    result = engine.transcribe(input_path, models_dir=tmp_path, asr_params={})

    # --- Assert ---
    mock_tempfile.assert_called_once()
    mock_convert.assert_called_once_with(input_path, output_dir=temp_dir_path) # Expect Path object
    mock_diar_model.assert_called_once_with(str(wav_path))
    mock_interval_tree.assert_called_once()
    mock_interval_tree.return_value.add.assert_not_called() # Tree should be empty

    mock_asr_model.transcribe.assert_called_once_with(
        str(wav_path),
        mode="ctc_prefix_beam_search",
        beam_size=10,
        length_penalty=0.0,
        ctc_weight=0.1,
        reverse_weight=0.0,
        blank_penalty=0.0,
        verbatimicity=1.0,
        format="ctm"
    )

    # Speaker helper should be called for each word, returning UNKNOWN
    expected_speaker_calls = [
        mock.call(0.50, 0.30, mock_interval_tree.return_value),
        mock.call(1.20, 0.40, mock_interval_tree.return_value),
    ]
    mock_speaker_helper.assert_has_calls(expected_speaker_calls)

    # Expect result with UNKNOWN speakers
    expected_result = [
        (0.50, 0.80, "UNKNOWN", "hello"),
        (1.20, 1.60, "UNKNOWN", "world"),
    ]
    assert result == expected_result

    # Verify temp dir cleanup still happened
    mock_temp_dir_instance.cleanup.assert_called_once()
    # Check for warning print about empty diarization
    mock_print.assert_any_call("Warning: Diarization returned no segments.")

@mock.patch('builtins.print')
@mock.patch('tempfile.TemporaryDirectory')
@mock.patch('reverb_gui.pipeline.engine.convert_to_wav')
@mock.patch('reverb_gui.pipeline.engine.IntervalTree') # Needed for diarization part
@mock.patch('reverb_gui.pipeline.engine.speaker_for_word') # Needed for alignment part
@pytest.mark.usefixtures("mock_models")
def test_transcribe_passes_asr_params(
    mock_speaker_helper: mock.MagicMock, # Decorator 5
    mock_interval_tree_cls: mock.MagicMock, # Decorator 4: IntervalTree class
    mock_convert: mock.MagicMock,       # Decorator 3
    mock_tempfile: mock.MagicMock,      # Decorator 2
    mock_print: mock.MagicMock,         # Decorator 1
    tmp_path # tmp_path fixture
    # mock_models fixture is applied via decorator
) -> None:
    """Verify that asr_params are correctly unpacked and passed to asr_model.transcribe."""
    # --- Arrange ---
    input_path = tmp_path / "test_input.mp3"
    input_path.touch() # Create dummy file
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    temp_dir_path = tmp_path / "temp_transcribe"
    wav_path = temp_dir_path / f"{input_path.stem}.wav"

    # Mock temp dir creation
    mock_temp_dir_instance = mock.MagicMock()
    mock_temp_dir_instance.name = str(temp_dir_path)
    mock_tempfile.return_value = mock_temp_dir_instance

    mock_convert.return_value = wav_path

    # Mock IntervalTree instance and its methods if necessary for diarization part
    mock_tree_instance = mock.MagicMock()
    mock_interval_tree_cls.return_value = mock_tree_instance

    # Mock speaker assignment (return dummy speaker)
    mock_speaker_helper.return_value = "SPK_X"

    # Get mocked models from the fixture-populated cache
    mock_asr_model = engine._cached_models['asr']
    mock_diar_model = engine._cached_models['diarization']
    # Mock ASR output (simple CTM format)
    mock_asr_model.transcribe.return_value = "1 /fake/audio.wav 1 0.1 0.5 hello\n1 /fake/audio.wav 1 0.7 0.4 world"
    # Mock Diarization output (can use helper from conftest)
    mock_diar_model.return_value = create_empty_mock_annotation() # Ensure it runs

    # Define specific ASR parameters to test
    test_asr_params = {
        "mode": "attention_rescoring",
        "beam_size": 5,
        "length_penalty": -0.5,
        "ctc_weight": 0.7,
        "reverse_weight": 0.2,
        "blank_penalty": 1.1,
        "verbatimicity": 0.9,
    }
    expected_asr_call_args = {
        "mode": "attention_rescoring",
        "beam_size": 5,
        "length_penalty": -0.5,
        "ctc_weight": 0.7,
        "reverse_weight": 0.2,
        "blank_penalty": 1.1,
        "verbatimicity": 0.9,
        "format": "ctm", # This is hardcoded in the engine
    }

    # --- Act ---
    engine.transcribe(input_path, models_dir, asr_params=test_asr_params)

    # --- Assert ---
    # Verify that the mocked asr_model's transcribe method was called correctly
    mock_asr_model.transcribe.assert_called_once()
    # Check positional argument (wav path)
    call_args, call_kwargs = mock_asr_model.transcribe.call_args
    assert call_args[0] == str(wav_path)
    # Check keyword arguments (ASR parameters)
    assert call_kwargs == expected_asr_call_args
