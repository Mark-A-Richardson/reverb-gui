import sys
import pathlib
import pytest
from unittest import mock

from intervaltree import Interval, IntervalTree
from pyannote.core import Annotation, Segment

from reverb_gui.pipeline import engine

# Define some realistic mock data
MOCK_CTM_OUTPUT = """
fake_audio.wav 1 0.320 0.180 hello 1.00
fake_audio.wav 1 0.500 0.250 world 1.00
fake_audio.wav 1 1.100 0.400 from 1.00
fake_audio.wav 1 1.500 0.300 cascade 1.00
"""

def create_mock_annotation() -> Annotation:
    """Creates a mock pyannote Annotation object for testing."""
    annotation = Annotation()
    # Matches the structure expected by the original test_transcribe_success
    annotation[Segment(0.0, 1.0)] = "SPK_0"
    annotation[Segment(0.9, 2.0)] = "SPK_1" # Overlapping segment
    return annotation

def create_empty_mock_annotation() -> Annotation:
    """Creates an empty mock pyannote Annotation object.

    This simulates the case where diarization finds no speech segments.
    """
    return Annotation()

# --- Fixtures ---

@pytest.fixture(autouse=True)
def reset_engine_globals(monkeypatch):
    """Ensures model cache and loaded status are reset for each test.

    Also prevents modification of sys.path which might happen in _load_models_if_needed.
    """
    original_sys_path = sys.path[:]

    # Reset the engine's global state
    monkeypatch.setattr(engine, '_models_loaded', False)
    monkeypatch.setattr(engine, '_cached_models', {})
    # Temporarily replace sys.path with a copy
    monkeypatch.setattr(sys, 'path', sys.path[:])

    yield # Test runs here

    # Restore original state
    # monkeypatch handles engine state restoration automatically
    # Restore sys.path explicitly
    sys.path = original_sys_path

@pytest.fixture
def sample_speaker_tree() -> IntervalTree:
    """Provides a sample IntervalTree for testing speaker_for_word.

    Based on the version defined in the original test_engine.py (lines 101-113).
    """
    tree = IntervalTree()
    # SPK_0: 0.0s - 2.0s
    tree.add(Interval(0.0, 2.0, "SPK_0"))
    # SPK_1: 1.5s - 3.5s (overlaps SPK_0)
    tree.add(Interval(1.5, 3.5, "SPK_1"))
    # SPK_0: 4.0s - 5.0s (gap between 3.5s and 4.0s)
    tree.add(Interval(4.0, 5.0, "SPK_0"))
    # SPK_2: 6.0s - 7.0s (another gap)
    tree.add(Interval(6.0, 7.0, "SPK_2"))
    return tree

@pytest.fixture # Removed autouse=True
def mock_models(monkeypatch):
    """Fixture to mock model loading and interaction.

    Sets up mock ASR and Diarization models in the engine's cache and marks
    models as loaded. This prevents actual model loading during unit tests.
    Tests needing this fixture must explicitly request it or use a marker.
    Based on the fixture defined in the original test_engine.py (lines 403-426).
    """
    # If models are already loaded (e.g., by a previous test's specific setup),
    # respect that state. This allows tests specifically testing loading to work.
    if not engine._models_loaded:
        mock_asr_model = mock.MagicMock(name="MockASRModel_Fixture")
        mock_diar_model = mock.MagicMock(name="MockDiarModel_Fixture")
        mock_cache = {
            'asr': mock_asr_model,
            'diarization': mock_diar_model
        }
        monkeypatch.setattr(engine, '_cached_models', mock_cache)
        monkeypatch.setattr(engine, '_models_loaded', True)
    else:
        # If models were already loaded, just yield without changing state
        pass

    # Mock the functions that *perform* loading, so they don't run
    # Use start/stop for manual patch management within the fixture scope
    patcher_load_asr = mock.patch('reverb_gui.pipeline.engine.load_asr_model', autospec=True)
    patcher_diar_pipeline = mock.patch('reverb_gui.pipeline.engine.DiarizationPipeline', autospec=True)
    patcher_get_token = mock.patch('reverb_gui.pipeline.engine._get_hf_token', autospec=True)

    # Start the patchers
    mock_load_asr_func = patcher_load_asr.start()
    mock_diar_pipeline_cls = patcher_diar_pipeline.start()
    mock_get_token_func = patcher_get_token.start()

    # Provide default return values consistent with original fixture
    mock_get_token_func.return_value = "mock_hf_token_fixture"
    # Retrieve potentially pre-existing mocks if engine was already 'loaded'
    diar_mock_in_cache = engine._cached_models.get('diarization', mock.MagicMock(name="FallbackDiarMockFixture"))
    asr_mock_in_cache = engine._cached_models.get('asr', mock.MagicMock(name="FallbackASRMockFixture"))
    mock_diar_pipeline_cls.from_pretrained.return_value = diar_mock_in_cache
    mock_load_asr_func.return_value = asr_mock_in_cache

    yield # Allow tests to run with these mocks

    # Stop the patchers in reverse order
    patcher_get_token.stop()
    patcher_diar_pipeline.stop()
    patcher_load_asr.stop()

    # Cleanup: reset_engine_globals fixture handles resetting _models_loaded/_cached_models
