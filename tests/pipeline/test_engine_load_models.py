import pytest
import pathlib
from unittest import mock

# Module to test
from reverb_gui.pipeline import engine

# Note: Fixtures like `reset_engine_globals` are automatically used from conftest.py

# === Tests for _load_models_if_needed (originally in test_engine.py) ===

@pytest.mark.parametrize(
    "cuda_available, expected_device_str, expected_gpu_index",
    [
        (False, "cpu", None), # Test CPU case
        (True, "cuda", 0),    # Test CUDA case (assuming index 0)
    ]
)
@mock.patch('reverb_gui.pipeline.engine.ensure_models_are_downloaded')
@mock.patch('reverb_gui.pipeline.engine._get_hf_token')
@mock.patch('reverb_gui.pipeline.engine.torch.cuda.is_available')
@mock.patch('reverb_gui.pipeline.engine.torch.cuda.get_device_name')
@mock.patch('reverb_gui.pipeline.engine.torch.device')
@mock.patch('reverb_gui.pipeline.engine.DiarizationPipeline') # Mock the class
@mock.patch('reverb_gui.pipeline.engine.load_asr_model')
@mock.patch('builtins.print') # Mock print to suppress output during test
def test_load_models_if_needed_success(
    mock_print: mock.MagicMock,
    mock_load_asr: mock.MagicMock,
    mock_diar_pipeline_cls: mock.MagicMock, # Class mock
    mock_torch_device: mock.MagicMock,
    mock_cuda_get_name: mock.MagicMock,
    mock_cuda_available: mock.MagicMock,
    mock_get_token: mock.MagicMock,
    mock_ensure_models: mock.MagicMock,
    cuda_available: bool,
    expected_device_str: str,
    expected_gpu_index: int | None,
    # Note: `monkeypatch` is available via pytest, `reset_engine_globals` is from conftest
) -> None:
    """Test _load_models_if_needed successfully loads models (CPU/CUDA).
    Originally from test_engine.py line 153.
    """
    # --- Arrange ---
    # Global state reset by reset_engine_globals

    fake_models_dir = pathlib.Path("/fake/models/path")
    fake_token = "fake_hf_token"
    mock_ensure_models.return_value = fake_models_dir
    mock_get_token.return_value = fake_token
    mock_cuda_available.return_value = cuda_available
    mock_cuda_get_name.return_value = "Fake NVIDIA GPU"
    mock_torch_device_instance = mock.MagicMock(name="MockTorchDevice")
    mock_torch_device.return_value = mock_torch_device_instance

    # Mock the DiarizationPipeline instance and its methods
    mock_diar_instance = mock.MagicMock(name="MockDiarPipelineInstance")
    mock_diar_pipeline_cls.from_pretrained.return_value = mock_diar_instance

    # Mock the ASR model instance
    mock_asr_instance = mock.MagicMock(name="MockASRInstance")
    mock_load_asr.return_value = mock_asr_instance

    # Construct expected path for ASR model load call
    expected_asr_model_path_str = f"models--{engine.ASR_MODEL_ID.replace('/', '--')}"
    expected_asr_path = fake_models_dir / expected_asr_model_path_str

    # --- Act ---
    engine._load_models_if_needed()

    # --- Assert ---
    # Verify core setup functions were called
    mock_ensure_models.assert_called_once()
    mock_get_token.assert_called_once()

    # Verify CUDA checks and device setup
    if cuda_available:
        assert mock_cuda_available.call_count == 2 # Called in print and if
        mock_cuda_get_name.assert_called_once_with(expected_gpu_index)
        mock_torch_device.assert_called_once_with(expected_device_str)
    else:
        assert mock_cuda_available.call_count == 2 # Called in print and if
        mock_cuda_get_name.assert_not_called()
        mock_torch_device.assert_called_once_with(expected_device_str)

    # Verify model loading calls
    mock_diar_pipeline_cls.from_pretrained.assert_called_once_with(
        engine.DIARIZATION_MODEL_ID,
        use_auth_token=fake_token,
        cache_dir=str(fake_models_dir) # Expect string representation
    )
    mock_load_asr.assert_called_once_with(
        model=str(expected_asr_path), # Pass model path as string
        gpu=expected_gpu_index       # Pass the GPU index (or None)
    )

    # Verify models are cached
    assert engine._models_loaded
    assert engine._cached_models['asr'] == mock_asr_instance
    assert engine._cached_models['diarization'] == mock_diar_instance
    mock_print.assert_called() # Ensure some logging occurred

@mock.patch('reverb_gui.pipeline.engine.ensure_models_are_downloaded')
@mock.patch('reverb_gui.pipeline.engine.load_asr_model')
@mock.patch('reverb_gui.pipeline.engine.DiarizationPipeline')
def test_load_models_if_needed_caching(
    mock_diar_pipeline_cls: mock.MagicMock,
    mock_load_asr: mock.MagicMock,
    mock_ensure_models: mock.MagicMock,
    monkeypatch # Need to manually set initial loaded state
) -> None:
    """Test that models are not reloaded if already loaded.
    Originally from test_engine.py line 243.
    """
    # --- Arrange ---
    # Manually set the loaded state
    monkeypatch.setattr(engine, '_models_loaded', True)
    monkeypatch.setattr(engine, '_cached_models', {'asr': 'dummy_asr', 'diarization': 'dummy_diar', 'device': 'dummy_device'})

    # --- Act ---
    engine._load_models_if_needed() # Call the function again

    # --- Assert ---
    # Assert that loading functions were NOT called this time
    mock_ensure_models.assert_not_called()
    mock_diar_pipeline_cls.from_pretrained.assert_not_called()
    mock_load_asr.assert_not_called()
    # Check that the cache remains untouched
    assert engine._cached_models['asr'] == 'dummy_asr'

@mock.patch('reverb_gui.pipeline.engine.ensure_models_are_downloaded', side_effect=RuntimeError("Download failed!"))
@mock.patch('builtins.print')
def test_load_models_if_needed_fail_download(
    mock_print: mock.MagicMock,
    mock_ensure_models: mock.MagicMock,
    # monkeypatch # reset_engine_globals handles state reset
) -> None:
    """Test _load_models_if_needed raises RuntimeError if download fails.
    Originally from test_engine.py line 277.
    """
    # --- Act & Assert ---
    with pytest.raises(RuntimeError, match="Download failed!"):
        engine._load_models_if_needed()

    # Ensure state reflects failure
    assert engine._models_loaded is False
    assert engine._cached_models == {}
    mock_print.assert_called() # Should still print initial messages

@mock.patch('reverb_gui.pipeline.engine.ensure_models_are_downloaded')
@mock.patch('reverb_gui.pipeline.engine._get_hf_token', return_value="fake_token")
@mock.patch('reverb_gui.pipeline.engine.torch.cuda.is_available', return_value=False) # CPU
@mock.patch('reverb_gui.pipeline.engine.torch.device', return_value='cpu')
@mock.patch('reverb_gui.pipeline.engine.DiarizationPipeline.from_pretrained', side_effect=RuntimeError("Diarization load error!"))
@mock.patch('builtins.print')
def test_load_models_if_needed_fail_diar_load(
    mock_print: mock.MagicMock,
    mock_diar_from_pretrained: mock.MagicMock,
    mock_torch_device: mock.MagicMock,
    mock_cuda_available: mock.MagicMock,
    mock_get_token: mock.MagicMock,
    mock_ensure_models: mock.MagicMock,
    # monkeypatch
) -> None:
    """Test _load_models_if_needed raises RuntimeError if diarization model load fails.
    Originally from test_engine.py line 298.
    """
    # --- Arrange ---
    fake_models_dir = pathlib.Path("/fake/path")
    mock_ensure_models.return_value = fake_models_dir

    # --- Act & Assert ---
    with pytest.raises(RuntimeError, match="Diarization load error!"):
        engine._load_models_if_needed()

    # Ensure state reflects failure
    assert engine._models_loaded is False
    assert engine._cached_models == {}
    # Verify setup calls up to the point of failure
    mock_ensure_models.assert_called_once()
    mock_get_token.assert_called_once()
    assert mock_cuda_available.call_count == 2
    mock_torch_device.assert_called_once_with("cpu")
    mock_diar_from_pretrained.assert_called_once_with(
        engine.DIARIZATION_MODEL_ID, 
        use_auth_token="fake_token",
        cache_dir=str(fake_models_dir)
    )

@mock.patch('reverb_gui.pipeline.engine.ensure_models_are_downloaded')
@mock.patch('reverb_gui.pipeline.engine._get_hf_token', return_value="fake_token")
@mock.patch('reverb_gui.pipeline.engine.torch.cuda.is_available', return_value=False) # CPU
@mock.patch('reverb_gui.pipeline.engine.torch.device')
@mock.patch('reverb_gui.pipeline.engine.DiarizationPipeline')
@mock.patch('reverb_gui.pipeline.engine.load_asr_model', side_effect=RuntimeError("ASR load error!"))
@mock.patch('builtins.print')
def test_load_models_if_needed_fail_asr_load(
    mock_print: mock.MagicMock,
    mock_load_asr: mock.MagicMock,
    mock_diar_pipeline_cls: mock.MagicMock,
    mock_torch_device: mock.MagicMock,
    mock_cuda_available: mock.MagicMock,
    mock_get_token: mock.MagicMock,
    mock_ensure_models: mock.MagicMock,
    # monkeypatch
) -> None:
    """Test _load_models_if_needed raises RuntimeError if ASR model load fails.
    Originally from test_engine.py line 334.
    """
    # --- Arrange ---
    fake_models_dir = pathlib.Path("/fake/models/path")
    mock_ensure_models.return_value = fake_models_dir
    mock_diar_instance = mock.MagicMock(name="MockDiarPipelineInstance")
    mock_diar_pipeline_cls.from_pretrained.return_value = mock_diar_instance
    mock_cpu_device = mock.MagicMock(name="MockCPUDevice")
    mock_torch_device.return_value = mock_cpu_device

    expected_asr_model_path_str = f"models--{engine.ASR_MODEL_ID.replace('/', '--')}"
    expected_asr_path = fake_models_dir / expected_asr_model_path_str

    # --- Act & Assert ---
    with pytest.raises(RuntimeError, match="ASR load error!"):
        engine._load_models_if_needed()

    # Ensure state reflects failure
    assert engine._models_loaded is False
    assert engine._cached_models == {}
    # Verify setup calls up to the point of failure
    mock_ensure_models.assert_called_once()
    mock_get_token.assert_called_once()
    assert mock_cuda_available.call_count == 2
    mock_torch_device.assert_called_once_with("cpu")
    mock_diar_pipeline_cls.from_pretrained.assert_called_once()
    mock_load_asr.assert_called_once_with(
        model=str(expected_asr_path), # Pass model path as string
        gpu=None                     # Expect None for GPU index in CPU case
    )

@mock.patch('reverb_gui.pipeline.engine.torch.cuda.is_available', return_value=False)
@mock.patch('builtins.print')
def test_load_models_if_needed_already_loaded(
    mock_print: mock.MagicMock,
    mock_cuda_is_available: mock.MagicMock, # Added mock
    reset_engine_globals: None, # Added fixture
) -> None:
    """Test that models are not reloaded if already loaded.
    Originally from test_engine.py line 243.
    """
    # --- Arrange ---
    # Manually set the loaded state (reset_engine_globals ensures cleanup)
    engine._models_loaded = True
    engine._cached_models = {'asr': 'dummy_asr', 'diarization': 'dummy_diar'}

    # --- Act ---
    engine._load_models_if_needed() # Call the function again

    # --- Assert ---
    # Assert that the 'already loaded' message was printed exactly once
    expected_message = "Engine: Models already loaded. Device check: torch.cuda.is_available() -> False"
    mock_print.assert_called_once_with(expected_message)
    mock_cuda_is_available.assert_called_once() # Verify the check happened within print

    # Check that the cache remains untouched
    assert engine._cached_models['asr'] == 'dummy_asr'
    assert engine._cached_models['diarization'] == 'dummy_diar'
    assert len(engine._cached_models) == 2 # Ensure no extra keys were added
