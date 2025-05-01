import sys
import pathlib
from unittest import mock
import pytest

# Module to test
from reverb_gui import main


def test_launch_gui_success() -> None:
    """Test launch_gui successfully launches the GUI when models are found."""
    # Arrange
    fake_models_path = pathlib.Path("/fake/models")
    mock_main_window_instance = mock.MagicMock()
    mock_qapplication_instance = mock.MagicMock()
    mock_qapplication_instance.exec.return_value = 0 # Simulate normal exit

    # Mock dependencies
    mock_ensure_models = mock.patch('reverb_gui.main.ensure_models_are_downloaded', return_value=fake_models_path)
    mock_sys_exit = mock.patch('sys.exit')
    mock_print = mock.patch('builtins.print')
    mock_qapplication = mock.patch('reverb_gui.main.QApplication', return_value=mock_qapplication_instance)

    # Mock the deferred MainWindow import using sys.modules
    mock_mainwindow_module = mock.MagicMock()
    mock_mainwindow_module.MainWindow.return_value = mock_main_window_instance
    # Ensure the module mock has a MainWindow attribute that is a class/callable mock
    mock_mainwindow_class = mock.MagicMock(return_value=mock_main_window_instance)
    mock_mainwindow_module.MainWindow = mock_mainwindow_class

    # Patch sys.modules to inject the mock before the import happens in launch_gui
    mock_sys_modules = mock.patch.dict(sys.modules, {'reverb_gui.gui.mainwindow': mock_mainwindow_module})

    # Start mocks
    with mock_ensure_models as m_ensure, \
         mock_sys_exit as m_exit, \
         mock_print as m_print, \
         mock_qapplication as m_qapp, \
         mock_sys_modules:

        # Act
        main.launch_gui()

    # Assert
    m_ensure.assert_called_once() # Check models were ensured
    m_print.assert_any_call(f"Models verified/downloaded successfully in: {fake_models_path}")

    # Assert GUI components were initialized and run
    m_qapp.assert_called_once_with(sys.argv)
    mock_mainwindow_class.assert_called_once_with(models_dir=fake_models_path) # Check MainWindow was instantiated with models_dir
    mock_main_window_instance.show.assert_called_once()
    mock_qapplication_instance.exec.assert_called_once()
    m_exit.assert_called_once_with(0) # Check sys.exit called with the result of app.exec


def test_launch_gui_fail_models_missing() -> None:
    """Test launch_gui exits gracefully when models are missing."""
    # Arrange
    mock_ensure_models = mock.patch('reverb_gui.main.ensure_models_are_downloaded', return_value=None)
    mock_print = mock.patch('builtins.print') # Keep print mock for assertions
    mock_qapplication = mock.patch('reverb_gui.main.QApplication') # Keep QApplication mock for assertions
    # No need to mock MainWindow import here as it shouldn't happen

    # Start mocks
    with mock_ensure_models as m_ensure, \
         mock_print as m_print, \
         mock_qapplication as m_qapp:

        # Act & Assert: Expect SystemExit(1)
        with pytest.raises(SystemExit) as excinfo:
            main.launch_gui()

    # Assertions after the expected exit
    assert excinfo.value.code == 1 # Check the exit code
    m_ensure.assert_called_once()
    m_print.assert_any_call("\nFatal Error: Required models could not be verified or downloaded.")
    m_print.assert_any_call("Exiting application.")
    m_qapp.assert_not_called() # Ensure GUI wasn't initialized
