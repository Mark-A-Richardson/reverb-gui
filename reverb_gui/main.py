"""Main entry point for the reverb-gui application."""

import sys
from typing import NoReturn

from PySide6.QtWidgets import QApplication

# Import MainWindow AFTER checking models, to avoid Qt imports if exit needed
# from .gui.mainwindow import MainWindow # Deferred import
from .utils.model_downloader import ensure_models_are_downloaded


def launch_gui() -> NoReturn:
    """Initializes and runs the main application window.

    Ensures required models are downloaded before launching.
    """
    print("Starting reverb-gui application...")

    # 1. Ensure models are present before initializing GUI
    models_dir = ensure_models_are_downloaded()
    if models_dir is None:
        print("\nFatal Error: Required models could not be verified or downloaded.")
        print("Please check your internet connection, Hugging Face token (.env file), and accept any necessary model licenses on the Hugging Face Hub.")
        print("Exiting application.")
        sys.exit(1) # Exit with an error code
    else:
        print(f"Models verified/downloaded successfully in: {models_dir}")

    # 2. Initialize and launch the GUI only if models are ready
    # Import MainWindow now that we know we can proceed
    from .gui.mainwindow import MainWindow

    app = QApplication(sys.argv)
    # Pass models_dir to MainWindow
    window = MainWindow(models_dir=models_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
