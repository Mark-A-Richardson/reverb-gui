"""A custom QLabel widget that acts as a drag-and-drop target for files."""

import pathlib
from typing import List

from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QLabel, QFrame


class DropZone(QLabel):
    """A QLabel that accepts file drops and emits a signal with the file path."""

    # Signal emitting a pathlib.Path object when a file is dropped
    fileDropped = Signal(pathlib.Path)

    def __init__(self, text: str = "Drag and Drop File Here", parent=None) -> None:
        """Initializes the DropZone widget."""
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        # Basic styling - can be customized further
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                padding: 20px;
                font-size: 16px;
                color: #666;
            }
            QLabel[draggedOver="true"] { /* Custom property to indicate drag-over */
                border-color: #3399ff;
                color: #333;
            }
        """)
        self.setProperty("draggedOver", False) # Initialize custom property

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handles drag enter events. Accepts the drop if it contains file URLs."""
        if event.mimeData().hasUrls():
            print("DropZone: Drag Enter Accepted") # Debug
            self.setProperty("draggedOver", True)
            self.style().unpolish(self) # Force style recomputation
            self.style().polish(self)
            event.acceptProposedAction()
        else:
            print("DropZone: Drag Enter Ignored") # Debug
            event.ignore()

    def dragLeaveEvent(self, event) -> None: # No type hint needed for default QEvent
        """Handles drag leave events. Resets visual state."""
        print("DropZone: Drag Leave") # Debug
        self.setProperty("draggedOver", False)
        self.style().unpolish(self)
        self.style().polish(self)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handles drop events. Extracts file path and emits signal."""
        print("DropZone: Drop Event Triggered") # Debug
        self.setProperty("draggedOver", False) # Reset visual state
        self.style().unpolish(self)
        self.style().polish(self)

        mime_data = event.mimeData()
        if mime_data.hasUrls():
            urls: List[QUrl] = mime_data.urls()
            if urls:
                file_path_str = urls[0].toLocalFile()
                if file_path_str:
                    file_path = pathlib.Path(file_path_str)
                    print(f"DropZone: Emitting file: {file_path}") # Debug
                    self.fileDropped.emit(file_path) # Emit the signal
                    event.acceptProposedAction()
                    return

        print("DropZone: Drop Event Ignored - No valid file URL") # Debug
        event.ignore()
