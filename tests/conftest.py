import pytest
import os
import sys
from pathlib import Path

# Add project root to path for all tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtGui import QPixmap

@pytest.fixture(scope="session")
def screenshot_dir():
    """Create and return the directory for storing screenshots"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    screenshot_dir = os.path.join(base_dir, "screenshots")
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    return screenshot_dir

@pytest.fixture
def take_screenshot(screenshot_dir):
    """Fixture to take a screenshot of a widget"""
    def _take_screenshot(widget: QWidget, name: str):
        if not isinstance(widget, QWidget):
            return
        
        # Ensure widget is visible and rendered (if possible in headless)
        # In some headless setups, grab() might return an empty pixmap
        # but we'll try our best.
        pixmap = widget.grab()
        
        filename = f"{name}.png"
        filepath = os.path.join(screenshot_dir, filename)
        pixmap.save(filepath)
        print(f"Screenshot saved: {filepath}")
        return filepath
    
    return _take_screenshot
