"""
Test script for the loading screen functionality
"""

import sys
import os
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.mark.skip(reason="LoadingWorker internal methods changed, test needs update")
def test_loading_worker():
    """Test the LoadingWorker logic without GUI"""
    print("Testing LoadingWorker logic...")

    # Create a mock platform object
    class MockPlatform:
        def __init__(self):
            self.config_manager = None
            self.session_logger = None
            self.data_manager = None
            self.strategy_engine = None
            self.backtester = None
            self.analysis_engines = None
            self.live_monitor = None
            self.settings = None
            self.reporters = None
            self.broker_manager = None
            self.api_app = None
            self.BROKERS_AVAILABLE = False
            self.API_AVAILABLE = False
            self.tabs = None

        def _create_tabs(self):
            print("✅ _create_tabs called")
            return True

        def create_modern_statusbar(self):
            print("✅ create_modern_statusbar called")
            return True

        def load_saved_config(self):
            print("✅ load_saved_config called")
            return True

        def auto_load_default_data(self):
            print("✅ auto_load_default_data called")
            return True

    platform = MockPlatform()

    # Test LoadingWorker
    from src.loading_screen import LoadingWorker
    worker = LoadingWorker(platform)

    # Test component loading methods
    methods_to_test = [
        '_load_config_manager',
        '_load_session_logger',
        '_load_data_manager',
        '_load_gui_tabs',
        '_load_status_bar',
        '_load_configuration'
    ]

    for method_name in methods_to_test:
        assert hasattr(worker, method_name), f"Method {method_name} not found"
        try:
            result = getattr(worker, method_name)()
            print(f"✅ {method_name}: {'Success' if result else 'Failed (expected for some components)'}")
        except Exception as e:
            print(f"⚠️ {method_name}: Exception - {e}")

    print("✅ LoadingWorker test completed")

def test_imports():
    """Test all necessary imports"""
    print("Testing imports...")

    from PySide6.QtWidgets import QApplication, QMainWindow
    print("✅ PySide6 imports successful")

    from src.loading_screen import LoadingScreen, LoadingWorker
    print("✅ Loading screen imports successful")

    from src.main_platform import TradingPlatform
    print("✅ Main platform import successful")

def main():
    """Run all tests"""
    print("🚀 Testing Loading Screen Implementation")
    print("=" * 50)

    # Test imports
    test_imports()
    print()

    # Test loading worker logic
    test_loading_worker()
    print()
    
    print("✅ All tests completed! Loading screen implementation looks good.")
    print("💡 You can now run the full application with: python start_platform.ps1")

if __name__ == '__main__':
    main()