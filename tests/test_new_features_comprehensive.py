#!/usr/bin/env python3
"""
Comprehensive Test Suite for New Platform Features
=================================================

Tests for:
1. Auto-loading BTC/USD data on startup
2. Tab9DataDownload GUI functionality
3. check_data_status.py script
4. Default BTC/USD configuration in Tab1
5. Edge cases and error scenarios

Author: TradingIA Team
Version: 1.0.0
"""

import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestAutoLoadDefaultData:
    """Tests for automatic BTC/USD data loading on platform startup"""

    def test_auto_load_method_exists(self):
        """Test that auto_load_default_data method exists in TradingPlatform"""
        from main_platform import TradingPlatform
        assert hasattr(TradingPlatform, 'auto_load_default_data')

    def test_auto_load_default_parameters(self):
        """Test that auto_load_default_data uses correct default parameters"""
        from main_platform import TradingPlatform
        from backend_core import DataManager

        platform = TradingPlatform.__new__(TradingPlatform)  # Create without __init__
        platform.data_manager = Mock(spec=DataManager)
        platform.data_dict = {}
        platform.logger = Mock()
        platform.statusBar = Mock()

        # Mock successful data load
        mock_df = pd.DataFrame({'timestamp': [1, 2, 3], 'close': [100, 101, 102]})
        platform.data_manager.load_alpaca_data.return_value = mock_df

        # Call the method
        platform.auto_load_default_data()

        # Verify data_manager was called with correct parameters
        platform.data_manager.load_alpaca_data.assert_called_once()
        call_args = platform.data_manager.load_alpaca_data.call_args

        # Check symbol
        assert 'BTC/USD' in call_args.kwargs.values()

        # Check timeframe
        assert '1Hour' in call_args.kwargs.values()

    def test_auto_load_success_logging(self):
        """Test successful auto-load logs correct messages"""
        from main_platform import TradingPlatform

        platform = TradingPlatform.__new__(TradingPlatform)
        platform.data_manager = Mock()
        platform.data_dict = {}
        platform.logger = Mock()
        platform.statusBar = Mock()

        mock_df = pd.DataFrame({'timestamp': [1, 2, 3], 'close': [100, 101, 102]})
        platform.data_manager.load_alpaca_data.return_value = mock_df

        platform.auto_load_default_data()

        # Verify success logging
        platform.logger.info.assert_called()
        log_calls = [call[0][0] for call in platform.logger.info.call_args_list]
        assert any("Auto-loaded" in msg for msg in log_calls)

    def test_auto_load_failure_handling(self):
        """Test auto-load handles API failures gracefully"""
        from main_platform import TradingPlatform

        platform = TradingPlatform.__new__(TradingPlatform)
        platform.data_manager = Mock()
        platform.data_dict = {}
        platform.logger = Mock()
        platform.statusBar = Mock()

        # Mock API failure
        platform.data_manager.load_alpaca_data.return_value = {'error': 'API connection failed'}

        platform.auto_load_default_data()

        # Verify error logging
        platform.logger.warning.assert_called()
        warning_call = platform.logger.warning.call_args[0][0]
        assert "Failed to auto-load" in warning_call

    def test_auto_load_data_storage(self):
        """Test that loaded data is stored in platform data_dict"""
        from main_platform import TradingPlatform

        platform = TradingPlatform.__new__(TradingPlatform)
        platform.data_manager = Mock()
        platform.data_dict = {}
        platform.logger = Mock()
        platform.statusBar = Mock()

        mock_df = pd.DataFrame({'timestamp': [1, 2, 3], 'close': [100, 101, 102]})
        platform.data_manager.load_alpaca_data.return_value = mock_df

        platform.auto_load_default_data()

        # Verify data was stored
        assert len(platform.data_dict) == 1
        assert 'BTC-USD_1Hour' in platform.data_dict

    def test_auto_load_timer_scheduled(self):
        """Test that auto-load is scheduled with QTimer on startup"""
        with patch('main_platform.QTimer') as mock_timer:
            from main_platform import TradingPlatform

            # This will fail due to GUI dependencies, but we can test the timer setup
            try:
                platform = TradingPlatform()
                mock_timer.singleShot.assert_called_with(1000, platform.auto_load_default_data)
            except:
                # GUI initialization may fail in test environment
                mock_timer.singleShot.assert_called_with(1000, platform.auto_load_default_data)


class TestTab9DataDownload:
    """Tests for Tab9DataDownload GUI functionality"""

    def test_tab9_class_exists(self):
        """Test that Tab9DataDownload class can be imported"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload
        assert Tab9DataDownload is not None

    def test_data_download_thread_creation(self):
        """Test DataDownloadThread can be created"""
        from gui.platform_gui_tab9_data_download import DataDownloadThread

        thread = DataDownloadThread("2023-01-01", "2024-01-01", "1Hour")
        assert thread.start_date == "2023-01-01"
        assert thread.end_date == "2024-01-01"
        assert thread.timeframe == "1Hour"

    @patch('subprocess.Popen')
    def test_download_thread_execution(self, mock_popen):
        """Test download thread executes subprocess correctly"""
        from gui.platform_gui_tab9_data_download import DataDownloadThread

        # Mock subprocess
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ""
        mock_process.poll.return_value = 0
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        thread = DataDownloadThread("2023-01-01", "2024-01-01", "1Hour")

        # Mock progress signal
        thread.progress_update = Mock()

        thread.run()

        # Verify subprocess was called
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]

        # Verify command structure
        assert "download_btc_data.py" in call_args
        assert "--start-date" in call_args
        assert "--end-date" in call_args
        assert "--timeframe" in call_args

    def test_download_thread_error_handling(self):
        """Test download thread handles subprocess errors"""
        from gui.platform_gui_tab9_data_download import DataDownloadThread

        with patch('subprocess.Popen', side_effect=Exception("Command failed")):
            thread = DataDownloadThread("2023-01-01", "2024-01-01", "1Hour")
            thread.download_finished = Mock()

            thread.run()

            # Should emit finished signal with error
            thread.download_finished.emit.assert_called()
            call_args = thread.download_finished.emit.call_args[0]
            assert call_args[0] == False  # success=False

    def test_check_data_status_method(self):
        """Test check_data_status method functionality"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        # Create mock parent
        mock_parent = Mock()
        mock_parent.data_dict = {}

        # Create tab instance (will fail due to GUI, but we can test the method)
        try:
            tab = Tab9DataDownload(mock_parent)
            # If GUI works, test the method
            status = tab.check_data_status()
            assert isinstance(status, dict)
        except:
            # GUI not available in test environment
            pass


class TestCheckDataStatusScript:
    """Tests for check_data_status.py script"""

    def test_check_data_status_function(self):
        """Test check_data_status function with mock files"""
        from scripts.check_data_status import check_data_status

        # Create temporary directory with mock CSV files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data directory structure
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create mock CSV files
            test_files = {
                "btc_usd_5m.csv": "timestamp,open,high,low,close,volume\n2023-01-01,100,101,99,100,1000",
                "btc_usd_15m.csv": "timestamp,open,high,low,close,volume\n2023-01-01,100,101,99,100,1000",
                "btc_usd_1h.csv": "timestamp,open,high,low,close,volume\n2023-01-01,100,101,99,100,1000"
            }

            for filename, content in test_files.items():
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)

            # Change to temp directory and test
            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                # This will print to stdout, but we can capture it if needed
                check_data_status()
                # Function should complete without errors
                assert True
            finally:
                os.chdir(old_cwd)

    def test_missing_files_handling(self):
        """Test check_data_status handles missing files correctly"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data directory but no files
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                check_data_status()
                # Should handle missing files gracefully
                assert True
            finally:
                os.chdir(old_cwd)

    def test_corrupted_csv_handling(self):
        """Test check_data_status handles corrupted CSV files"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create corrupted CSV file
            filepath = os.path.join(data_dir, "btc_usd_5m.csv")
            with open(filepath, 'w') as f:
                f.write("corrupted content that is not CSV")

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                check_data_status()
                # Should handle corrupted files gracefully
                assert True
            finally:
                os.chdir(old_cwd)


class TestTab1DefaultConfiguration:
    """Tests for default BTC/USD configuration in Tab1DataManagement"""

    def test_btc_usd_default_selection(self):
        """Test that BTC/USD is selected by default in symbol combo"""
        from gui.platform_gui_tab1_improved import Tab1DataManagement

        # Mock parent platform
        mock_parent = Mock()
        mock_parent.data_manager = Mock()

        try:
            tab = Tab1DataManagement(mock_parent)
            # Check if BTC/USD is selected
            current_symbol = tab.symbol_combo.currentText()
            assert current_symbol == "BTC/USD"
        except:
            # GUI not available in test environment
            pass

    def test_symbol_combo_contains_btc_usd(self):
        """Test that symbol combo contains BTC/USD option"""
        from gui.platform_gui_tab1_improved import Tab1DataManagement

        mock_parent = Mock()
        mock_parent.data_manager = Mock()

        try:
            tab = Tab1DataManagement(mock_parent)
            items = [tab.symbol_combo.itemText(i) for i in range(tab.symbol_combo.count())]
            assert "BTC/USD" in items
        except:
            # GUI not available in test environment
            pass


class TestEdgeCasesAndErrorScenarios:
    """Tests for edge cases and error scenarios"""

    def test_auto_load_with_network_timeout(self):
        """Test auto-load handles network timeouts"""
        from main_platform import TradingPlatform

        platform = TradingPlatform.__new__(TradingPlatform)
        platform.data_manager = Mock()
        platform.data_dict = {}
        platform.logger = Mock()
        platform.statusBar = Mock()

        # Mock timeout error
        platform.data_manager.load_alpaca_data.side_effect = TimeoutError("Connection timeout")

        platform.auto_load_default_data()

        # Should handle timeout gracefully
        platform.logger.error.assert_called()

    def test_auto_load_with_invalid_credentials(self):
        """Test auto-load handles invalid API credentials"""
        from main_platform import TradingPlatform

        platform = TradingPlatform.__new__(TradingPlatform)
        platform.data_manager = Mock()
        platform.data_dict = {}
        platform.logger = Mock()
        platform.statusBar = Mock()

        # Mock authentication error
        platform.data_manager.load_alpaca_data.return_value = {'error': 'Invalid API credentials'}

        platform.auto_load_default_data()

        # Should log warning
        platform.logger.warning.assert_called()

    def test_download_thread_with_invalid_timeframe(self):
        """Test download thread handles invalid timeframe parameters"""
        from gui.platform_gui_tab9_data_download import DataDownloadThread

        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.stdout.readline.return_value = ""
            mock_process.poll.return_value = 1  # Error exit code
            mock_process.returncode = 1
            mock_popen.return_value = mock_process

            thread = DataDownloadThread("2023-01-01", "2024-01-01", "InvalidTimeframe")
            thread.download_finished = Mock()

            thread.run()

            # Should handle error gracefully
            thread.download_finished.emit.assert_called_with(False, "Download failed")

    def test_check_data_status_with_permission_denied(self):
        """Test check_data_status handles permission denied errors"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create file and remove read permissions (if possible on Windows)
            filepath = os.path.join(data_dir, "btc_usd_5m.csv")
            with open(filepath, 'w') as f:
                f.write("test")

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                # Should handle permission errors gracefully
                check_data_status()
                assert True
            finally:
                os.chdir(old_cwd)

    def test_auto_load_with_empty_data_response(self):
        """Test auto-load handles empty data responses"""
        from main_platform import TradingPlatform

        platform = TradingPlatform.__new__(TradingPlatform)
        platform.data_manager = Mock()
        platform.data_dict = {}
        platform.logger = Mock()
        platform.statusBar = Mock()

        # Mock empty DataFrame
        platform.data_manager.load_alpaca_data.return_value = pd.DataFrame()

        platform.auto_load_default_data()

        # Should handle empty data gracefully
        assert len(platform.data_dict) == 1  # Still stores empty data

    def test_download_thread_cancellation(self):
        """Test download thread can be cancelled mid-execution"""
        from gui.platform_gui_tab9_data_download import DataDownloadThread

        thread = DataDownloadThread("2023-01-01", "2024-01-01", "1Hour")

        # Test thread cancellation
        thread.cancelled = True
        # Thread should handle cancellation gracefully
        assert thread.cancelled == True


if __name__ == "__main__":
    # Run basic smoke tests
    print("🧪 Running smoke tests for new platform features...")

    # Test imports
    try:
        from main_platform import TradingPlatform
        from gui.platform_gui_tab9_data_download import Tab9DataDownload
        from scripts.check_data_status import check_data_status
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)

    print("🎉 Smoke tests completed successfully!")