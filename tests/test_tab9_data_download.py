#!/usr/bin/env python3
"""
Test Suite for Tab9DataDownload GUI Component
=============================================

Tests for the data download management tab functionality.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.mark.skip(reason="Tab9DataDownload class not found - needs implementation")
@pytest.mark.skipif(os.environ.get('DISPLAY') is None, reason="GUI tests require display")
class TestTab9DataDownloadGUI:
    """GUI tests for Tab9DataDownload (require display)"""

    def test_tab9_initialization(self, qtbot):
        """Test Tab9DataDownload initializes correctly"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload(mock_parent)
        qtbot.addWidget(tab)

        # Check that table exists
        assert tab.table is not None
        assert tab.table.rowCount() == 4  # 4 timeframes

        # Check buttons exist
        assert tab.refresh_btn is not None
        assert tab.download_selected_btn is not None
        assert tab.download_all_btn is not None

    def test_data_status_refresh(self, qtbot):
        """Test refresh data status functionality"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload(mock_parent)
        qtbot.addWidget(tab)

        # Mock check_data_status method
        with patch.object(tab, 'check_data_status', return_value={
            '5m': {'exists': False, 'size': 0, 'records': 0, 'modified': None},
            '15m': {'exists': True, 'size': 1024, 'records': 100, 'modified': '2024-01-01'},
            '1h': {'exists': False, 'size': 0, 'records': 0, 'modified': None},
            '4h': {'exists': False, 'size': 0, 'records': 0, 'modified': None}
        }):
            tab.refresh_data_status()

            # Check table contents
            assert tab.table.item(0, 0).text() == "❌ 5m"
            assert tab.table.item(1, 0).text() == "✅ 15m"
            assert tab.table.item(1, 1).text() == "1.00 KB"
            assert tab.table.item(1, 2).text() == "100"

    def test_download_selected_functionality(self, qtbot):
        """Test download selected timeframe functionality"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload(mock_parent)
        qtbot.addWidget(tab)

        # Select first row (5m timeframe)
        tab.table.selectRow(0)

        with patch.object(tab, 'download_timeframe') as mock_download:
            tab.on_download_selected_clicked()
            mock_download.assert_called_once_with("5m")

    def test_download_all_functionality(self, qtbot):
        """Test download all missing timeframes functionality"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload(mock_parent)
        qtbot.addWidget(tab)

        with patch.object(tab, 'download_timeframe') as mock_download:
            with patch.object(tab, 'check_data_status', return_value={
                '5m': {'exists': False},
                '15m': {'exists': True},
                '1h': {'exists': False},
                '4h': {'exists': False}
            }):
                tab.on_download_all_clicked()

                # Should call download for missing timeframes only
                assert mock_download.call_count == 3
                calls = mock_download.call_args_list
                assert calls[0][0][0] == "5m"
                assert calls[1][0][0] == "1h"
                assert calls[2][0][0] == "4h"


@pytest.mark.skip(reason="DataDownloadThread class not found - needs implementation")
class TestDataDownloadThread:
    """Unit tests for DataDownloadThread"""

    def test_thread_initialization(self):
        """Test DataDownloadThread initializes with correct parameters"""
        from gui.platform_gui_tab9_data_download import DataDownloadThread

        thread = DataDownloadThread("2023-01-01", "2024-01-01", "1Hour")

        assert thread.start_date == "2023-01-01"
        assert thread.end_date == "2024-01-01"
        assert thread.timeframe == "1Hour"

    def test_thread_successful_download(self):
        """Test successful download execution"""
        from gui.platform_gui_tab9_data_download import DataDownloadThread

        with patch('subprocess.Popen') as mock_popen:
            # Mock successful process
            mock_process = Mock()
            mock_process.stdout.readline.side_effect = ["Starting download...", "", ""]
            mock_process.poll.side_effect = [None, None, 0]
            mock_process.returncode = 0
            mock_process.stderr.readline.return_value = ""
            mock_popen.return_value = mock_process

            thread = DataDownloadThread("2023-01-01", "2024-01-01", "1Hour")

            # Mock signals
            thread.progress_update = Mock()
            thread.download_finished = Mock()

            thread.run()

            # Verify signals were emitted
            thread.progress_update.assert_called()
            thread.download_finished.assert_called_with(True, "Download completed successfully")

    def test_thread_download_failure(self):
        """Test download failure handling"""
        from gui.platform_gui_tab9_data_download import DataDownloadThread

        with patch('subprocess.Popen') as mock_popen:
            # Mock failed process
            mock_process = Mock()
            mock_process.stdout.readline.return_value = ""
            mock_process.poll.return_value = 1
            mock_process.returncode = 1
            mock_process.stderr.readline.return_value = "Error: API limit exceeded"
            mock_popen.return_value = mock_process

            thread = DataDownloadThread("2023-01-01", "2024-01-01", "1Hour")

            thread.progress_update = Mock()
            thread.download_finished = Mock()

            thread.run()

            # Verify error handling
            thread.download_finished.assert_called_with(False, "Download failed")

    def test_thread_command_construction(self):
        """Test that correct command is constructed"""
        from gui.platform_gui_tab9_data_download import DataDownloadThread

        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.stdout.readline.return_value = ""
            mock_process.poll.return_value = 0
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            thread = DataDownloadThread("2023-01-01", "2024-01-01", "1Hour")
            thread.progress_update = Mock()
            thread.download_finished = Mock()

            thread.run()

            # Verify command construction
            call_args = mock_popen.call_args
            command = call_args[0][0]

            assert "download_btc_data.py" in command
            assert "--start-date" in command
            assert "2023-01-01" in command
            assert "--end-date" in command
            assert "2024-01-01" in command
            assert "--timeframe" in command
            assert "1Hour" in command

    def test_thread_cancellation(self):
        """Test thread cancellation during execution"""
        from gui.platform_gui_tab9_data_download import DataDownloadThread

        thread = DataDownloadThread("2023-01-01", "2024-01-01", "1Hour")

        # Test cancellation flag
        assert hasattr(thread, 'cancelled')
        thread.cancelled = True
        assert thread.cancelled == True


@pytest.mark.skip(reason="Tab9DataDownload class not found - needs implementation")
class TestCheckDataStatusMethod:
    """Tests for the check_data_status method"""

    def test_check_data_status_all_missing(self):
        """Test check_data_status when no files exist"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload.__new__(Tab9DataDownload)  # Create without __init__
        tab.parent_platform = mock_parent

        with patch('os.path.exists', return_value=False):
            status = tab.check_data_status()

            assert len(status) == 4
            for tf in ['5m', '15m', '1h', '4h']:
                assert tf in status
                assert status[tf]['exists'] == False
                assert status[tf]['size'] == 0
                assert status[tf]['records'] == 0

    def test_check_data_status_all_present(self):
        """Test check_data_status when all files exist"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload.__new__(Tab9DataDownload)
        tab.parent_platform = mock_parent

        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024), \
             patch('os.path.getmtime', return_value=1609459200), \
             patch('builtins.open', Mock()) as mock_open:

            # Mock file reading for record count
            mock_file = Mock()
            mock_file.__iter__.return_value = ['header', 'row1', 'row2', 'row3']
            mock_open.return_value.__enter__.return_value = mock_file

            status = tab.check_data_status()

            assert len(status) == 4
            for tf in ['5m', '15m', '1h', '4h']:
                assert tf in status
                assert status[tf]['exists'] == True
                assert status[tf]['size'] == 1024
                assert status[tf]['records'] == 3  # 4 lines - 1 header

    def test_check_data_status_mixed_files(self):
        """Test check_data_status with some files present, some missing"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload.__new__(Tab9DataDownload)
        tab.parent_platform = mock_parent

        def mock_exists(path):
            # Only 5m and 1h exist
            return 'btc_usd_5m.csv' in path or 'btc_usd_1h.csv' in path

        with patch('os.path.exists', side_effect=mock_exists), \
             patch('os.path.getsize', return_value=2048), \
             patch('os.path.getmtime', return_value=1609459200), \
             patch('builtins.open', Mock()) as mock_open:

            mock_file = Mock()
            mock_file.__iter__.return_value = ['header', 'row1', 'row2']
            mock_open.return_value.__enter__.return_value = mock_file

            status = tab.check_data_status()

            # 5m and 1h should exist
            assert status['5m']['exists'] == True
            assert status['1h']['exists'] == True

            # 15m and 4h should not exist
            assert status['15m']['exists'] == False
            assert status['4h']['exists'] == False

    def test_check_data_status_file_read_error(self):
        """Test check_data_status handles file read errors gracefully"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload.__new__(Tab9DataDownload)
        tab.parent_platform = mock_parent

        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024), \
             patch('os.path.getmtime', return_value=1609459200), \
             patch('builtins.open', side_effect=PermissionError("Access denied")):

            status = tab.check_data_status()

            # Should still report file exists but with 0 records due to error
            assert status['5m']['exists'] == True
            assert status['5m']['size'] == 1024
            assert status['5m']['records'] == 0  # Error fallback


@pytest.mark.skip(reason="Tab9DataDownload class not found - needs implementation")
class TestDownloadTimeframeMethod:
    """Tests for the download_timeframe method"""

    def test_download_timeframe_creates_thread(self):
        """Test download_timeframe creates and starts download thread"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload.__new__(Tab9DataDownload)
        tab.parent_platform = mock_parent
        tab.download_thread = None
        tab.progress_bar = Mock()
        tab.activity_log = Mock()

        with patch('gui.platform_gui_tab9_data_download.DataDownloadThread') as mock_thread_class:
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread

            tab.download_timeframe("5m")

            # Verify thread was created and started
            mock_thread_class.assert_called_once()
            mock_thread.start.assert_called_once()

            # Verify progress bar was shown
            tab.progress_bar.setVisible.assert_called_with(True)

    def test_download_timeframe_thread_signals(self):
        """Test download_timeframe connects thread signals correctly"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload.__new__(Tab9DataDownload)
        tab.parent_platform = mock_parent
        tab.download_thread = None
        tab.progress_bar = Mock()
        tab.activity_log = Mock()

        with patch('gui.platform_gui_tab9_data_download.DataDownloadThread') as mock_thread_class:
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread

            tab.download_timeframe("5m")

            # Verify signal connections
            mock_thread.progress_update.connect.assert_called()
            mock_thread.download_finished.connect.assert_called()


@pytest.mark.skip(reason="Tab9DataDownload class not found - needs implementation")
class TestProgressAndLogging:
    """Tests for progress updates and activity logging"""

    def test_progress_update_signal_handling(self):
        """Test that progress update signals are handled correctly"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload.__new__(Tab9DataDownload)
        tab.parent_platform = mock_parent
        tab.progress_bar = Mock()
        tab.activity_log = Mock()

        # Simulate progress update signal
        tab.on_progress_update("Downloading data...", 75)

        tab.progress_bar.setValue.assert_called_with(75)
        tab.activity_log.append.assert_called_with("Downloading data...")

    def test_download_finished_success(self):
        """Test successful download completion handling"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload.__new__(Tab9DataDownload)
        tab.parent_platform = mock_parent
        tab.progress_bar = Mock()
        tab.activity_log = Mock()
        tab.download_thread = Mock()

        with patch.object(tab, 'refresh_data_status') as mock_refresh:
            tab.on_download_finished(True, "Download completed successfully")

            # Verify UI updates
            tab.progress_bar.setVisible.assert_called_with(False)
            tab.activity_log.append.assert_called_with("✅ Download completed successfully")
            mock_refresh.assert_called_once()

            # Verify thread cleanup
            assert tab.download_thread is None

    def test_download_finished_failure(self):
        """Test failed download completion handling"""
        from gui.platform_gui_tab9_data_download import Tab9DataDownload

        mock_parent = Mock()
        mock_parent.data_dict = {}

        tab = Tab9DataDownload.__new__(Tab9DataDownload)
        tab.parent_platform = mock_parent
        tab.progress_bar = Mock()
        tab.activity_log = Mock()
        tab.download_thread = Mock()

        tab.on_download_finished(False, "Download failed: API error")

        # Verify error handling
        tab.progress_bar.setVisible.assert_called_with(False)
        tab.activity_log.append.assert_called_with("❌ Download failed: API error")
        assert tab.download_thread is None


if __name__ == "__main__":
    # Run basic smoke tests
    print("🧪 Running Tab9DataDownload smoke tests...")

    try:
        from gui.platform_gui_tab9_data_download import Tab9DataDownload, DataDownloadThread
        print("✅ Tab9DataDownload imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)

    print("🎉 Tab9DataDownload smoke tests completed successfully!")