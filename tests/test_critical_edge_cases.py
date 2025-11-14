#!/usr/bin/env python3
"""
Test Crítico: Edge Case de API Timeout
=====================================

Este test demuestra uno de los edge cases más críticos identificados:
el manejo de timeouts de red durante la carga automática de datos.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_auto_load_api_timeout_edge_case():
    """
    Test CRÍTICO: Manejo de timeouts de red durante carga automática

    Este edge case puede causar:
    - Aplicación congelada esperando respuesta
    - Experiencia de usuario degradada
    - Pérdida de confianza en la funcionalidad automática
    """
    print("🧪 Testing CRITICAL edge case: API timeout during auto-load")

    from main_platform import TradingPlatform

    # Create platform instance without GUI
    platform = TradingPlatform.__new__(TradingPlatform)
    platform.data_manager = Mock()
    platform.data_dict = {}
    platform.logger = Mock()
    platform.statusBar = Mock()

    # Simulate network timeout (realistic scenario)
    import time
    def slow_timeout_response(*args, **kwargs):
        time.sleep(0.1)  # Simulate network delay
        raise TimeoutError("Connection timed out after 30 seconds")

    platform.data_manager.load_alpaca_data.side_effect = slow_timeout_response

    # Execute auto-load (this should handle timeout gracefully)
    start_time = time.time()
    platform.auto_load_default_data()
    end_time = time.time()

    # Verify timeout was handled within reasonable time (< 1 second for test)
    execution_time = end_time - start_time
    assert execution_time < 1.0, f"Auto-load took too long: {execution_time:.2f}s"

    # Verify error was logged appropriately
    platform.logger.error.assert_called_once()
    error_call = platform.logger.error.call_args[0][0]
    assert "auto_load_default_data" in error_call.lower()

    # Verify data_dict remains empty (no corrupted state)
    assert len(platform.data_dict) == 0

    # Verify status bar shows appropriate message (if implemented)
    if platform.statusBar().showMessage.called:
        status_call = platform.statusBar().showMessage.call_args
        # Should not show success message
        assert "auto-loaded" not in str(status_call).lower()

    print("✅ API timeout handled correctly")
    print(".2f")
    print(f"✅ Error logged: {error_call}")
    print("✅ Data integrity maintained")


def test_auto_load_rate_limit_edge_case():
    """
    Test CRÍTICO: Manejo de rate limits de API

    Alpaca API tiene límites de tasa que pueden causar errores 429.
    """
    print("\n🧪 Testing CRITICAL edge case: API rate limit")

    from main_platform import TradingPlatform

    platform = TradingPlatform.__new__(TradingPlatform)
    platform.data_manager = Mock()
    platform.data_dict = {}
    platform.logger = Mock()
    platform.statusBar = Mock()

    # Simulate rate limit error (HTTP 429)
    rate_limit_error = {'error': 'Too many requests. Rate limit exceeded. Try again in 60 seconds.'}
    platform.data_manager.load_alpaca_data.return_value = rate_limit_error

    platform.auto_load_default_data()

    # Verify rate limit was handled gracefully
    platform.logger.warning.assert_called_once()
    warning_call = platform.logger.warning.call_args[0][0]
    assert "rate limit" in warning_call.lower() or "too many requests" in warning_call.lower()

    # Verify no data was stored
    assert len(platform.data_dict) == 0

    print("✅ Rate limit handled correctly")
    print(f"✅ Warning logged: {warning_call}")


def test_auto_load_disk_full_edge_case():
    """
    Test CRÍTICO: Manejo de disco lleno durante carga automática
    """
    print("\n🧪 Testing CRITICAL edge case: Disk full during auto-load")

    from main_platform import TradingPlatform

    platform = TradingPlatform.__new__(TradingPlatform)
    platform.data_manager = Mock()
    platform.data_dict = {}
    platform.logger = Mock()
    platform.statusBar = Mock()

    # Simulate disk full by making data_dict raise error on assignment
    class DiskFullDict(dict):
        def __setitem__(self, key, value):
            raise OSError("No space left on device")

    platform.data_dict = DiskFullDict()

    platform.auto_load_default_data()

    # Verify disk full error was handled
    platform.logger.error.assert_called_once()
    error_call = platform.logger.error.call_args[0][0]
    assert "disk" in error_call.lower() or "space" in error_call.lower() or "device" in error_call.lower()

    print("✅ Disk full error handled correctly")
    print(f"✅ Error logged: {error_call}")


def test_download_thread_network_interruption():
    """
    Test CRÍTICO: Interrupción de red durante descarga en Tab9
    """
    print("\n🧪 Testing CRITICAL edge case: Network interruption during download")

    from gui.platform_gui_tab9_data_download import DataDownloadThread

    with patch('subprocess.Popen') as mock_popen:
        # Simulate network interruption during download
        mock_process = Mock()

        # Simulate process that fails with network error
        # First calls to readline return output, then empty string
        mock_process.stdout.readline.side_effect = [
            "Starting download...",
            "Downloaded 10%...",
            "Network error: Connection reset",
            ""  # Empty string signals end of output
        ]
        mock_process.stderr.read.return_value = "Connection reset by peer"

        # poll() returns None while process is running, then return code when done
        mock_process.poll.side_effect = [None, None, None, 1]  # Finally returns error code

        mock_popen.return_value = mock_process

        thread = DataDownloadThread("2023-01-01", "2024-01-01", "1Hour")
        thread.progress_update = Mock()
        thread.download_finished = Mock()

        # Mock the emit method for Qt signals
        thread.progress_update.emit = Mock()
        thread.download_finished.emit = Mock()

        thread.run()

    # Verify network error was handled
    thread.download_finished.emit.assert_called_once()
    call_args = thread.download_finished.emit.call_args[0]

    # Should indicate failure (first argument should be False)
    assert call_args[0] == False, f"Expected failure but got success={call_args[0]}"

    # Should contain error information (second argument)
    error_message = call_args[1]
    assert "error" in error_message.lower() or "failed" in error_message.lower(), f"Error message should indicate failure: {error_message}"

    print("✅ Network interruption handled correctly")
    print(f"✅ Download finished emitted with: success={call_args[0]}, message='{call_args[1]}'")


if __name__ == "__main__":
    print("🚨 CRITICAL EDGE CASES TEST SUITE")
    print("=" * 50)

    try:
        test_auto_load_api_timeout_edge_case()
        test_auto_load_rate_limit_edge_case()
        test_auto_load_disk_full_edge_case()
        test_download_thread_network_interruption()

        print("\n" + "=" * 50)
        print("🎉 ALL CRITICAL EDGE CASES PASSED!")
        print("✅ System is resilient to common failure scenarios")
        print("✅ User experience protected against critical errors")

    except Exception as e:
        print(f"\n❌ CRITICAL TEST FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)