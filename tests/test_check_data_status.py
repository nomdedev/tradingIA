#!/usr/bin/env python3
"""
Test Suite for check_data_status.py Script
=========================================

Tests for the data status checking utility script.
"""

import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, mock_open
from io import StringIO

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

class TestCheckDataStatusScript:
    """Tests for check_data_status.py script functionality"""

    def test_check_data_status_function_all_missing(self):
        """Test check_data_status with no files present"""
        import subprocess

        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                # Run the script directly
                result = subprocess.run([
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "..", "scripts", "check_data_status.py")
                ], capture_output=True, text=True, timeout=10)

                # Should complete successfully
                assert result.returncode == 0
                assert "Estado de Datos BTC/USD" in result.stdout

            finally:
                os.chdir(old_cwd)

    def test_check_data_status_function_with_files(self):
        """Test check_data_status with existing files"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data directory structure
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create test CSV files
            test_data = {
                "btc_usd_5m.csv": "timestamp,open,high,low,close,volume\n2023-01-01 00:00:00,100,101,99,100,1000\n2023-01-01 00:05:00,100,102,98,101,1100",
                "btc_usd_15m.csv": "timestamp,open,high,low,close,volume\n2023-01-01 00:00:00,100,101,99,100,1000",
                "btc_usd_1h.csv": "timestamp,open,high,low,close,volume\n2023-01-01 00:00:00,100,101,99,100,1000\n2023-01-01 01:00:00,100,102,98,101,1100\n2023-01-01 02:00:00,101,103,99,102,1200"
            }

            for filename, content in test_data.items():
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                # Should not raise exception and should print status
                check_data_status()
            finally:
                os.chdir(old_cwd)

    def test_check_data_status_function_mixed_files(self):
        """Test check_data_status with some files present, some missing"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create only some files
            test_data = {
                "btc_usd_5m.csv": "timestamp,open,high,low,close,volume\n2023-01-01 00:00:00,100,101,99,100,1000",
                "btc_usd_1h.csv": "timestamp,open,high,low,close,volume\n2023-01-01 00:00:00,100,101,99,100,1000"
            }

            for filename, content in test_data.items():
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                check_data_status()
            finally:
                os.chdir(old_cwd)

    def test_check_data_status_function_corrupted_csv(self):
        """Test check_data_status handles corrupted CSV files"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create corrupted CSV file
            filepath = os.path.join(data_dir, "btc_usd_5m.csv")
            with open(filepath, 'w') as f:
                f.write("This is not a valid CSV file\nIt has no proper structure")

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                # Should handle corrupted file gracefully
                check_data_status()
            finally:
                os.chdir(old_cwd)

    def test_check_data_status_function_empty_csv(self):
        """Test check_data_status handles empty CSV files"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create empty CSV file
            filepath = os.path.join(data_dir, "btc_usd_5m.csv")
            with open(filepath, 'w') as f:
                f.write("")  # Empty file

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                check_data_status()
            finally:
                os.chdir(old_cwd)

    def test_check_data_status_function_header_only_csv(self):
        """Test check_data_status handles CSV files with header only"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create CSV file with header only
            filepath = os.path.join(data_dir, "btc_usd_5m.csv")
            with open(filepath, 'w') as f:
                f.write("timestamp,open,high,low,close,volume")

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                check_data_status()
            finally:
                os.chdir(old_cwd)

    def test_check_data_status_function_large_file(self):
        """Test check_data_status handles large files efficiently"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create a moderately large CSV file (10,000 rows)
            filepath = os.path.join(data_dir, "btc_usd_5m.csv")
            with open(filepath, 'w') as f:
                f.write("timestamp,open,high,low,close,volume\n")
                for i in range(10000):
                    f.write(f"2023-01-01 {i:02d}:00:00,100,101,99,100,1000\n")

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                # Should handle large file without memory issues
                check_data_status()
            finally:
                os.chdir(old_cwd)

    def test_check_data_status_function_permission_denied(self):
        """Test check_data_status handles permission denied errors"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create file and make it unreadable (if possible)
            filepath = os.path.join(data_dir, "btc_usd_5m.csv")
            with open(filepath, 'w') as f:
                f.write("timestamp,open,high,low,close,volume\n2023-01-01 00:00:00,100,101,99,100,1000")

            # Try to remove read permission (may not work on Windows)
            try:
                os.chmod(filepath, 0o000)
            except:
                pass  # Skip if not supported

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                check_data_status()
            finally:
                os.chdir(old_cwd)
                try:
                    os.chmod(filepath, 0o644)  # Restore permissions
                except:
                    pass

    def test_check_data_status_function_unicode_content(self):
        """Test check_data_status handles Unicode content in files"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create CSV file with Unicode content
            filepath = os.path.join(data_dir, "btc_usd_5m.csv")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("timestamp,open,high,low,close,volume\n")
                f.write("2023-01-01 00:00:00,100.5,101.25,99.75,100.8,1000\n")

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                check_data_status()
            finally:
                os.chdir(old_cwd)

    def test_check_data_status_function_multiple_timeframes_complete(self):
        """Test check_data_status with all timeframes present"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create all timeframe files
            timeframes = ['5m', '15m', '1h', '4h']
            for tf in timeframes:
                filepath = os.path.join(data_dir, f"btc_usd_{tf}.csv")
                with open(filepath, 'w') as f:
                    f.write("timestamp,open,high,low,close,volume\n")
                    f.write("2023-01-01 00:00:00,100,101,99,100,1000\n")
                    f.write("2023-01-01 01:00:00,100,102,98,101,1100\n")

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                check_data_status()
            finally:
                os.chdir(old_cwd)

    def test_check_data_status_output_capture(self):
        """Test that check_data_status produces expected output"""
        from scripts.check_data_status import check_data_status

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, "data", "raw")
            os.makedirs(data_dir)

            # Create one test file
            filepath = os.path.join(data_dir, "btc_usd_5m.csv")
            with open(filepath, 'w') as f:
                f.write("timestamp,open,high,low,close,volume\n2023-01-01 00:00:00,100,101,99,100,1000")

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                # Capture stdout to verify output
                import io
                from contextlib import redirect_stdout

                f = io.StringIO()
                with redirect_stdout(f):
                    check_data_status()
                output = f.getvalue()

                # Verify expected content in output
                assert "Estado de Datos BTC/USD" in output
                assert "✅ 5m:" in output
                assert "📊 Registros: 1" in output
                assert "Resumen" in output
                assert "Archivos disponibles: 1/4" in output

            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    # Run basic smoke tests
    print("🧪 Running check_data_status smoke tests...")

    try:
        # Test direct script execution
        import subprocess
        result = subprocess.run([
            sys.executable,
            os.path.join(os.path.dirname(__file__), "..", "scripts", "check_data_status.py")
        ], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            print("✅ check_data_status script executes successfully")
        else:
            print(f"❌ check_data_status script failed: {result.stderr}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error running check_data_status: {e}")
        sys.exit(1)

    print("🎉 check_data_status smoke tests completed successfully!")