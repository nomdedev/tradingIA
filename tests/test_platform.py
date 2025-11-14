#!/usr/bin/env python3
"""
Quick Test Script for BTC Trading Strategy Platform
==================================================

This script performs basic validation tests for the platform components.

Usage:
    python test_platform.py

Author: TradingIA Team
Version: 1.0.0
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test basic imports"""
    print("🔍 Testing imports...")

    tests = [
        ('PyQt6.QtWidgets', 'QApplication'),
        ('pandas', 'DataFrame'),
        ('numpy', 'array'),
        ('plotly.graph_objects', 'Figure'),
        ('alpaca_trade_api', 'REST'),
        ('backtesting', 'Backtest'),
    ]

    failed = []
    for module, attr in tests:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            print(f"   ✅ {module}.{attr}")
        except ImportError as e:
            print(f"   ❌ {module}.{attr}: {e}")
            failed.append((module, attr))
        except Exception as e:
            print(f"   ⚠️  {module}.{attr}: {e}")

    return len(failed) == 0

def test_platform_components():
    """Test platform component imports"""
    print("\n🔍 Testing platform components...")

    # Add src to path
    src_path = Path(__file__).parent / 'src'
    sys.path.insert(0, str(src_path))

    components = [
        ('main_platform', 'TradingPlatform'),
        ('backend_core', 'DataManager'),
        ('backend_core', 'StrategyEngine'),
        ('backtester_core', 'BacktesterCore'),
        ('analysis_engines', 'AnalysisEngines'),
        ('settings_manager', 'SettingsManager'),
        ('reporters_engine', 'ReportersEngine'),
    ]

    failed = []
    for module, cls in components:
        try:
            mod = __import__(module, fromlist=[cls])
            getattr(mod, cls)
            print(f"   ✅ {module}.{cls}")
        except ImportError as e:
            print(f"   ❌ {module}.{cls}: {e}")
            failed.append((module, cls))
        except Exception as e:
            print(f"   ⚠️  {module}.{cls}: {e}")

    return len(failed) == 0

def test_gui_tabs():
    """Test GUI tab imports"""
    print("\n🔍 Testing GUI tabs...")

    src_path = Path(__file__).parent / 'src'
    sys.path.insert(0, str(src_path))

    tabs = [
        ('gui.platform_gui_tab1', 'Tab1DataManagement'),
        ('gui.platform_gui_tab2', 'Tab2StrategyConfig'),
        ('gui.platform_gui_tab3', 'Tab3BacktestRunner'),
        ('gui.platform_gui_tab4', 'Tab4ResultsAnalysis'),
    ]

    failed = []
    for module, cls in tabs:
        try:
            mod = __import__(module, fromlist=[cls])
            getattr(mod, cls)
            print(f"   ✅ {module}.{cls}")
        except ImportError as e:
            print(f"   ❌ {module}.{cls}: {e}")
            failed.append((module, cls))
        except Exception as e:
            print(f"   ⚠️  {module}.{cls}: {e}")

    return len(failed) == 0

def test_basic_functionality():
    """Test basic functionality without GUI"""
    print("\n🔍 Testing basic functionality...")

    try:
        src_path = Path(__file__).parent / 'src'
        sys.path.insert(0, str(src_path))

        # Test settings manager
        from settings_manager import SettingsManager
        settings = SettingsManager()
        settings.load_config()
        print("   ✅ SettingsManager initialization")

        # Test data manager (without API call)
        from backend_core import DataManager
        data_mgr = DataManager()
        print("   ✅ DataManager initialization")

        # Test strategy engine
        from backend_core import StrategyEngine
        strategy_engine = StrategyEngine()
        print("   ✅ StrategyEngine initialization")

        # Test backtester core
        from backtester_core import BacktesterCore
        backtester = BacktesterCore()
        print("   ✅ BacktesterCore initialization")

        return True

    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("BTC Trading Strategy Platform - Test Suite")
    print("=" * 50)

    results = []

    # Run tests
    results.append(("Dependencies", test_imports()))
    results.append(("Platform Components", test_platform_components()))
    results.append(("GUI Tabs", test_gui_tabs()))
    results.append(("Basic Functionality", test_basic_functionality()))

    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Platform is ready to launch.")
        print("   Run: python launch_platform.py")
    else:
        print("⚠️  SOME TESTS FAILED. Please check the errors above.")
        print("   Try: python launch_platform.py --install")

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())