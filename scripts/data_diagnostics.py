"""
Data Loading Diagnostic Script
Comprehensive diagnostic tool for data loading issues in the trading platform
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_environment():
    """Check environment and dependencies"""
    print("🔍 ENVIRONMENT DIAGNOSTICS")
    print("=" * 50)

    # Check Python version
    print(f"Python Version: {sys.version}")

    # Check if we're in virtual environment
    in_venv = sys.prefix != sys.base_prefix
    print(f"Virtual Environment: {'✅ Active' if in_venv else '❌ Not active'}")

    # Check key directories
    dirs_to_check = [
        'data/raw',
        'data/cache',
        'config',
        'logs',
        'reports/sessions'
    ]

    print("\n📁 DIRECTORY STATUS:")
    for dir_path in dirs_to_check:
        full_path = project_root / dir_path
        exists = full_path.exists()
        print(f"  {dir_path}: {'✅' if exists else '❌'} ({full_path})")
        if exists:
            try:
                contents = list(full_path.glob('*'))
                print(f"    Contains {len(contents)} items")
            except:
                print("    Cannot read contents")

    # Check API credentials
    print("\n🔑 API CREDENTIALS:")
    try:
        from config.mtf_config import ALPACA_CONFIG
        has_key = bool(ALPACA_CONFIG.get('api_key'))
        has_secret = bool(ALPACA_CONFIG.get('secret_key'))
        print(f"  Alpaca API Key: {'✅' if has_key else '❌'}")
        print(f"  Alpaca Secret: {'✅' if has_secret else '❌'}")
        print(f"  Alpaca Base URL: {ALPACA_CONFIG.get('base_url', 'Not set')}")
    except Exception as e:
        print(f"  ❌ Cannot load Alpaca config: {e}")

    # Check data fetcher
    print("\n📡 DATA FETCHER STATUS:")
    try:
        from api.data_fetcher import DataFetcher
        print("  DataFetcher import: ✅")
        try:
            fetcher = DataFetcher()
            print("  DataFetcher initialization: ✅")
        except Exception as e:
            print(f"  DataFetcher initialization: ❌ ({e})")
    except ImportError as e:
        print(f"  DataFetcher import: ❌ ({e})")

    # Check data manager
    try:
        from core.backend_core import DataManager
        print("  DataManager import: ✅")
    except ImportError as e:
        print(f"  DataManager import: ❌ ({e})")

def check_data_files():
    """Check existing data files"""
    print("\n📊 DATA FILES DIAGNOSTICS")
    print("=" * 50)

    data_dirs = [
        project_root / 'data' / 'raw',
        project_root / 'data' / 'cache'
    ]

    for data_dir in data_dirs:
        print(f"\n🔍 Checking {data_dir}:")
        if not data_dir.exists():
            print(f"  ❌ Directory does not exist")
            continue

        csv_files = list(data_dir.glob('*.csv'))
        json_files = list(data_dir.glob('*.json'))

        print(f"  CSV files: {len(csv_files)}")
        print(f"  JSON files: {len(json_files)}")

        # Check BTC data files
        expected_files = [
            'btc_usd_5m.csv', 'btc_usd_15m.csv',
            'btc_usd_1h.csv', 'btc_usd_4h.csv'
        ]

        for filename in expected_files:
            filepath = data_dir / filename
            if filepath.exists():
                try:
                    size = filepath.stat().st_size
                    print(f"  ✅ {filename}: {size:,} bytes")
                except:
                    print(f"  ✅ {filename}: exists (size unknown)")
            else:
                print(f"  ❌ {filename}: missing")

def test_data_loading():
    """Test actual data loading functionality"""
    print("\n🧪 DATA LOADING TESTS")
    print("=" * 50)

    # Test DataManager
    print("Testing DataManager...")
    try:
        from core.backend_core import DataManager
        dm = DataManager()
        print("  ✅ DataManager created")

        # Try to load data
        print("  Testing data load...")
        result = dm.load_alpaca_data(
            symbol='BTC/USD',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1Hour'
        )

        if isinstance(result, dict) and 'error' in result:
            print(f"  ❌ Data load failed: {result['error']}")
        elif hasattr(result, 'shape'):
            print(f"  ✅ Data loaded: {result.shape[0]} rows, {result.shape[1]} columns")
            print(f"     Columns: {list(result.columns)}")
        else:
            print(f"  ❓ Unexpected result type: {type(result)}")

    except Exception as e:
        print(f"  ❌ DataManager test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test DataFetcher
    print("\nTesting DataFetcher...")
    try:
        from api.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        print("  ✅ DataFetcher created")

        # Try to get data
        print("  Testing data fetch...")
        df = fetcher.get_historical_data(
            symbol='BTC/USD',
            timeframe='1Hour',
            start_date='2024-01-01',
            end_date='2024-01-02'
        )

        if df is not None and len(df) > 0:
            print(f"  ✅ Data fetched: {len(df)} rows")
            print(f"     Date range: {df.index.min()} to {df.index.max()}")
        else:
            print("  ❌ No data returned")

    except Exception as e:
        print(f"  ❌ DataFetcher test failed: {e}")
        import traceback
        traceback.print_exc()

def check_recent_logs():
    """Check recent log files for errors"""
    print("\n📋 RECENT LOGS ANALYSIS")
    print("=" * 50)

    log_dir = project_root / 'logs'
    if not log_dir.exists():
        print("❌ Logs directory does not exist")
        return

    log_files = list(log_dir.glob('*.log'))
    if not log_files:
        print("❌ No log files found")
        return

    # Check most recent log file
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    print(f"📄 Analyzing latest log: {latest_log.name}")

    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-50:]  # Last 50 lines

        errors = [line for line in lines if 'ERROR' in line.upper() or 'EXCEPTION' in line.upper()]
        warnings = [line for line in lines if 'WARNING' in line.upper() or 'WARN' in line.upper()]

        print(f"  Recent errors: {len(errors)}")
        print(f"  Recent warnings: {len(warnings)}")

        if errors:
            print("\n🚨 RECENT ERRORS:")
            for error in errors[-5:]:  # Show last 5
                print(f"    {error.strip()}")

    except Exception as e:
        print(f"❌ Cannot read log file: {e}")

def generate_diagnostic_report():
    """Generate comprehensive diagnostic report"""
    print("\n📊 GENERATING DIAGNOSTIC REPORT")
    print("=" * 50)

    report = {
        'timestamp': datetime.now().isoformat(),
        'diagnostics': {
            'environment': {},
            'data_files': {},
            'loading_tests': {},
            'logs': {}
        },
        'recommendations': []
    }

    # Environment info
    report['diagnostics']['environment'] = {
        'python_version': sys.version,
        'virtual_env': sys.prefix != sys.base_prefix,
        'project_root': str(project_root)
    }

    # Check for missing components
    missing_components = []

    # Check data files
    raw_dir = project_root / 'data' / 'raw'
    expected_files = ['btc_usd_5m.csv', 'btc_usd_15m.csv', 'btc_usd_1h.csv', 'btc_usd_4h.csv']
    missing_files = []

    for filename in expected_files:
        if not (raw_dir / filename).exists():
            missing_files.append(filename)

    if missing_files:
        report['recommendations'].append({
            'issue': 'Missing data files',
            'description': f'The following data files are missing: {missing_files}',
            'solution': 'Run data download from the Data Download tab in the application'
        })

    # Check API credentials
    try:
        from config.mtf_config import ALPACA_CONFIG
        if not ALPACA_CONFIG.get('api_key') or not ALPACA_CONFIG.get('secret_key'):
            report['recommendations'].append({
                'issue': 'Missing Alpaca API credentials',
                'description': 'Alpaca API key and/or secret are not configured',
                'solution': 'Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables'
            })
    except:
        report['recommendations'].append({
            'issue': 'Cannot load Alpaca configuration',
            'description': 'Unable to import Alpaca configuration from config.mtf_config',
            'solution': 'Check config/mtf_config.py file and environment variables'
        })

    # Save report
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)

    report_file = reports_dir / f'data_diagnostic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✅ Diagnostic report saved to: {report_file}")

    # Print recommendations
    if report['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec['issue']}: {rec['description']}")
            print(f"   Solution: {rec['solution']}")
    else:
        print("\n✅ No issues detected!")

def main():
    """Run complete diagnostics"""
    print("🔧 TRADING PLATFORM DATA DIAGNOSTICS")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Project Root: {project_root}")
    print()

    check_environment()
    check_data_files()
    test_data_loading()
    check_recent_logs()
    generate_diagnostic_report()

    print("\n" + "=" * 60)
    print("🎯 DIAGNOSTICS COMPLETE")
    print("Check the generated report in reports/ for detailed findings")

if __name__ == '__main__':
    main()