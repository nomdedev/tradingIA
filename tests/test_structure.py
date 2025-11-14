"""
Quick Test Script
Verify new project structure is working
"""

import sys
from pathlib import Path

print("🧪 Testing New Project Structure")
print("=" * 60)

# Test 1: Imports
print("\n1️⃣ Testing imports...")
try:
    from config.config import validate_config, TRADING_CONFIG
    print("  ✅ config.config imported")
    
    from src.data_fetcher import DataFetcher
    print("  ✅ src.data_fetcher imported")
    
    from src.indicators import calculate_ifvg_enhanced
    print("  ✅ src.indicators imported")
    
    print("  ✅ All imports successful!")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Configuration
print("\n2️⃣ Testing configuration...")
try:
    validate_config()
    print("  ✅ Configuration validated")
    print(f"  Symbol: {TRADING_CONFIG['symbol']}")
    print(f"  Timeframes: {TRADING_CONFIG['timeframes']}")
except ValueError as e:
    print(f"  ⚠️ Configuration warning: {e}")
    print("  (Expected if .env not configured)")
except Exception as e:
    print(f"  ❌ Configuration error: {e}")

# Test 3: Data Fetcher (without API call)
print("\n3️⃣ Testing Data Fetcher...")
try:
    # Just test initialization (will fail if no API keys, that's OK)
    try:
        fetcher = DataFetcher()
        print("  ✅ DataFetcher initialized")
    except ValueError as e:
        print(f"  ⚠️ DataFetcher needs API keys: {e}")
        print("  (Expected if .env not configured)")
except Exception as e:
    print(f"  ❌ DataFetcher error: {e}")

# Test 4: Indicators with sample data
print("\n4️⃣ Testing Indicators...")
try:
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='5min')
    prices = 45000 * (1 + np.random.normal(0.0001, 0.003, 500)).cumprod()
    
    df = pd.DataFrame({
        'Open': prices,
        'High': prices * (1 + np.abs(np.random.normal(0, 0.001, 500))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.001, 500))),
        'Close': prices,
        'Volume': np.random.uniform(100, 1000, 500)
    }, index=dates)
    
    print(f"  Sample data: {len(df)} bars")
    
    # Test IFVG
    params = {'atr_period': 14, 'atr_multi': 0.2, 'mitigation_lookback': 5, 'min_gap_size': 0.001}
    bull_signals, bear_signals, confidence_scores = calculate_ifvg_enhanced(df, params)
    
    bull_count = bull_signals.sum()
    bear_count = bear_signals.sum()
    
    print(f"  IFVG signals: {bull_count} bull, {bear_count} bear")
    print("  ✅ Indicators working!")
    
except Exception as e:
    print(f"  ❌ Indicators error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Directory structure
print("\n5️⃣ Testing directory structure...")
required_dirs = ['src', 'config', 'data', 'tests', 'results', 'docs']
project_root = Path(__file__).parent

for dir_name in required_dirs:
    dir_path = project_root / dir_name
    if dir_path.exists():
        print(f"  ✅ {dir_name}/ exists")
    else:
        print(f"  ⚠️ {dir_name}/ missing")

# Test 6: Required files
print("\n6️⃣ Testing required files...")
required_files = [
    'main.py',
    'requirements.txt',
    '.env.template',
    'src/__init__.py',
    'src/data_fetcher.py',
    'src/indicators.py',
    'src/backtester.py',
    'src/paper_trader.py',
    'src/dashboard.py',
    'src/optimization.py',
    'config/__init__.py',
    'config/config.py',
]

for file_name in required_files:
    file_path = project_root / file_name
    if file_path.exists():
        print(f"  ✅ {file_name}")
    else:
        print(f"  ❌ {file_name} missing")

# Summary
print("\n" + "=" * 60)
print("✅ Structure test completed!")
print("\n📋 Next steps:")
print("  1. Copy .env.template to .env and add your Alpaca credentials")
print("  2. Run: pip install -r requirements.txt")
print("  3. Test: python main.py --mode backtest --help")
print("  4. Run: python demo_sistema_completo.py (optional demo)")
