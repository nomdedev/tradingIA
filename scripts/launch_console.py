#!/usr/bin/env python3
"""
BTC Trading Platform - Console Version
========================================

Versión de consola sin GUI para cuando PyQt6 no está disponible.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

def main_console():
    """Main console interface"""
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  BTC Trading Strategy Platform - Console Mode".center(58) + "║")
    print("║" + "  Backend Components Active".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    
    print("\n📊 Loading components...")
    
    try:
        from backend_core import DataManager, StrategyEngine
        from backtester_core import BacktesterCore
        from analysis_engines import AnalysisEngines
        
        print("✅ DataManager loaded")
        print("✅ StrategyEngine loaded")
        print("✅ BacktesterCore loaded")
        print("✅ AnalysisEngines loaded")
        
        print("\n" + "─" * 60)
        print("🎯 Available Options:")
        print("─" * 60)
        print("1. Run Backtest")
        print("2. Optimize Strategy")
        print("3. Analyze Market Data")
        print("4. View Configuration")
        print("5. Exit")
        print("─" * 60)
        
        while True:
            choice = input("\n👉 Select option (1-5): ").strip()
            
            if choice == '1':
                print("\n🔄 Starting backtest...")
                print("💡 Configure parameters in config/trading_config.yaml")
                print("⚠️  Feature in development")
                
            elif choice == '2':
                print("\n🔍 Starting optimization...")
                print("💡 This will test multiple parameter combinations")
                print("⚠️  Feature in development")
                
            elif choice == '3':
                print("\n📈 Analyzing market data...")
                print("💡 Regime detection and causality tests")
                print("⚠️  Feature in development")
                
            elif choice == '4':
                print("\n⚙️  Current Configuration:")
                print("   API: Alpaca Markets")
                print("   Symbol: BTC/USD")
                print("   Strategy: Multi-timeframe IFVG")
                
            elif choice == '5':
                print("\n👋 Exiting platform...")
                break
            else:
                print("❌ Invalid option. Please select 1-5")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Error loading components: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == '__main__':
    try:
        success = main_console()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Platform closed by user")
        sys.exit(0)
