#!/usr/bin/env python3
"""
TradingIA - Advanced Trading Platform
Main entry point for the BTC Trading Strategy Platform

Usage:
    python main.py --help

Author: TradingIA Team
Version: 1.0.0
"""

import argparse
import sys
import os
from pathlib import Path

# Add src and new modules to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent / 'api'))
sys.path.insert(0, str(Path(__file__).parent / 'utils'))

def main():
    parser = argparse.ArgumentParser(description='TradingIA - Advanced Trading Platform')
    parser.add_argument('--mode', choices=['gui', 'backtest', 'live', 'optimize'],
                       default='gui', help='Execution mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--strategy', type=str, help='Strategy name')
    parser.add_argument('--symbol', type=str, default='BTC/USD', help='Trading symbol')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.mode == 'gui':
        from src.main_platform import main as gui_main
        gui_main()
    elif args.mode == 'backtest':
        from core.execution.backtester_core import BacktesterCore
        # TODO: Implement CLI backtest
        print("Backtest mode not yet implemented")
    elif args.mode == 'live':
        from src.paper_trader import run_live_trading
        run_live_trading(args)
    elif args.mode == 'optimize':
        from src.optimization import run_optimization
        run_optimization(args)

if __name__ == '__main__':
    main()