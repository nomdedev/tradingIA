"""
Main Entry Point for BTC IFVG Backtesting System
Command-line interface for backtesting, paper trading, and optimization
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config.config import (
    get_config, validate_config, TRADING_CONFIG, 
    BACKTEST_CONFIG, LOGGING_CONFIG
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'logs' / 'trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_backtest(args):
    """Run backtesting mode"""
    logger.info("🔄 Starting Backtest Mode")
    logger.info("=" * 60)
    
    try:
        # Import here to avoid circular dependencies
        from src.data_fetcher import DataFetcher
        from src.indicators import calculate_all_indicators
        # from src.backtester import Backtester  # Will create next
        
        # Fetch data
        fetcher = DataFetcher()
        logger.info(f"Fetching data for {args.symbol} ({args.timeframe})")
        logger.info(f"Period: {args.start} to {args.end}")
        
        # Get multi-timeframe data
        dfs = fetcher.get_multi_tf_data(args.symbol)
        
        if not dfs or '5Min' not in dfs or dfs['5Min'].empty:
            logger.error("❌ No data fetched, aborting backtest")
            return
        
        df_5min = dfs['5Min']
        df_15min = dfs.get('15Min')
        df_1h = dfs.get('1H')
        
        logger.info(f"Data loaded: 5Min={len(df_5min)} bars")
        
        # Calculate indicators
        logger.info("Calculating indicators...")
        df_signals = calculate_all_indicators(df_5min, df_15min, df_1h)
        
        signals = df_signals[df_signals['signal'] != 'hold']
        logger.info(f"📊 Generated {len(signals)} trading signals")
        
        # Run backtest
        # backtester = Backtester(
        #     df_signals,
        #     initial_capital=BACKTEST_CONFIG['initial_capital'],
        #     risk_per_trade=BACKTEST_CONFIG['risk_per_trade']
        # )
        # results = backtester.run_backtest()
        
        logger.info("✅ Backtest completed!")
        # logger.info(f"Final Capital: ${results['final_capital']:,.2f}")
        # logger.info(f"Win Rate: {results['win_rate']*100:.2f}%")
        # logger.info(f"Sharpe Ratio: {results['sharpe']:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()


def run_paper_trading(args):
    """Run paper trading mode"""
    logger.info("📈 Starting Paper Trading Mode")
    logger.info("=" * 60)
    
    try:
        # from src.paper_trader import PaperTrader  # Will create next
        
        logger.info(f"Symbol: {args.symbol}")
        logger.info(f"Capital: ${args.capital:,.2f}")
        logger.info("Press Ctrl+C to stop...")
        
        # paper_trader = PaperTrader(symbol=args.symbol, capital=args.capital)
        # paper_trader.start()
        
        logger.info("Paper trading not yet implemented")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Paper trading stopped by user")
    except Exception as e:
        logger.error(f"❌ Paper trading failed: {e}")
        import traceback
        traceback.print_exc()


def run_optimization(args):
    """Run parameter optimization"""
    logger.info("⚙️ Starting Optimization Mode")
    logger.info("=" * 60)
    
    try:
        # from src.optimization import GridSearch  # Will create next
        
        logger.info(f"Walk-forward periods: {args.periods}")
        logger.info("This may take several minutes...")
        
        # optimizer = GridSearch(periods=args.periods)
        # best_params = optimizer.optimize()
        
        # logger.info("✅ Optimization completed!")
        # logger.info(f"Best parameters: {best_params}")
        
        logger.info("Optimization not yet implemented")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


def run_dashboard(args):
    """Run Streamlit dashboard"""
    logger.info("📊 Starting Dashboard")
    logger.info("=" * 60)
    
    import subprocess
    import os
    
    dashboard_path = Path(__file__).parent / 'src' / 'dashboard.py'
    
    if not dashboard_path.exists():
        logger.error(f"❌ Dashboard not found at {dashboard_path}")
        logger.info("Dashboard not yet implemented")
        return
    
    try:
        os.system(f'streamlit run {dashboard_path}')
    except Exception as e:
        logger.error(f"❌ Dashboard failed: {e}")


def main():
    """Main entry point with argument parsing"""
    
    # ASCII Art Banner
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║  BTC IFVG Backtesting System                              ║
    ║  Fair Value Gaps + Volume Profile + EMAs                  ║
    ║  Version 1.0 - November 2025                              ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    parser = argparse.ArgumentParser(
        description='BTC IFVG Backtesting and Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest
  python main.py --mode backtest --start 2023-01-01 --end 2025-11-12
  
  # Paper trading
  python main.py --mode paper --symbol BTCUSD --capital 10000
  
  # Optimize parameters
  python main.py --mode optimize --periods 6
  
  # Launch dashboard
  python main.py --mode dashboard
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['backtest', 'paper', 'optimize', 'dashboard'],
        help='Operation mode'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default=TRADING_CONFIG['symbol'],
        help='Trading symbol (default: BTCUSD)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='5Min',
        choices=['1Min', '5Min', '15Min', '1H'],
        help='Primary timeframe (default: 5Min)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default=TRADING_CONFIG['start_date'],
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=TRADING_CONFIG['end_date'],
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=BACKTEST_CONFIG['initial_capital'],
        help='Initial capital for trading (default: 10000)'
    )
    
    parser.add_argument(
        '--periods',
        type=int,
        default=6,
        help='Number of periods for walk-forward optimization (default: 6)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate configuration
    try:
        validate_config()
        logger.info("✅ Configuration validated")
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        logger.error("Please check your .env file and config/config.py")
        sys.exit(1)
    
    # Create logs directory
    logs_dir = Path(__file__).parent / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Route to appropriate mode
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.mode == 'backtest':
        run_backtest(args)
    elif args.mode == 'paper':
        run_paper_trading(args)
    elif args.mode == 'optimize':
        run_optimization(args)
    elif args.mode == 'dashboard':
        run_dashboard(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
