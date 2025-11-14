#!/usr/bin/env python3
"""
BTC Trading System Runner
=========================

Main entry point for executable and CLI modes.

Modes:
- monitor: Live trading with 24/7 monitoring
- backtest: Historical backtesting
- opt: Parameter optimization
- sensitivity: Parameter sensitivity analysis

Usage:
    python run_trading.py --mode=monitor --symbol=BTCUSD
    python run_trading.py --mode=backtest --start=2023-01-01 --end=2024-01-01
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.optimizer import StrategyOptimizer
from src.backtester import AdvancedBacktester
from src.mtf_data_handler import MultiTFDataHandler

def setup_logging(mode: str, log_level: str = 'INFO'):
    """Setup logging configuration"""

    # Create logs directory
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)

    # Setup logging
    log_file = log_dir / f'trading_{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting BTC Trading System - Mode: {mode}")
    logger.info(f"📁 Log file: {log_file}")

    return logger

def run_monitor_mode(symbol: str = 'BTCUSD', logger=None):
    """Run live monitoring mode"""

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"👀 Starting monitor mode for {symbol}")

    # Initialize components
    data_handler = MultiTFDataHandler()

    # Monitor loop
    check_interval = int(os.getenv('MONITOR_INTERVAL', '60'))

    while True:
        try:
            logger.info("🔍 Checking market conditions...")

            # Get latest data (last 24h for monitoring)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)

            dfs = data_handler.get_multi_tf_data(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            if dfs:
                logger.info(f"✅ Data loaded: {list(dfs.keys())}")

                # Here you would implement live trading logic
                # For now, just log market status

                latest_price = dfs['entry']['Close'].iloc[-1]
                logger.info(f"💰 Latest {symbol} price: ${latest_price:.2f}")

            else:
                logger.warning("⚠️ No data available")

            # Wait for next check
            logger.info(f"⏰ Waiting {check_interval} seconds...")
            time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("🛑 Monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Monitor error: {e}")
            time.sleep(30)  # Wait before retry

def run_backtest_mode(start_date: str, end_date: str, symbol: str = 'BTCUSD', logger=None):
    """Run backtesting mode"""

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"📊 Running backtest for {symbol}: {start_date} to {end_date}")

    # Initialize components
    data_handler = MultiTFDataHandler()
    backtester = AdvancedBacktester()

    try:
        # Load data
        dfs = data_handler.get_multi_tf_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        if not dfs:
            logger.error("❌ No data available for backtest")
            return

        logger.info(f"✅ Data loaded: {list(dfs.keys())}")

        # Run backtest with default parameters
        params = {
            'atr_multi': 1.5,
            'va_percent': 0.7,
            'vp_rows': 120,
            'vol_thresh': 1.2,
            'tp_rr': 2.2,
            'min_confidence': 0.6,
            'ema_fast_5m': 18,
            'ema_slow_5m': 48,
            'ema_fast_15m': 18,
            'ema_slow_15m': 48,
            'ema_fast_1h': 95,
            'ema_slow_1h': 210
        }

        result = backtester.run_optimized_backtest(dfs, params)

        # Log results
        metrics = result['metrics']
        logger.info("📈 Backtest Results:")
        logger.info(f"   Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
        logger.info(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
        logger.info(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"   Total Trades: {metrics.get('total_trades', 0)}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = project_root / 'results' / f'backtest_{symbol}_{timestamp}.json'

        with open(results_file, 'w') as f:
            import json
            json.dump({
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'params': params,
                'metrics': metrics,
                'timestamp': timestamp
            }, f, indent=2, default=str)

        logger.info(f"💾 Results saved to: {results_file}")

    except Exception as e:
        logger.error(f"❌ Backtest error: {e}")
        raise

def run_optimization_mode(symbol: str = 'BTCUSD', method: str = 'bayes', logger=None):
    """Run parameter optimization"""

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"🎯 Running optimization: {method} for {symbol}")

    # Initialize components
    data_handler = MultiTFDataHandler()
    optimizer = StrategyOptimizer()

    try:
        # Load data (last 6 months for optimization)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        dfs = data_handler.get_multi_tf_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if not dfs:
            logger.error("❌ No data available for optimization")
            return

        logger.info(f"✅ Data loaded: {list(dfs.keys())}")

        if method == 'bayes':
            # Bayesian optimization
            param_bounds = [(1.0, 2.0), (0.8, 1.5), (1.5, 2.5)]  # atr_multi, vol_thresh, tp_rr
            result = optimizer.bayes_opt_sharpe(dfs, param_bounds=param_bounds, n_calls=20)

            logger.info("🎯 Optimization Results:")
            logger.info(f"   Best Score: {result['best_score']:.3f}")
            logger.info(f"   Best Params: {result['best_params']}")

        elif method == 'frontier':
            # Efficient frontier
            param_ranges = {'atr_multi': (1.0, 2.0), 'vol_thresh': (0.8, 1.5), 'tp_rr': (1.5, 2.5)}
            risks, sharpes, param_combinations = optimizer.efficient_frontier_params(dfs, param_ranges, n_points=10)

            best_idx = np.argmax(sharpes)
            logger.info("🎯 Best Frontier Point:")
            logger.info(f"   Sharpe: {sharpes[best_idx]:.3f}")
            logger.info(f"   Risk: {risks[best_idx]:.1%}")
            logger.info(f"   Params: {param_combinations[best_idx]}")

        else:
            logger.error(f"❌ Unknown optimization method: {method}")

    except Exception as e:
        logger.error(f"❌ Optimization error: {e}")
        raise

def run_sensitivity_mode(symbol: str = 'BTCUSD', logger=None):
    """Run parameter sensitivity analysis"""

    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"🔍 Running sensitivity analysis for {symbol}")

    # Initialize components
    data_handler = MultiTFDataHandler()
    optimizer = StrategyOptimizer()

    try:
        # Load data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        dfs = data_handler.get_multi_tf_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if not dfs:
            logger.error("❌ No data available for sensitivity analysis")
            return

        logger.info(f"✅ Data loaded: {list(dfs.keys())}")

        # Run sensitivity analysis
        results = optimizer.param_sensitivity_heatmap(dfs)

        logger.info("✅ Sensitivity analysis completed")
        logger.info("📊 Generated heatmaps for: " + ", ".join(results.keys()))

    except Exception as e:
        logger.error(f"❌ Sensitivity analysis error: {e}")
        raise

def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description='BTC Trading System Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live monitoring
  python run_trading.py --mode=monitor --symbol=BTCUSD

  # Backtesting
  python run_trading.py --mode=backtest --start=2023-01-01 --end=2024-01-01

  # Optimization
  python run_trading.py --mode=opt --method=bayes

  # Sensitivity analysis
  python run_trading.py --mode=sensitivity
        """
    )

    parser.add_argument(
        '--mode',
        choices=['monitor', 'backtest', 'opt', 'sensitivity'],
        default='monitor',
        help='Operation mode'
    )

    parser.add_argument(
        '--symbol',
        default='BTCUSD',
        help='Trading symbol (default: BTCUSD)'
    )

    parser.add_argument(
        '--start',
        help='Start date for backtest (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        help='End date for backtest (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--method',
        choices=['bayes', 'frontier'],
        default='bayes',
        help='Optimization method'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.mode, args.log_level)

    try:
        if args.mode == 'monitor':
            run_monitor_mode(args.symbol, logger)

        elif args.mode == 'backtest':
            if not args.start or not args.end:
                parser.error("--start and --end dates required for backtest mode")
            run_backtest_mode(args.start, args.end, args.symbol, logger)

        elif args.mode == 'opt':
            run_optimization_mode(args.symbol, args.method, logger)

        elif args.mode == 'sensitivity':
            run_sensitivity_mode(args.symbol, logger)

        logger.info("✅ Operation completed successfully")

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()