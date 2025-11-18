"""
Backtesting Script for Squeeze ADX TTM Strategy
Compares performance with and without multi-timeframe parameters
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy
from scripts.parameter_importance_analyzer import run_parameter_importance_analysis
from scripts.multitimeframe_impact_analyzer import run_multitimeframe_analysis
from src.data_fetcher import DataFetcher


def load_market_data(symbol: str = "BTC/USD", timeframe: str = "5Min",
                    start_date: str = "2024-01-01",
                    end_date: str = "2025-11-12") -> pd.DataFrame:
    """
    Load market data for backtesting

    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        start_date: Start date for data
        end_date: End date for data

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Loading {symbol} data for {timeframe} timeframe...")

    # Try to load from local data first
    data_file = f"data/btc_{timeframe.lower()}.csv"

    if os.path.exists(data_file):
        print(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)

        # Convert timestamp if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        # Ensure proper column names
        df.columns = df.columns.str.lower()
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns in {data_file}")
            return pd.DataFrame()

        return df

    # If local data not available, try to fetch
    print("Local data not found, attempting to fetch from API...")
    try:
        fetcher = DataFetcher()
        df = fetcher.fetch_crypto_data(symbol, timeframe, start_date, end_date)

        if not df.empty:
            # Save for future use
            os.makedirs('data', exist_ok=True)
            df.to_csv(data_file)
            print(f"Data saved to {data_file}")

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def create_multi_tf_data(df_5min: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create multi-timeframe data dictionary

    Args:
        df_5min: 5-minute data

    Returns:
        Dict with multiple timeframes
    """
    multi_tf_data = {'5Min': df_5min}

    # Create 15-minute data
    df_15min = df_5min.resample('15Min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    multi_tf_data['15Min'] = df_15min

    # Create 1-hour data
    df_1h = df_5min.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    multi_tf_data['1H'] = df_1h

    return multi_tf_data


def run_comparative_backtest(strategy: SqueezeMomentumADXTTMStrategy,
                           df_multi_tf: Dict[str, pd.DataFrame]) -> Dict:
    """
    Run comparative backtest with and without multi-timeframe features

    Args:
        strategy: The strategy to test
        df_multi_tf: Multi-timeframe market data

    Returns:
        Dict with comparison results
    """
    print("\n" + "="*60)
    print("COMPARATIVE BACKTEST: Squeeze ADX TTM Strategy")
    print("="*60)

    base_params = strategy.get_parameters()

    # Test 1: Without multi-timeframe confirmation
    print("\n1. Testing WITHOUT multi-timeframe confirmation...")
    strategy.set_parameters({**base_params, 'higher_tf_weight': 0.0, 'lower_tf_weight': 0.0})
    signals_no_mtf = strategy.generate_signals(df_multi_tf)['signals']

    # Calculate performance metrics for no multi-TF
    returns_no_mtf = calculate_returns(signals_no_mtf, df_multi_tf['5Min']['close'])
    metrics_no_mtf = calculate_performance_metrics(returns_no_mtf)

    print("   Signals generated: {}".format(len(signals_no_mtf[signals_no_mtf != 0])))
    print(".4f")
    print(".2%")
    print(".2f")

    # Test 2: With multi-timeframe confirmation
    print("\n2. Testing WITH multi-timeframe confirmation...")
    strategy.set_parameters(base_params)  # Reset to default parameters
    signals_with_mtf = strategy.generate_signals(df_multi_tf)['signals']

    # Calculate performance metrics for with multi-TF
    returns_with_mtf = calculate_returns(signals_with_mtf, df_multi_tf['5Min']['close'])
    metrics_with_mtf = calculate_performance_metrics(returns_with_mtf)

    print("   Signals generated: {}".format(len(signals_with_mtf[signals_with_mtf != 0])))
    print(".4f")
    print(".2%")
    print(".2f")

    # Calculate improvement
    improvement = {
        'total_return': metrics_with_mtf['total_return'] - metrics_no_mtf['total_return'],
        'win_rate': metrics_with_mtf['win_rate'] - metrics_no_mtf['win_rate'],
        'sharpe_ratio': metrics_with_mtf['sharpe_ratio'] - metrics_no_mtf['sharpe_ratio'],
        'max_drawdown': metrics_with_mtf['max_drawdown'] - metrics_no_mtf['max_drawdown'],
        'signal_reduction': len(signals_no_mtf[signals_no_mtf != 0]) - len(signals_with_mtf[signals_with_mtf != 0])
    }

    print("\n3. IMPROVEMENT ANALYSIS:")
    print(".4f")
    print(".2%")
    print(".2f")
    print(".2f")
    print("   Signal reduction: {} ({:.1%})".format(
        improvement['signal_reduction'],
        improvement['signal_reduction'] / len(signals_no_mtf[signals_no_mtf != 0]) if len(signals_no_mtf[signals_no_mtf != 0]) > 0 else 0
    ))

    results = {
        'without_multi_tf': {
            'metrics': metrics_no_mtf,
            'signal_count': len(signals_no_mtf[signals_no_mtf != 0])
        },
        'with_multi_tf': {
            'metrics': metrics_with_mtf,
            'signal_count': len(signals_with_mtf[signals_with_mtf != 0])
        },
        'improvement': improvement,
        'test_period': {
            'start': df_multi_tf['5Min'].index.min(),
            'end': df_multi_tf['5Min'].index.max(),
            'total_bars': len(df_multi_tf['5Min'])
        }
    }

    return results


def calculate_returns(signals: pd.Series, prices: pd.Series) -> pd.Series:
    """Calculate returns from signals and prices"""
    returns = pd.Series(0.0, index=signals.index)
    position = 0
    entry_price = 0

    for i, signal in enumerate(signals):
        if signal == 1 and position == 0:  # Buy signal
            position = 1
            entry_price = prices.iloc[i]
        elif signal == -1 and position == 1:  # Sell signal
            if entry_price > 0:
                ret = (prices.iloc[i] - entry_price) / entry_price
                returns.iloc[i] = ret
            position = 0
            entry_price = 0

    return returns


def calculate_performance_metrics(returns: pd.Series) -> Dict:
    """Calculate comprehensive performance metrics"""
    if returns.empty or len(returns[returns != 0]) == 0:
        return {
            'total_return': 0,
            'win_rate': 0,
            'total_trades': 0,
            'avg_trade': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'profit_factor': 0
        }

    trade_returns = returns[returns != 0]

    # Basic metrics
    total_return = trade_returns.sum()
    win_rate = (trade_returns > 0).mean()
    total_trades = len(trade_returns)
    avg_trade = trade_returns.mean()

    # Sharpe ratio (assuming daily returns, adjust as needed)
    if len(trade_returns) > 1:
        # Sharpe con risk-free rate
        rf_daily = 0.04 / 252
        excess_returns = trade_returns - rf_daily
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0
    else:
        sharpe_ratio = 0

    # Maximum drawdown
    cumulative = (1 + trade_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Profit factor
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'avg_trade': avg_trade,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor
    }


def save_backtest_results(results: Dict, filename: str = None) -> None:
    """Save backtest results to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"squeeze_adx_ttm_backtest_{timestamp}.json"

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=convert_numpy)

    print(f"\nBacktest results saved to {filename}")


def main():
    """Main backtesting function"""
    print("Squeeze ADX TTM Strategy Backtesting")
    print("=" * 50)

    # Load market data
    df_5min = load_market_data()

    if df_5min.empty:
        print("Error: No market data available. Please check data sources.")
        return

    print(f"Loaded {len(df_5min)} bars of 5-minute data")

    # Create multi-timeframe data
    df_multi_tf = create_multi_tf_data(df_5min)
    print(f"Created multi-timeframe data: {list(df_multi_tf.keys())}")

    # Initialize strategy
    strategy = SqueezeMomentumADXTTMStrategy()
    print(f"Initialized strategy: {strategy.name}")

    # Run comparative backtest
    backtest_results = run_comparative_backtest(strategy, df_multi_tf)

    # Save results
    save_backtest_results(backtest_results)

    # Optional: Run parameter importance analysis
    run_full_analysis = input("\n¿Desea ejecutar análisis completo de importancia de parámetros? (y/n): ").lower().strip()
    if run_full_analysis == 'y':
        print("\nEjecutando análisis de importancia de parámetros...")
        run_parameter_importance_analysis(strategy, df_multi_tf)

    # Optional: Run multi-timeframe impact analysis
    run_mtf_analysis = input("\n¿Desea ejecutar análisis de impacto multi-timeframe? (y/n): ").lower().strip()
    if run_mtf_analysis == 'y':
        print("\nEjecutando análisis de impacto multi-timeframe...")
        run_multitimeframe_analysis(strategy, df_5min)

    print("\n" + "="*60)
    print("BACKTESTING COMPLETED")
    print("="*60)
    print("\nResumen:")
    print("- Estrategia probada: Squeeze ADX TTM")
    print("- Datos utilizados: BTC/USD 5Min con confirmación 15Min/1H")
    print("- Parámetros evaluados: Multi-timeframe confirmation")
    print("- Resultados guardados en archivos JSON y reportes")


if __name__ == "__main__":
    main()