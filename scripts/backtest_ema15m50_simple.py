"""
Script Simplificado de Backtest Comparativo con Filtro EMA15m50

Compara el desempeño de las estrategias al agregar un filtro post-procesamiento
de proximidad a la EMA50 de 15 minutos, sin modificar las estrategias originales.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import talib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy
from strategies.vp_ifvg_ema_strategy import VPIFVGEmaStrategy  
from backtest_optimized_2to1 import OptimizedBacktester


def calculate_ema15m50(df: pd.DataFrame) -> pd.Series:
    """
    Calculate EMA50 on 15-minute timeframe and align to 5-minute bars
    
    Args:
        df: DataFrame with 5-minute OHLCV data and DatetimeIndex
        
    Returns:
        Series with EMA50 values aligned to 5m bars (forward filled)
    """
    # Resample to 15 minutes
    df_15m = df.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Calculate EMA50 on 15m data
    ema50_15m = talib.EMA(df_15m['close'].values, timeperiod=50)
    df_15m['ema50'] = ema50_15m
    
    # Align back to 5m timeframe using forward fill
    ema50_5m = df_15m['ema50'].reindex(df.index, method='ffill')
    
    return ema50_5m


def apply_ema15m50_filter(df_with_signals: pd.DataFrame, proximity_pct: float = 2.0) -> pd.DataFrame:
    """
    Apply EMA15m50 proximity filter to existing signals
    
    Args:
        df_with_signals: DataFrame with signals already generated
        proximity_pct: Percentage proximity threshold (default 2%)
        
    Returns:
        DataFrame with filtered signals
    """
    # Calculate 15m EMA50
    df_with_signals['ema15m50'] = calculate_ema15m50(df_with_signals)
    
    # Calculate distance to EMA
    df_with_signals['ema15m50_distance_pct'] = (
        abs(df_with_signals['close'] - df_with_signals['ema15m50']) / 
        df_with_signals['ema15m50'] * 100
    )
    
    # Mark bars near EMA
    df_with_signals['near_ema15m50'] = (
        df_with_signals['ema15m50_distance_pct'] <= proximity_pct
    )
    
    return df_with_signals


def run_backtest_with_strategy(strategy_class, strategy_name: str, df: pd.DataFrame,
                               apply_ema_filter: bool = False, initial_capital: float = 10000) -> dict:
    """
    Run backtest with a strategy, optionally applying EMA15m50 filter
    
    Args:
        strategy_class: Strategy class to instantiate
        strategy_name: Name for reporting
        df: DataFrame with OHLCV data
        apply_ema_filter: Whether to apply EMA15m50 proximity filter
        initial_capital: Starting capital
        
    Returns:
        Dictionary with backtest results
    """
    print(f"\n{'='*80}")
    print(f"Ejecutando: {strategy_name}")
    print(f"{'='*80}\n")
    
    # Initialize strategy
    strategy = strategy_class()
    strategy.set_parameters({
        'min_adx_entry': 15,
        'min_squeeze_momentum': 0.15,
        'stop_loss_atr_mult': 1.0,
        'take_profit_atr_mult': 3.0,
    })
    
    # Initialize and run backtester
    backtester = OptimizedBacktester(
        strategy=strategy,
        initial_capital=int(initial_capital)
    )
    
    # Set EMA filter flag for backtester
    backtester.use_ema_filter = apply_ema_filter
    
    # Run backtest
    backtester.run_optimized_backtest(df)
    
    # Calculate metrics
    metrics = backtester._calculate_optimized_metrics(target_rr_ratio=3.0)
    
    # Prepare results
    results = {
        'strategy_name': strategy_name,
        'total_trades': metrics.get('total_trades', 0),
        'win_rate': metrics.get('win_rate', 0),
        'profit_factor': metrics.get('profit_factor', 0),
        'total_return': metrics.get('total_return_pct', 0),
        'max_drawdown': metrics.get('max_drawdown', 0),
        'expectancy': metrics.get('mathematical_expectancy', 0),
        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
        'ema_filter_applied': apply_ema_filter,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Print summary
    print(f"\nResultados:")
    print(f"   - Total Trades: {results['total_trades']}")
    print(f"   - Win Rate: {results['win_rate']:.1f}%")
    print(f"   - Profit Factor: {results['profit_factor']:.2f}")
    print(f"   - Total Return: {results['total_return']:.2f}%")
    print(f"   - Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"   - Expectancy: {results['expectancy']:.4f}")
    print(f"   - Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    return results


def compare_results(original_results: dict, enhanced_results: dict) -> dict:
    """Compare results between original and enhanced strategy"""
    comparison = {
        'original': original_results['strategy_name'],
        'enhanced': enhanced_results['strategy_name'],
        'improvements': {}
    }
    
    metrics = [
        ('total_trades', 'Total Trades'),
        ('win_rate', 'Win Rate (%)'),
        ('profit_factor', 'Profit Factor'),
        ('total_return', 'Total Return (%)'),
        ('max_drawdown', 'Max Drawdown (%)'),
        ('expectancy', 'Expectancy'),
        ('sharpe_ratio', 'Sharpe Ratio'),
    ]
    
    print(f"\n{'='*80}")
    print(f"COMPARACION")
    print(f"{'='*80}\n")
    
    for metric_key, metric_name in metrics:
        original_val = original_results.get(metric_key, 0)
        enhanced_val = enhanced_results.get(metric_key, 0)
        
        change = enhanced_val - original_val
        change_pct = (change / abs(original_val) * 100) if original_val != 0 else 0
        
        # For drawdown, lower is better
        if metric_key == 'max_drawdown':
            improvement = change < 0
        else:
            improvement = change > 0
        
        arrow = "+" if improvement else "-"
        
        print(f"{arrow} {metric_name}:")
        print(f"   Original: {original_val:.4f}")
        print(f"   Enhanced: {enhanced_val:.4f}")
        print(f"   Change: {change:+.4f} ({change_pct:+.2f}%)")
        
        comparison['improvements'][metric_key] = {
            'original': original_val,
            'enhanced': enhanced_val,
            'change': change,
            'change_pct': change_pct,
            'improved': improvement
        }
    
    # Overall assessment
    improved_count = sum(1 for v in comparison['improvements'].values() if v['improved'])
    total_metrics = len(comparison['improvements'])
    improvement_pct = improved_count / total_metrics * 100
    
    print(f"\nRESUMEN: {improved_count}/{total_metrics} metricas mejoradas ({improvement_pct:.1f}%)\n")
    
    comparison['summary'] = {
        'improved_metrics': improved_count,
        'total_metrics': total_metrics,
        'improvement_percentage': improvement_pct
    }
    
    return comparison


def load_data(symbol: str = 'BTC', timeframe: str = '5m') -> pd.DataFrame:
    """Load data for backtest"""
    data_file = project_root / f'data/btc_{timeframe.replace("m", "Min")}.csv'
    
    if not data_file.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {data_file}")
    
    print(f"Cargando datos: {data_file}")
    df = pd.read_csv(data_file)
    
    # Normalize column names
    df.columns = df.columns.str.lower()
    
    # Handle timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'unnamed: 0' in df.columns:
        df['timestamp'] = pd.to_datetime(df['unnamed: 0'])
        df = df.set_index('timestamp')
        df = df.drop(columns=['unnamed: 0'])
    else:
        df.index = pd.to_datetime(df.index)
    
    # Use last 5000 bars
    df = df.tail(5000)
    
    print(f"Datos cargados: {len(df)} barras")
    print(f"Periodo: {df.index[0]} a {df.index[-1]}\n")
    
    return df


def save_results(all_results: dict, output_dir: Path):
    """Save all results to files"""
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    json_file = output_dir / 'ema15m50_simple_comparison.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Resultados guardados: {json_file}")
    
    # Create summary table
    summary_data = []
    for key, results in all_results.items():
        if key.endswith('_comparison'):
            continue
        summary_data.append({
            'Strategy': results['strategy_name'],
            'Trades': results['total_trades'],
            'Win Rate (%)': f"{results['win_rate']:.1f}",
            'Profit Factor': f"{results['profit_factor']:.2f}",
            'Return (%)': f"{results['total_return']:.2f}",
            'Max DD (%)': f"{results['max_drawdown']:.2f}",
            'Expectancy': f"{results['expectancy']:.4f}",
            'Sharpe': f"{results['sharpe_ratio']:.2f}",
            'EMA Filter': 'Yes' if results['ema_filter_applied'] else 'No'
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_file = output_dir / 'ema15m50_simple_comparison.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"Tabla guardada: {csv_file}")
    
    return json_file, csv_file


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("BACKTEST COMPARATIVO: Filtro EMA15m50 (Simplificado)")
    print("="*80 + "\n")
    
    # Load data
    df = load_data()
    
    # Results storage
    all_results = {}
    
    # Test Squeeze+ADX+TTM
    print("\n" + "="*80)
    print("TEST 1: Squeeze+ADX+TTM Strategy")
    print("="*80)
    
    squeeze_original = run_backtest_with_strategy(
        SqueezeMomentumADXTTMStrategy,
        "Squeeze+ADX+TTM (Original)",
        df.copy(),
        apply_ema_filter=False
    )
    all_results['squeeze_original'] = squeeze_original
    
    squeeze_enhanced = run_backtest_with_strategy(
        SqueezeMomentumADXTTMStrategy,
        "Squeeze+ADX+TTM+EMA15m50",
        df.copy(),
        apply_ema_filter=True
    )
    all_results['squeeze_enhanced'] = squeeze_enhanced
    
    squeeze_comparison = compare_results(squeeze_original, squeeze_enhanced)
    all_results['squeeze_comparison'] = squeeze_comparison
    
    # Save results
    output_dir = project_root / 'results' / 'ema15m50_enhancement'
    json_file, csv_file = save_results(all_results, output_dir)
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETADO")
    print("="*80)
    print(f"\nArchivos generados:")
    print(f"   - {json_file}")
    print(f"   - {csv_file}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
