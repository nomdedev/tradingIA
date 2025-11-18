"""
Optimización de Umbral de Proximidad a EMA15m50

Este script prueba diferentes umbrales de proximidad a la EMA50 de 15 minutos
para determinar qué tan cerca debe estar el precio para maximizar:
- Win Rate
- Profit Factor
- Expectancy matemática

Probará umbrales desde 0.5% hasta 5% en incrementos de 0.25%
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import talib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy
from backtest_optimized_2to1 import OptimizedBacktester


def calculate_ema15m50(df: pd.DataFrame) -> pd.Series:
    """Calculate EMA50 on 15-minute timeframe aligned to 5m bars"""
    df_15m = df.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    ema50_15m = talib.EMA(df_15m['close'].values, timeperiod=50)
    df_15m['ema50'] = ema50_15m
    
    ema50_5m = df_15m['ema50'].reindex(df.index, method='ffill')
    return ema50_5m


def apply_ema_proximity_filter_to_trades(df: pd.DataFrame, trades: list, 
                                        proximity_pct: float) -> tuple:
    """
    Filter trades post-execution based on EMA proximity
    
    Args:
        df: DataFrame with price data and EMA
        trades: List of trade dictionaries from backtester
        proximity_pct: Percentage proximity threshold
        
    Returns:
        (filtered_trades, stats_dict)
    """
    # Calculate EMA and proximity
    df['ema15m50'] = calculate_ema15m50(df)
    df['ema15m50_distance_pct'] = (
        abs(df['close'] - df['ema15m50']) / df['ema15m50'] * 100
    )
    df['near_ema15m50'] = df['ema15m50_distance_pct'] <= proximity_pct
    
    # Filter trades
    filtered_trades = []
    total_trades = len([t for t in trades if t['status'] == 'closed'])
    
    for trade in trades:
        if trade['status'] != 'closed':
            continue
            
        entry_idx = trade['entry_idx']
        
        # Check if entry occurred near EMA
        if entry_idx < len(df) and df.iloc[entry_idx]['near_ema15m50']:
            filtered_trades.append(trade)
    
    # Calculate stats
    bars_near_ema = df['near_ema15m50'].sum()
    bars_near_ema_pct = bars_near_ema / len(df) * 100
    trades_kept = len(filtered_trades)
    trades_kept_pct = (trades_kept / total_trades * 100) if total_trades > 0 else 0
    
    stats = {
        'total_original_trades': total_trades,
        'filtered_trades': trades_kept,
        'trades_kept_pct': trades_kept_pct,
        'bars_near_ema': bars_near_ema,
        'bars_near_ema_pct': bars_near_ema_pct,
        'proximity_threshold': proximity_pct
    }
    
    return filtered_trades, stats


def calculate_metrics_from_trades(trades: list, initial_capital: float = 10000) -> dict:
    """Calculate trading metrics from trade list"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'expectancy': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
    
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] < 0]
    
    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    total_profit = sum(t['pnl'] for t in winning_trades)
    total_loss = abs(sum(t['pnl'] for t in losing_trades))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    total_pnl = sum(t['pnl'] for t in trades)
    total_return = (total_pnl / initial_capital) * 100
    
    expectancy = sum(t['pnl'] for t in trades) / total_trades if total_trades > 0 else 0
    
    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0
    
    # Calculate consecutive wins/losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_streak = 0
    last_result = None
    
    for trade in trades:
        if trade['pnl'] > 0:
            if last_result == 'win':
                current_streak += 1
            else:
                current_streak = 1
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
            last_result = 'win'
        else:
            if last_result == 'loss':
                current_streak += 1
            else:
                current_streak = 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
            last_result = 'loss'
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses
    }


def run_backtest_with_proximity(df: pd.DataFrame, proximity_pct: float,
                                initial_capital: float = 10000) -> dict:
    """Run backtest and apply proximity filter"""
    print(f"\nProbando proximidad: {proximity_pct}%")
    
    # Initialize strategy
    strategy = SqueezeMomentumADXTTMStrategy()
    strategy.set_parameters({
        'min_adx_entry': 15,
        'min_squeeze_momentum': 0.15,
        'stop_loss_atr_mult': 1.0,
        'take_profit_atr_mult': 3.0,
    })
    
    # Initialize backtester
    backtester = OptimizedBacktester(
        strategy=strategy,
        initial_capital=int(initial_capital)
    )
    
    # Run backtest (without EMA filter in backtester)
    backtester.use_ema_filter = False
    backtester.run_optimized_backtest(df.copy())
    
    # Get all trades
    all_trades = [t for t in backtester.trades if t['status'] == 'closed']
    
    # Apply proximity filter post-execution
    filtered_trades, filter_stats = apply_ema_proximity_filter_to_trades(
        df.copy(), all_trades, proximity_pct
    )
    
    # Calculate metrics for filtered trades
    metrics = calculate_metrics_from_trades(filtered_trades, initial_capital)
    
    # Combine results
    results = {
        'proximity_pct': proximity_pct,
        **metrics,
        **filter_stats
    }
    
    print(f"   Trades: {results['total_trades']} ({results['trades_kept_pct']:.1f}% retenidos)")
    print(f"   Win Rate: {results['win_rate']:.1f}%")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Expectancy: {results['expectancy']:.2f}")
    
    return results


def plot_optimization_results(results_df: pd.DataFrame, output_dir: Path):
    """Create visualization of optimization results"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Optimización de Proximidad a EMA50 15m', fontsize=16, fontweight='bold')
    
    # 1. Win Rate vs Proximity
    ax1 = axes[0, 0]
    ax1.plot(results_df['proximity_pct'], results_df['win_rate'], 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=50, color='r', linestyle='--', alpha=0.3, label='50% WR')
    ax1.set_xlabel('Proximidad a EMA (%)')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rate vs Proximidad')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Mark best win rate
    best_wr_idx = results_df['win_rate'].idxmax()
    best_wr = results_df.loc[best_wr_idx]
    ax1.plot(best_wr['proximity_pct'], best_wr['win_rate'], 'r*', markersize=20, 
             label=f'Mejor: {best_wr["proximity_pct"]}%')
    
    # 2. Profit Factor vs Proximity
    ax2 = axes[0, 1]
    ax2.plot(results_df['proximity_pct'], results_df['profit_factor'], 'o-', 
             linewidth=2, markersize=8, color='green')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='Breakeven')
    ax2.set_xlabel('Proximidad a EMA (%)')
    ax2.set_ylabel('Profit Factor')
    ax2.set_title('Profit Factor vs Proximidad')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Mark best PF
    best_pf_idx = results_df['profit_factor'].idxmax()
    best_pf = results_df.loc[best_pf_idx]
    ax2.plot(best_pf['proximity_pct'], best_pf['profit_factor'], 'r*', markersize=20,
             label=f'Mejor: {best_pf["proximity_pct"]}%')
    
    # 3. Total Trades vs Proximity
    ax3 = axes[1, 0]
    ax3.plot(results_df['proximity_pct'], results_df['total_trades'], 'o-', 
             linewidth=2, markersize=8, color='orange')
    ax3.set_xlabel('Proximidad a EMA (%)')
    ax3.set_ylabel('Número de Trades')
    ax3.set_title('Trades Totales vs Proximidad')
    ax3.grid(True, alpha=0.3)
    
    # 4. Expectancy vs Proximity
    ax4 = axes[1, 1]
    ax4.plot(results_df['proximity_pct'], results_df['expectancy'], 'o-',
             linewidth=2, markersize=8, color='purple')
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Breakeven')
    ax4.set_xlabel('Proximidad a EMA (%)')
    ax4.set_ylabel('Expectancy')
    ax4.set_title('Expectancy vs Proximidad')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Mark best expectancy
    best_exp_idx = results_df['expectancy'].idxmax()
    best_exp = results_df.loc[best_exp_idx]
    ax4.plot(best_exp['proximity_pct'], best_exp['expectancy'], 'r*', markersize=20,
             label=f'Mejor: {best_exp["proximity_pct"]}%')
    
    # 5. Bars Near EMA vs Proximity
    ax5 = axes[2, 0]
    ax5.plot(results_df['proximity_pct'], results_df['bars_near_ema_pct'], 'o-',
             linewidth=2, markersize=8, color='brown')
    ax5.set_xlabel('Proximidad a EMA (%)')
    ax5.set_ylabel('% Barras Cerca de EMA')
    ax5.set_title('Tiempo Cerca de EMA vs Proximidad')
    ax5.grid(True, alpha=0.3)
    
    # 6. Composite Score
    ax6 = axes[2, 1]
    
    # Normalize metrics to 0-100 scale
    results_df_norm = results_df.copy()
    results_df_norm['wr_norm'] = results_df['win_rate']
    results_df_norm['pf_norm'] = np.clip(results_df['profit_factor'] * 50, 0, 100)
    results_df_norm['exp_norm'] = np.clip((results_df['expectancy'] / results_df['expectancy'].abs().max()) * 50 + 50, 0, 100)
    
    # Composite score: average of normalized metrics
    results_df_norm['composite_score'] = (
        results_df_norm['wr_norm'] * 0.4 + 
        results_df_norm['pf_norm'] * 0.4 + 
        results_df_norm['exp_norm'] * 0.2
    )
    
    ax6.plot(results_df_norm['proximity_pct'], results_df_norm['composite_score'], 'o-',
             linewidth=2, markersize=8, color='darkblue')
    ax6.set_xlabel('Proximidad a EMA (%)')
    ax6.set_ylabel('Score Compuesto')
    ax6.set_title('Score Compuesto vs Proximidad\n(40% WR + 40% PF + 20% Exp)')
    ax6.grid(True, alpha=0.3)
    
    # Mark best composite
    best_comp_idx = results_df_norm['composite_score'].idxmax()
    best_comp = results_df_norm.loc[best_comp_idx]
    ax6.plot(best_comp['proximity_pct'], best_comp['composite_score'], 'r*', markersize=20,
             label=f'Mejor: {best_comp["proximity_pct"]}%')
    ax6.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'ema_proximity_optimization.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nGrafico guardado: {plot_file}")
    plt.close()


def load_data() -> pd.DataFrame:
    """Load BTC 5m data"""
    data_file = project_root / 'data/btc_5Min.csv'
    
    print(f"Cargando datos: {data_file}")
    df = pd.read_csv(data_file)
    
    # Normalize columns
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


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("OPTIMIZACION DE PROXIMIDAD A EMA50 15m")
    print("="*80 + "\n")
    
    # Load data
    df = load_data()
    
    # Define proximity thresholds to test
    proximity_thresholds = np.arange(0.5, 5.25, 0.25)
    print(f"Probando {len(proximity_thresholds)} umbrales de proximidad:")
    print(f"Desde {proximity_thresholds[0]}% hasta {proximity_thresholds[-1]}%")
    print(f"Incremento: 0.25%\n")
    
    # Run optimization
    results = []
    
    for proximity_pct in proximity_thresholds:
        result = run_backtest_with_proximity(df.copy(), proximity_pct)
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find optimal thresholds
    print("\n" + "="*80)
    print("RESULTADOS DE OPTIMIZACION")
    print("="*80 + "\n")
    
    # Best Win Rate
    best_wr_row = results_df.loc[results_df['win_rate'].idxmax()]
    print(f"MEJOR WIN RATE:")
    print(f"   Proximidad: {best_wr_row['proximity_pct']}%")
    print(f"   Win Rate: {best_wr_row['win_rate']:.2f}%")
    print(f"   Profit Factor: {best_wr_row['profit_factor']:.2f}")
    print(f"   Total Trades: {best_wr_row['total_trades']}")
    print(f"   Expectancy: {best_wr_row['expectancy']:.2f}")
    
    # Best Profit Factor
    best_pf_row = results_df.loc[results_df['profit_factor'].idxmax()]
    print(f"\nMEJOR PROFIT FACTOR:")
    print(f"   Proximidad: {best_pf_row['proximity_pct']}%")
    print(f"   Profit Factor: {best_pf_row['profit_factor']:.2f}")
    print(f"   Win Rate: {best_pf_row['win_rate']:.2f}%")
    print(f"   Total Trades: {best_pf_row['total_trades']}")
    print(f"   Expectancy: {best_pf_row['expectancy']:.2f}")
    
    # Best Expectancy
    best_exp_row = results_df.loc[results_df['expectancy'].idxmax()]
    print(f"\nMEJOR EXPECTANCY:")
    print(f"   Proximidad: {best_exp_row['proximity_pct']}%")
    print(f"   Expectancy: {best_exp_row['expectancy']:.2f}")
    print(f"   Win Rate: {best_exp_row['win_rate']:.2f}%")
    print(f"   Profit Factor: {best_exp_row['profit_factor']:.2f}")
    print(f"   Total Trades: {best_exp_row['total_trades']}")
    
    # Composite score
    results_df['wr_norm'] = results_df['win_rate']
    results_df['pf_norm'] = np.clip(results_df['profit_factor'] * 50, 0, 100)
    results_df['exp_norm'] = np.clip((results_df['expectancy'] / results_df['expectancy'].abs().max()) * 50 + 50, 0, 100)
    results_df['composite_score'] = (
        results_df['wr_norm'] * 0.4 + 
        results_df['pf_norm'] * 0.4 + 
        results_df['exp_norm'] * 0.2
    )
    
    best_comp_row = results_df.loc[results_df['composite_score'].idxmax()]
    print(f"\nMEJOR SCORE COMPUESTO (40% WR + 40% PF + 20% Exp):")
    print(f"   Proximidad: {best_comp_row['proximity_pct']}%")
    print(f"   Score: {best_comp_row['composite_score']:.2f}")
    print(f"   Win Rate: {best_comp_row['win_rate']:.2f}%")
    print(f"   Profit Factor: {best_comp_row['profit_factor']:.2f}")
    print(f"   Expectancy: {best_comp_row['expectancy']:.2f}")
    print(f"   Total Trades: {best_comp_row['total_trades']}")
    
    # Save results
    output_dir = project_root / 'results' / 'ema_proximity_optimization'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_file = output_dir / 'proximity_optimization_results.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"\n\nResultados guardados: {csv_file}")
    
    # Save JSON
    json_file = output_dir / 'proximity_optimization_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Resultados guardados: {json_file}")
    
    # Create visualization
    plot_optimization_results(results_df, output_dir)
    
    # Summary table (top 10 by composite score)
    print("\n" + "="*80)
    print("TOP 10 CONFIGURACIONES (por Score Compuesto)")
    print("="*80 + "\n")
    
    top_10 = results_df.nlargest(10, 'composite_score')[
        ['proximity_pct', 'total_trades', 'win_rate', 'profit_factor', 
         'expectancy', 'composite_score']
    ]
    
    print(top_10.to_string(index=False))
    
    print("\n" + "="*80)
    print("OPTIMIZACION COMPLETADA")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
