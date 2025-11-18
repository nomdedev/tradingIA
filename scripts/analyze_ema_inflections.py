"""
Análisis de Puntos de Inflexión en EMAs Multi-Timeframe
========================================================

Este script analiza el comportamiento del precio y los indicadores cuando se acerca
a diferentes EMAs en timeframes superiores (15m y 1h).

Configuraciones a analizar:
1. 15m EMA50 - Proximidad 2%
2. 15m EMA200 - Proximidad 2%
3. 1h EMA20 - Proximidad 2%
4. 1h EMA50 - Proximidad 2%

Para cada configuración, se analiza:
- Comportamiento de ADX cuando el precio está cerca de la EMA
- Estado del Squeeze Momentum cuando hay proximidad
- Señales de TTM Squeeze
- Dirección de la tendencia (precio vs EMA)
- Métricas del backtest con ese filtro
"""

import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy

# Import OptimizedBacktester from backtest_optimized_2to1.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_optimized_2to1 import OptimizedBacktester


def analyze_ema_proximity_behavior(df_5m, df_higher_tf, ema_period, timeframe_name, proximity_pct=2.0):
    """
    Analiza el comportamiento de los indicadores cuando el precio está cerca de una EMA
    en un timeframe superior.
    
    Parameters:
    -----------
    df_5m : DataFrame
        Datos de 5 minutos con indicadores
    df_higher_tf : DataFrame
        Datos del timeframe superior (15m o 1h)
    ema_period : int
        Período de la EMA a analizar (20, 50, 200)
    timeframe_name : str
        Nombre del timeframe ('15m' o '1h')
    proximity_pct : float
        Porcentaje de proximidad para considerar (default 2%)
        
    Returns:
    --------
    dict : Análisis detallado del comportamiento
    """
    
    # Calculate EMA on higher timeframe
    df_higher_tf = df_higher_tf.copy()
    ema_col = f'ema{ema_period}'
    df_higher_tf[ema_col] = talib.EMA(df_higher_tf['close'], timeperiod=ema_period)
    
    # Align with 5m timeframe
    df_5m = df_5m.copy()
    df_5m[ema_col] = df_higher_tf[ema_col].reindex(df_5m.index, method='ffill')
    
    # Calculate proximity
    df_5m['ema_distance_pct'] = ((df_5m['close'] - df_5m[ema_col]) / df_5m[ema_col] * 100).abs()
    df_5m['near_ema'] = df_5m['ema_distance_pct'] <= proximity_pct
    
    # Calculate EMA slope (trending direction)
    df_5m['ema_slope'] = df_5m[ema_col].diff(5)
    df_5m['ema_trending_up'] = df_5m['ema_slope'] > 0
    
    # Price position relative to EMA
    df_5m['price_above_ema'] = df_5m['close'] > df_5m[ema_col]
    
    # Analyze indicator behavior when near EMA
    near_ema_data = df_5m[df_5m['near_ema']]
    
    analysis = {
        'ema_period': ema_period,
        'timeframe': timeframe_name,
        'proximity_pct': proximity_pct,
        'total_bars': len(df_5m),
        'bars_near_ema': len(near_ema_data),
        'pct_time_near_ema': len(near_ema_data) / len(df_5m) * 100,
        
        # ADX behavior near EMA
        'avg_adx_near_ema': near_ema_data['adx'].mean() if 'adx' in near_ema_data.columns else 0,
        'avg_adx_overall': df_5m['adx'].mean() if 'adx' in df_5m.columns else 0,
        
        # Squeeze behavior near EMA
        'squeeze_on_near_ema_pct': (near_ema_data['squeeze_on'].sum() / len(near_ema_data) * 100) if 'squeeze_on' in near_ema_data.columns and len(near_ema_data) > 0 else 0,
        'squeeze_on_overall_pct': (df_5m['squeeze_on'].sum() / len(df_5m) * 100) if 'squeeze_on' in df_5m.columns else 0,
        
        # Momentum behavior near EMA
        'avg_squeeze_momentum_near_ema': near_ema_data['squeeze_momentum'].mean() if 'squeeze_momentum' in near_ema_data.columns else 0,
        'avg_squeeze_momentum_overall': df_5m['squeeze_momentum'].mean() if 'squeeze_momentum' in df_5m.columns else 0,
        
        # EMA trending
        'pct_ema_trending_up': (df_5m['ema_trending_up'].sum() / len(df_5m) * 100) if len(df_5m) > 0 else 0,
        'pct_price_above_ema': (df_5m['price_above_ema'].sum() / len(df_5m) * 100) if len(df_5m) > 0 else 0,
        
        # Reversal detection near EMA
        'reversals_near_ema': 0,  # Will be calculated
    }
    
    # Detect price reversals when near EMA
    reversals = 0
    for i in range(1, len(df_5m) - 1):
        if df_5m.iloc[i]['near_ema']:
            # Check if price crossed EMA
            price_cross = (df_5m.iloc[i-1]['price_above_ema'] != df_5m.iloc[i+1]['price_above_ema'])
            if price_cross:
                reversals += 1
    
    analysis['reversals_near_ema'] = reversals
    analysis['reversal_rate'] = reversals / len(near_ema_data) * 100 if len(near_ema_data) > 0 else 0
    
    return analysis


def run_backtest_with_ema_filter(df_5m, df_15m, df_1h, ema_config):
    """
    Ejecuta un backtest con filtro de proximidad a EMA específica
    
    Parameters:
    -----------
    df_5m, df_15m, df_1h : DataFrames
        Datos de diferentes timeframes
    ema_config : dict
        Configuración de la EMA: {'timeframe': '15m', 'period': 50, 'proximity': 2.0}
    """
    
    # Prepare data based on timeframe
    if ema_config['timeframe'] == '15m':
        df_higher = df_15m.copy()
    else:  # '1h'
        df_higher = df_1h.copy()
    
    # Calculate EMA on higher timeframe
    ema_col = f"ema{ema_config['period']}"
    df_higher[ema_col] = talib.EMA(df_higher['close'], timeperiod=ema_config['period'])
    
    # Align with 5m data
    df_5m_test = df_5m.copy()
    df_5m_test[ema_col] = df_higher[ema_col].reindex(df_5m_test.index, method='ffill')
    
    # Calculate proximity
    df_5m_test['ema_distance_pct'] = ((df_5m_test['close'] - df_5m_test[ema_col]) / df_5m_test[ema_col] * 100).abs()
    df_5m_test['near_ema'] = df_5m_test['ema_distance_pct'] <= ema_config['proximity']
    
    # Create strategy with best parameters
    strategy = SqueezeMomentumADXTTMStrategy()
    strategy.min_adx_entry = 15
    strategy.max_adx_entry = 100
    strategy.min_squeeze_momentum = 0.15
    strategy.stop_loss_atr_mult = 1.0
    strategy.take_profit_atr_mult = 3.0
    strategy.use_volume_filter = False
    strategy.use_multitimeframe = False
    strategy.use_poc_filter = False
    strategy.use_adx_slope = False
    strategy.trailing_stop = True
    strategy.trailing_activation = 0.5
    
    # Create backtester with EMA filter disabled (we'll filter later)
    backtester = OptimizedBacktester(strategy)
    backtester.cooldown_bars = 1
    backtester.use_ema_filter = False  # Disable default multi-timeframe EMA filter
    
    # Run backtest
    metrics = backtester.run_optimized_backtest(df_5m_test, df_15m, df_1h)
    
    # Filter trades to only those that passed the EMA proximity filter
    filtered_trades = []
    for trade in backtester.trades:
        entry_idx = trade.get('entry_idx', 0)
        if entry_idx < len(df_5m_test) and df_5m_test.iloc[entry_idx]['near_ema']:
            filtered_trades.append(trade)
    
    # Recalculate metrics with filtered trades
    if len(filtered_trades) > 0:
        backtester.trades = filtered_trades
        
        # Rebuild equity curve from filtered trades
        equity = [backtester.initial_capital]
        for trade in filtered_trades:
            equity.append(equity[-1] + trade.get('pnl', 0))
        backtester.equity_curve = equity
        
        # Recalculate metrics
        metrics = backtester._calculate_optimized_metrics(2.0)
    else:
        # No trades passed filter
        metrics = {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return_pct': 0,
            'max_drawdown': 0,
            'mathematical_expectancy': 0,
            'sharpe_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_duration': 0,
            'rr_ratio_achieved': 0
        }
    
    return metrics, backtester


def plot_ema_analysis(ema_config, analysis, metrics, backtester, save_path):
    """
    Genera gráficos completos del análisis de EMA
    """
    
    config_name = f"{ema_config['timeframe']}_EMA{ema_config['period']}"
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f"Análisis de Proximidad a {config_name} (±{ema_config['proximity']}%)\n"
                 f"Profit Factor: {metrics['profit_factor']:.2f} | Win Rate: {metrics['win_rate']:.1f}% | "
                 f"Return: {metrics['total_return_pct']:.2f}%",
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :2])
    equity_curve = backtester.equity_curve
    ax1.plot(equity_curve, linewidth=2, color='#2E86AB')
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Capital Inicial')
    ax1.set_title(f'Curva de Capital - {config_name}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Número de Trade')
    ax1.set_ylabel('Capital ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Fill area
    ax1.fill_between(range(len(equity_curve)), equity_curve, 10000, 
                      where=(np.array(equity_curve) >= 10000), alpha=0.3, color='green', label='Profit')
    ax1.fill_between(range(len(equity_curve)), equity_curve, 10000, 
                      where=(np.array(equity_curve) < 10000), alpha=0.3, color='red', label='Loss')
    
    # 2. Indicator Analysis Near EMA
    ax2 = fig.add_subplot(gs[0, 2])
    indicator_data = {
        'ADX Near EMA': analysis['avg_adx_near_ema'],
        'ADX Overall': analysis['avg_adx_overall'],
        'Squeeze %\nNear EMA': analysis['squeeze_on_near_ema_pct'],
        'Squeeze %\nOverall': analysis['squeeze_on_overall_pct']
    }
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = ax2.bar(range(len(indicator_data)), list(indicator_data.values()), color=colors, alpha=0.7)
    ax2.set_xticks(range(len(indicator_data)))
    ax2.set_xticklabels(list(indicator_data.keys()), rotation=45, ha='right', fontsize=9)
    ax2.set_title('Comportamiento de Indicadores', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Valor')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Returns Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    trade_returns = [trade['pnl'] for trade in backtester.trades]
    if trade_returns:
        ax3.hist(trade_returns, bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break Even')
        ax3.set_title('Distribución de Retornos por Trade', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Retorno ($)')
        ax3.set_ylabel('Frecuencia')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Win/Loss Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    wins = sum(1 for trade in backtester.trades if trade['pnl'] > 0)
    losses = sum(1 for trade in backtester.trades if trade['pnl'] <= 0)
    
    wedges, texts, autotexts = ax4.pie([wins, losses], 
                                        labels=['Wins', 'Losses'],
                                        colors=['#2ECC71', '#E74C3C'],
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        explode=(0.05, 0.05))
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax4.set_title(f'Win/Loss Ratio\n{wins}W / {losses}L', fontsize=11, fontweight='bold')
    
    # 5. EMA Proximity Stats
    ax5 = fig.add_subplot(gs[1, 2])
    proximity_stats = {
        'Time Near\nEMA (%)': analysis['pct_time_near_ema'],
        'Reversal\nRate (%)': analysis['reversal_rate'],
        'Price Above\nEMA (%)': analysis['pct_price_above_ema'],
        'EMA Uptrend\n(%)': analysis['pct_ema_trending_up']
    }
    
    colors_prox = ['#3498DB', '#E67E22', '#9B59B6', '#1ABC9C']
    bars_prox = ax5.bar(range(len(proximity_stats)), list(proximity_stats.values()), 
                        color=colors_prox, alpha=0.7)
    ax5.set_xticks(range(len(proximity_stats)))
    ax5.set_xticklabels(list(proximity_stats.keys()), rotation=45, ha='right', fontsize=9)
    ax5.set_title('Estadísticas de Proximidad a EMA', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Porcentaje (%)')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar in bars_prox:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 6. Cumulative Returns
    ax6 = fig.add_subplot(gs[2, 0])
    if trade_returns:
        cumulative_returns = np.cumsum(trade_returns)
        ax6.plot(cumulative_returns, linewidth=2, color='#2E86AB')
        ax6.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax6.fill_between(range(len(cumulative_returns)), cumulative_returns, 0,
                         where=(np.array(cumulative_returns) >= 0), alpha=0.3, color='green')
        ax6.fill_between(range(len(cumulative_returns)), cumulative_returns, 0,
                         where=(np.array(cumulative_returns) < 0), alpha=0.3, color='red')
        ax6.set_title('Retornos Acumulados', fontsize=11, fontweight='bold')
        ax6.set_xlabel('Número de Trade')
        ax6.set_ylabel('Retorno Acumulado ($)')
        ax6.grid(True, alpha=0.3)
    
    # 7. Key Metrics Table
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')
    
    metrics_table = [
        ['Métrica', 'Valor'],
        ['Total Trades', f"{metrics['total_trades']}"],
        ['Win Rate', f"{metrics['win_rate']:.1f}%"],
        ['Profit Factor', f"{metrics['profit_factor']:.2f}"],
        ['Total Return', f"{metrics['total_return_pct']:.2f}%"],
        ['Max Drawdown', f"{metrics['max_drawdown']:.2f}%"],
        ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
        ['Avg Win', f"${metrics['avg_win']:.2f}"],
        ['Avg Loss', f"${metrics['avg_loss']:.2f}"],
        ['Mathematical Expectancy', f"{metrics['mathematical_expectancy']:.4f}"],
        ['Bars Near EMA', f"{analysis['bars_near_ema']} ({analysis['pct_time_near_ema']:.1f}%)"],
        ['Reversals Near EMA', f"{analysis['reversals_near_ema']}"]
    ]
    
    table = ax7.table(cellText=metrics_table, cellLoc='left', loc='center',
                      colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics_table)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Gráfico guardado: {save_path}")


def main():
    print("\n" + "="*80)
    print("🔍 ANÁLISIS DE PUNTOS DE INFLEXIÓN EN EMAs MULTI-TIMEFRAME")
    print("="*80)
    
    # Load data
    print("\n📂 Cargando datos...")
    df_5m = pd.read_csv('data/btc_5Min.csv', index_col=0, parse_dates=True)
    df_5m.columns = df_5m.columns.str.lower()
    df_5m = df_5m.tail(5000)  # Use last 5000 bars for analysis
    
    df_15m = pd.read_csv('data/btc_15Min.csv', index_col=0, parse_dates=True)
    df_15m.columns = df_15m.columns.str.lower()
    
    df_1h = pd.read_csv('data/btc_1H.csv', index_col=0, parse_dates=True)
    df_1h.columns = df_1h.columns.str.lower()
    
    print(f"✅ Datos cargados: {len(df_5m)} barras de 5m")
    
    # Calculate base indicators for analysis
    print("\n📊 Calculando indicadores base...")
    df_5m['adx'] = talib.ADX(df_5m['high'], df_5m['low'], df_5m['close'], timeperiod=14)
    
    # Bollinger Bands for Squeeze
    bb_upper, bb_middle, bb_lower = talib.BBANDS(df_5m['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    
    # Keltner Channels for Squeeze
    atr = talib.ATR(df_5m['high'], df_5m['low'], df_5m['close'], timeperiod=20)
    ema20 = talib.EMA(df_5m['close'], timeperiod=20)
    kc_upper = ema20 + (atr * 1.5)
    kc_lower = ema20 - (atr * 1.5)
    
    df_5m['squeeze_on'] = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    
    # Linear regression for momentum
    df_5m['squeeze_momentum'] = talib.LINEARREG(df_5m['close'] - ((ema20 + ema20) / 2), timeperiod=20)
    
    # EMA configurations to test
    ema_configs = [
        {'timeframe': '15m', 'period': 50, 'proximity': 2.0},
        {'timeframe': '15m', 'period': 200, 'proximity': 2.0},
        {'timeframe': '1h', 'period': 20, 'proximity': 2.0},
        {'timeframe': '1h', 'period': 50, 'proximity': 2.0},
    ]
    
    results_summary = []
    
    for i, config in enumerate(ema_configs, 1):
        print(f"\n{'='*80}")
        print(f"📈 ANÁLISIS {i}/{len(ema_configs)}: {config['timeframe']} EMA{config['period']}")
        print(f"{'='*80}")
        
        # Select higher timeframe data
        df_higher = df_15m if config['timeframe'] == '15m' else df_1h
        
        # Analyze behavior
        print(f"\n🔍 Analizando comportamiento cerca de {config['timeframe']} EMA{config['period']}...")
        analysis = analyze_ema_proximity_behavior(
            df_5m, df_higher, config['period'], config['timeframe'], config['proximity']
        )
        
        print(f"\n📊 Estadísticas de Proximidad:")
        print(f"   • Tiempo cerca de EMA: {analysis['pct_time_near_ema']:.1f}%")
        print(f"   • ADX promedio cerca de EMA: {analysis['avg_adx_near_ema']:.1f}")
        print(f"   • Squeeze activo cerca de EMA: {analysis['squeeze_on_near_ema_pct']:.1f}%")
        print(f"   • Tasa de reversión cerca de EMA: {analysis['reversal_rate']:.1f}%")
        print(f"   • Precio sobre EMA: {analysis['pct_price_above_ema']:.1f}%")
        print(f"   • EMA en tendencia alcista: {analysis['pct_ema_trending_up']:.1f}%")
        
        # Run backtest
        print(f"\n🔄 Ejecutando backtest con filtro de proximidad a {config['timeframe']} EMA{config['period']}...")
        metrics, backtester = run_backtest_with_ema_filter(df_5m, df_15m, df_1h, config)
        
        print(f"\n✅ Resultados del Backtest:")
        print(f"   • Total Trades: {metrics['total_trades']}")
        print(f"   • Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   • Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   • Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"   • Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"   • Mathematical Expectancy: {metrics['mathematical_expectancy']:.4f}")
        print(f"   • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        
        # Generate plot
        config_name = f"{config['timeframe']}_EMA{config['period']}"
        save_path = f"results/ema_analysis_{config_name}_PF{metrics['profit_factor']:.2f}.png"
        
        print(f"\n📊 Generando gráfico de análisis...")
        plot_ema_analysis(config, analysis, metrics, backtester, save_path)
        
        # Store results
        results_summary.append({
            'config': f"{config['timeframe']} EMA{config['period']}",
            'total_trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'return': metrics['total_return_pct'],
            'max_dd': metrics['max_drawdown'],
            'expectancy': metrics['mathematical_expectancy'],
            'sharpe': metrics.get('sharpe_ratio', 0),
            'time_near_ema': analysis['pct_time_near_ema'],
            'reversal_rate': analysis['reversal_rate'],
            'avg_adx_near': analysis['avg_adx_near_ema'],
            'squeeze_pct_near': analysis['squeeze_on_near_ema_pct']
        })
    
    # Print summary table
    print(f"\n{'='*80}")
    print("📋 RESUMEN COMPARATIVO DE TODAS LAS CONFIGURACIONES")
    print(f"{'='*80}\n")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results_summary)
    
    print(summary_df.to_string(index=False))
    
    # Find best configuration
    best_pf = summary_df.loc[summary_df['profit_factor'].idxmax()]
    best_wr = summary_df.loc[summary_df['win_rate'].idxmax()]
    best_exp = summary_df.loc[summary_df['expectancy'].idxmax()]
    
    print(f"\n🏆 MEJORES CONFIGURACIONES:")
    print(f"   • Mejor Profit Factor: {best_pf['config']} (PF: {best_pf['profit_factor']:.2f})")
    print(f"   • Mejor Win Rate: {best_wr['config']} (WR: {best_wr['win_rate']:.1f}%)")
    print(f"   • Mejor Expectancy: {best_exp['config']} (Exp: {best_exp['expectancy']:.4f})")
    
    # Save summary to CSV
    summary_path = 'results/ema_analysis_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Resumen guardado en: {summary_path}")
    
    print(f"\n{'='*80}")
    print("✅ ANÁLISIS COMPLETADO")
    print(f"{'='*80}")
    print(f"\n📊 Gráficos generados en: results/ema_analysis_*.png")
    print(f"📄 Resumen CSV: {summary_path}")


if __name__ == "__main__":
    main()
