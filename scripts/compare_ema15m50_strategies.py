"""
Script de Backtest Comparativo con Filtro EMA15m50

Compara las estrategias originales vs versiones mejoradas con filtro EMA15m50:
1. Squeeze+ADX+TTM vs Squeeze+ADX+TTM+EMA15m50
2. VP+IFVG+EMA vs VP+IFVG+EMA+EMA15m50

Objetivo: Validar si el filtro de proximidad a la EMA50 de 15 minutos
mejora las métricas de trading basándose en el análisis previo que mostró:
- 56% win rate cerca de 15m EMA50
- 0.96 profit factor (mejor que otras EMAs)
- Mejor expectancy entre todas las configuraciones
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy
from strategies.squeeze_adx_ttm_ema15m50_strategy import SqueezeMomentumADXTTMEMA15m50Strategy
from strategies.vp_ifvg_ema_strategy import VPIFVGEmaStrategy
from strategies.vp_ifvg_ema_ema15m50_strategy import VPIFVGEmaEMA15m50Strategy
from backtest_optimized_2to1 import OptimizedBacktester


def load_data(symbol: str = 'BTC', timeframe: str = '5m') -> pd.DataFrame:
    """
    Cargar datos para backtest
    
    Args:
        symbol: Símbolo del activo
        timeframe: Timeframe de los datos
        
    Returns:
        DataFrame con datos OHLCV
    """
    data_file = project_root / f'data/btc_{timeframe.replace("m", "Min")}.csv'
    
    if not data_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_file}")
    
    print(f"📂 Cargando datos desde: {data_file}")
    df = pd.read_csv(data_file)
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Ensure timestamp column exists and convert to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif df.index.name == 'timestamp':
        df.index = pd.to_datetime(df.index)
    elif 'unnamed: 0' in df.columns:
        # Use first column as timestamp
        df['timestamp'] = pd.to_datetime(df['unnamed: 0'])
        df = df.set_index('timestamp')
        df = df.drop(columns=['unnamed: 0'])
    else:
        # Try to infer timestamp from index or create one
        df.index = pd.to_datetime(df.index)
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Columna requerida '{col}' no encontrada en los datos")
    
    # Use last 5000 bars for consistent comparison
    df = df.tail(5000)
    
    print(f"✅ Datos cargados: {len(df)} barras de 5 minutos")
    print(f"   • Período: {df.index[0]} a {df.index[-1]}")
    
    return df


def run_single_backtest(strategy_class, strategy_name: str, df: pd.DataFrame, 
                       initial_capital: float = 10000) -> dict:
    """
    Ejecutar un backtest individual
    
    Args:
        strategy_class: Clase de la estrategia
        strategy_name: Nombre de la estrategia
        df: DataFrame con datos OHLCV
        initial_capital: Capital inicial
        
    Returns:
        Diccionario con resultados del backtest
    """
    print(f"\n{'='*80}")
    print(f"🔄 Ejecutando backtest: {strategy_name}")
    print(f"{'='*80}\n")
    
    # Initialize strategy
    strategy = strategy_class()
    
    # Configure strategy parameters (consistent across all tests)
    strategy.set_parameters({
        'min_adx_entry': 15,
        'min_squeeze_momentum': 0.15,
        'stop_loss_atr_mult': 1.0,
        'take_profit_atr_mult': 3.0,  # 3:1 ratio for better results
    })
    
    # Initialize backtester
    backtester = OptimizedBacktester(
        strategy=strategy,
        initial_capital=int(initial_capital)
    )
    
    # Run backtest using run_optimized_backtest
    backtester.run_optimized_backtest(df)
    
    # Calculate metrics using the backtest method
    metrics = backtester._calculate_optimized_metrics(target_rr_ratio=3.0)
    
    # Prepare results dictionary with expected format
    results = {
        'total_trades': metrics.get('total_trades', 0),
        'win_rate': metrics.get('win_rate', 0),
        'profit_factor': metrics.get('profit_factor', 0),
        'total_return': metrics.get('total_return_pct', 0),
        'max_drawdown': metrics.get('max_drawdown', 0),
        'expectancy': metrics.get('mathematical_expectancy', 0),
        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
        'strategy_name': strategy_name,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Print summary
    print("\n✅ Resultados del Backtest:")
    print(f"   • Total Trades: {results['total_trades']}")
    print(f"   • Win Rate: {results['win_rate']:.1f}%")
    print(f"   • Profit Factor: {results['profit_factor']:.2f}")
    print(f"   • Total Return: {results['total_return']:.2f}%")
    print(f"   • Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"   • Mathematical Expectancy: {results['expectancy']:.4f}")
    print(f"   • Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    # Add strategy name to results
    results['strategy_name'] = strategy_name
    results['timestamp'] = datetime.now().isoformat()
    
    return results


def compare_results(original_results: dict, enhanced_results: dict) -> dict:
    """
    Comparar resultados entre estrategia original y mejorada
    
    Args:
        original_results: Resultados de estrategia original
        enhanced_results: Resultados de estrategia con EMA15m50
        
    Returns:
        Diccionario con comparación
    """
    comparison = {
        'original': original_results['strategy_name'],
        'enhanced': enhanced_results['strategy_name'],
        'improvements': {}
    }
    
    metrics = [
        ('total_trades', 'Total Trades', 'abs'),
        ('win_rate', 'Win Rate (%)', 'pct'),
        ('profit_factor', 'Profit Factor', 'abs'),
        ('total_return', 'Total Return (%)', 'pct'),
        ('max_drawdown', 'Max Drawdown (%)', 'pct_inverse'),
        ('expectancy', 'Expectancy', 'abs'),
        ('sharpe_ratio', 'Sharpe Ratio', 'abs'),
    ]
    
    print(f"\n{'='*80}")
    print(f"📊 COMPARACIÓN: {original_results['strategy_name']} vs {enhanced_results['strategy_name']}")
    print(f"{'='*80}\n")
    
    for metric_key, metric_name, comparison_type in metrics:
        original_val = original_results.get(metric_key, 0)
        enhanced_val = enhanced_results.get(metric_key, 0)
        
        if comparison_type == 'abs':
            change = enhanced_val - original_val
            change_pct = (change / abs(original_val) * 100) if original_val != 0 else 0
        elif comparison_type == 'pct':
            change = enhanced_val - original_val
            change_pct = change  # Already in percentage
        else:  # pct_inverse (lower is better)
            change = original_val - enhanced_val  # Invert for drawdown
            change_pct = -change  # Negative change means improvement
        
        improvement = change > 0
        arrow = "📈" if improvement else "📉"
        
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
    
    print(f"\n{'='*80}")
    print(f"🎯 RESUMEN: {improved_count}/{total_metrics} métricas mejoradas ({improvement_pct:.1f}%)")
    print(f"{'='*80}\n")
    
    comparison['summary'] = {
        'improved_metrics': improved_count,
        'total_metrics': total_metrics,
        'improvement_percentage': improvement_pct
    }
    
    return comparison


def save_results(all_results: dict, output_dir: Path):
    """
    Guardar todos los resultados en archivos
    
    Args:
        all_results: Diccionario con todos los resultados
        output_dir: Directorio de salida
    """
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON with all results
    json_file = output_dir / 'ema15m50_comparison_results.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"💾 Resultados guardados en: {json_file}")
    
    # Create summary table
    summary_data = []
    for strategy_key, results in all_results.items():
        if strategy_key.endswith('_comparison'):
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
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_file = output_dir / 'ema15m50_comparison_summary.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"📄 Tabla resumen guardada en: {csv_file}")
    
    return json_file, csv_file


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("🚀 BACKTEST COMPARATIVO: Estrategias con Filtro EMA15m50")
    print("="*80 + "\n")
    
    # Load data
    df = load_data(symbol='BTC', timeframe='5m')
    
    # Results storage
    all_results = {}
    
    # ========================================================================
    # Test 1: Squeeze+ADX+TTM Original vs Enhanced
    # ========================================================================
    print("\n" + "="*80)
    print("📋 TEST 1: Squeeze+ADX+TTM Strategy")
    print("="*80)
    
    squeeze_original = run_single_backtest(
        SqueezeMomentumADXTTMStrategy,
        "Squeeze+ADX+TTM (Original)",
        df.copy()
    )
    all_results['squeeze_original'] = squeeze_original
    
    squeeze_enhanced = run_single_backtest(
        SqueezeMomentumADXTTMEMA15m50Strategy,
        "Squeeze+ADX+TTM+EMA15m50",
        df.copy()
    )
    all_results['squeeze_enhanced'] = squeeze_enhanced
    
    squeeze_comparison = compare_results(squeeze_original, squeeze_enhanced)
    all_results['squeeze_comparison'] = squeeze_comparison
    
    # ========================================================================
    # Test 2: VP+IFVG+EMA Original vs Enhanced
    # ========================================================================
    print("\n" + "="*80)
    print("📋 TEST 2: VP+IFVG+EMA Strategy")
    print("="*80)
    
    vp_original = run_single_backtest(
        VPIFVGEmaStrategy,
        "VP+IFVG+EMA (Original)",
        df.copy()
    )
    all_results['vp_original'] = vp_original
    
    vp_enhanced = run_single_backtest(
        VPIFVGEmaEMA15m50Strategy,
        "VP+IFVG+EMA+EMA15m50",
        df.copy()
    )
    all_results['vp_enhanced'] = vp_enhanced
    
    vp_comparison = compare_results(vp_original, vp_enhanced)
    all_results['vp_comparison'] = vp_comparison
    
    # ========================================================================
    # Save Results
    # ========================================================================
    output_dir = project_root / 'results' / 'ema15m50_enhancement'
    json_file, csv_file = save_results(all_results, output_dir)
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("🎉 BACKTEST COMPARATIVO COMPLETADO")
    print("="*80)
    print(f"\n📊 Resumen General:")
    print(f"   • Squeeze+ADX+TTM: {squeeze_comparison['summary']['improved_metrics']}/{squeeze_comparison['summary']['total_metrics']} métricas mejoradas")
    print(f"   • VP+IFVG+EMA: {vp_comparison['summary']['improved_metrics']}/{vp_comparison['summary']['total_metrics']} métricas mejoradas")
    print(f"\n💾 Archivos generados:")
    print(f"   • {json_file}")
    print(f"   • {csv_file}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
