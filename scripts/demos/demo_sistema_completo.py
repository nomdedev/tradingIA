"""
Demo Script - Sistema IFVG Trading Completo
============================================
Demuestra todas las funcionalidades del sistema:
1. Backtesting
2. Optimization (Grid Search)
3. Walk-Forward Analysis
4. Monte Carlo Simulation
5. Visualización de resultados

Uso:
    python demo_sistema_completo.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from src.data_fetcher import DataFetcher
from src.indicators import calculate_all_indicators
from src.backtester import Backtester
from src.optimization import ParameterOptimizer


def demo_backtesting():
    """Demostración de backtesting básico"""
    print("\n" + "="*80)
    print("🎯 DEMO 1: BACKTESTING BÁSICO")
    print("="*80)
    
    # Fetch data
    logger.info("Cargando datos históricos...")
    fetcher = DataFetcher()
    df = fetcher.get_historical_data(
        symbol='BTCUSD',
        timeframe='5Min',
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    if df is None or len(df) == 0:
        logger.error("No se pudieron cargar datos")
        return None
    
    logger.info(f"✅ Datos cargados: {len(df)} barras")
    
    # Calculate indicators
    logger.info("Calculando indicadores...")
    df = calculate_all_indicators(df)
    logger.info(f"✅ Indicadores calculados")
    
    # Run backtest
    logger.info("Ejecutando backtest...")
    backtester = Backtester(
        df=df,
        initial_capital=10000,
        risk_per_trade=0.02,
        commission=0.001,
        slippage=0.0005
    )
    
    metrics = backtester.run_backtest(
        use_stop_loss=True,
        use_take_profit=True,
        sl_atr_multiplier=1.5,
        tp_risk_reward=2.0
    )
    
    # Print results
    print("\n📊 Resultados del Backtest:")
    print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
    print(f"  Total Return: {metrics.get('total_return', 0):.2f}%")
    print(f"  Final Capital: ${metrics.get('final_capital', 0):,.2f}")
    
    # Save results
    backtester.save_results('results/demo_backtest')
    logger.info("✅ Resultados guardados en results/demo_backtest_*")
    
    return df


def demo_grid_search():
    """Demostración de Grid Search"""
    print("\n" + "="*80)
    print("🔍 DEMO 2: GRID SEARCH OPTIMIZATION")
    print("="*80)
    
    # Define parameter grid
    param_grid = {
        'risk_per_trade': [0.01, 0.02, 0.03],
        'sl_atr_multiplier': [1.0, 1.5, 2.0],
        'tp_risk_reward': [1.5, 2.0, 2.5]
    }
    
    logger.info(f"Parámetros a testear:")
    for key, values in param_grid.items():
        logger.info(f"  {key}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    logger.info(f"Total combinaciones: {total_combinations}")
    
    # Run grid search
    optimizer = ParameterOptimizer()
    results_df = optimizer.grid_search(
        param_grid=param_grid,
        start_date='2024-01-01',
        end_date='2024-06-30',  # Shorter period for demo
        initial_capital=10000,
        optimize_metric='sharpe_ratio',
        max_workers=2  # Adjust based on CPU
    )
    
    if len(results_df) > 0:
        # Print top 5 results
        print("\n🏆 Top 5 Configuraciones:")
        print(results_df.head(5).to_string())
        
        # Best parameters
        best = results_df.iloc[0]
        print(f"\n✨ Mejor Configuración:")
        print(f"  Risk per Trade: {best['risk_per_trade']:.3f}")
        print(f"  SL Multiplier: {best['sl_atr_multiplier']:.1f}")
        print(f"  TP Risk/Reward: {best['tp_risk_reward']:.1f}")
        print(f"  → Sharpe Ratio: {best['sharpe_ratio']:.3f}")
        
        # Save results
        results_df.to_csv('results/demo_grid_search.csv', index=False)
        optimizer.save_results('results/demo_optimization.json')
        logger.info("✅ Resultados guardados")
    
    return results_df


def demo_walk_forward():
    """Demostración de Walk-Forward Analysis"""
    print("\n" + "="*80)
    print("📈 DEMO 3: WALK-FORWARD ANALYSIS")
    print("="*80)
    
    param_grid = {
        'risk_per_trade': [0.01, 0.02],
        'sl_atr_multiplier': [1.5, 2.0],
        'tp_risk_reward': [2.0, 2.5]
    }
    
    optimizer = ParameterOptimizer()
    
    wf_results = optimizer.walk_forward_analysis(
        param_grid=param_grid,
        start_date='2024-01-01',
        end_date='2024-12-31',
        train_period_days=90,
        test_period_days=30,
        initial_capital=10000,
        optimize_metric='sharpe_ratio'
    )
    
    if wf_results:
        print("\n📊 Resultados Walk-Forward:")
        print(f"  Windows tested: {wf_results['num_windows']}")
        print(f"  Avg Test Sharpe: {wf_results['avg_test_metric']:.3f} ± {wf_results['std_test_metric']:.3f}")
        print(f"  Range: [{wf_results['min_test_metric']:.3f}, {wf_results['max_test_metric']:.3f}]")
        
        # Analysis of degradation
        window_results = wf_results['window_results']
        train_sharpes = [w['train_metric'] for w in window_results]
        test_sharpes = [w['test_metric'] for w in window_results]
        
        avg_degradation = np.mean([t - te for t, te in zip(train_sharpes, test_sharpes)])
        print(f"\n  Avg Degradation (Train → Test): {avg_degradation:.3f}")
        
        if avg_degradation > 0.5:
            print("  ⚠️ WARNING: Possible overfitting detected!")
        else:
            print("  ✅ Good generalization!")
    
    return wf_results


def demo_monte_carlo():
    """Demostración de Monte Carlo Simulation"""
    print("\n" + "="*80)
    print("🎲 DEMO 4: MONTE CARLO SIMULATION")
    print("="*80)
    
    # Load historical trades
    trades_file = Path('results/demo_backtest_trades.csv')
    
    if not trades_file.exists():
        logger.warning("No hay trades históricos. Ejecute demo_backtesting() primero.")
        return None
    
    trades_df = pd.read_csv(trades_file)
    logger.info(f"Trades cargados: {len(trades_df)}")
    
    # Run Monte Carlo
    optimizer = ParameterOptimizer()
    mc_results = optimizer.monte_carlo_simulation(
        trades_df=trades_df,
        n_simulations=1000,
        initial_capital=10000
    )
    
    print("\n📊 Resultados Monte Carlo (1000 simulaciones):")
    print("\n💰 Final Capital:")
    print(f"  Mean: ${mc_results['final_capital']['mean']:,.2f}")
    print(f"  Std: ${mc_results['final_capital']['std']:,.2f}")
    print(f"  Range: [${mc_results['final_capital']['min']:,.2f}, ${mc_results['final_capital']['max']:,.2f}]")
    
    print("\n📉 Max Drawdown:")
    print(f"  Mean: {mc_results['max_drawdown']['mean']:.2f}%")
    print(f"  Worst case (95th): {mc_results['max_drawdown']['percentiles'][95]:.2f}%")
    
    print("\n📈 Sharpe Ratio:")
    print(f"  Mean: {mc_results['sharpe_ratio']['mean']:.3f}")
    print(f"  Range: [{mc_results['sharpe_ratio']['min']:.3f}, {mc_results['sharpe_ratio']['max']:.3f}]")
    
    print(f"\n🎯 Probability of Profit: {mc_results['probability_profit']:.1f}%")
    print(f"⚠️ Risk of Ruin (>50% loss): {mc_results['risk_of_ruin']:.1f}%")
    
    # Percentiles
    print("\n📊 Percentiles de Final Capital:")
    for p, v in mc_results['final_capital']['percentiles'].items():
        print(f"  {p}th: ${v:,.2f}")
    
    return mc_results


def demo_complete_workflow():
    """Workflow completo de análisis"""
    print("\n" + "="*80)
    print("🚀 DEMO COMPLETO: WORKFLOW DE ANÁLISIS")
    print("="*80)
    
    # Step 1: Backtest
    print("\n📌 Step 1: Backtest inicial")
    df = demo_backtesting()
    
    if df is None:
        logger.error("Falló el backtest. Abortando demo.")
        return
    
    input("\nPresione Enter para continuar con Grid Search...")
    
    # Step 2: Grid Search
    print("\n📌 Step 2: Grid Search")
    grid_results = demo_grid_search()
    
    input("\nPresione Enter para continuar con Walk-Forward...")
    
    # Step 3: Walk-Forward
    print("\n📌 Step 3: Walk-Forward Analysis")
    wf_results = demo_walk_forward()
    
    input("\nPresione Enter para continuar con Monte Carlo...")
    
    # Step 4: Monte Carlo
    print("\n📌 Step 4: Monte Carlo Simulation")
    mc_results = demo_monte_carlo()
    
    # Summary
    print("\n" + "="*80)
    print("✅ DEMO COMPLETADO")
    print("="*80)
    print("\nArchivos generados:")
    print("  - results/demo_backtest_trades.csv")
    print("  - results/demo_backtest_equity.csv")
    print("  - results/demo_backtest_metrics.json")
    print("  - results/demo_grid_search.csv")
    print("  - results/demo_optimization.json")
    
    print("\n📊 Para visualizar resultados:")
    print("  python main.py --mode dashboard")
    print("  Luego seleccionar 'Backtest Results'")
    
    print("\n🎉 Demo finalizado exitosamente!")


if __name__ == '__main__':
    try:
        demo_complete_workflow()
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error en demo: {e}", exc_info=True)
