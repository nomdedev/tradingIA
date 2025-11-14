#!/usr/bin/env python3
"""
Script para ejecutar optimización completa del sistema
=====================================================
Uso:
    python scripts/run_optimization.py --mode=sensitivity
    python scripts/run_optimization.py --mode=bayes
    python scripts/run_optimization.py --mode=walk_forward
    python scripts/run_optimization.py --mode=all
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.optimizer import StrategyOptimizer
from src.mtf_data_handler import MultiTFDataHandler
from config.mtf_config import TRADING_CONFIG, RESULTS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data():
    """Carga datos de ejemplo para testing"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    logger.info("📥 Cargando datos de ejemplo...")
    
    # Generar datos sintéticos (reemplazar con Alpaca en producción)
    n_bars = 2000
    start_date = datetime(2024, 1, 1)
    
    dates_5m = [start_date + timedelta(minutes=5*i) for i in range(n_bars)]
    
    # Precios BTC simulados
    base_price = 45000
    returns = np.random.normal(0.0001, 0.003, n_bars)
    prices = base_price * (1 + returns).cumprod()
    
    # 5-minute data
    df_5m = pd.DataFrame({
        'timestamp': dates_5m,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_bars))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_bars)
    })
    df_5m.set_index('timestamp', inplace=True)
    
    # Añadir ATR
    df_5m['ATR'] = df_5m['close'] * 0.02
    
    # Resample to 15m
    df_15m = df_5m.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'ATR': 'mean'
    }).dropna()
    
    # Resample to 1h
    df_1h = df_5m.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'ATR': 'mean'
    }).dropna()
    
    dfs = {'5m': df_5m, '15m': df_15m, '1h': df_1h}
    
    logger.info(f"✅ Datos cargados: {len(df_5m)} barras 5m, {len(df_15m)} barras 15m, {len(df_1h)} barras 1h")
    
    return dfs


def run_sensitivity_analysis(optimizer, dfs):
    """Ejecuta análisis de sensibilidad"""
    logger.info("\n" + "="*60)
    logger.info("🔍 SENSITIVITY ANALYSIS - HEATMAPS")
    logger.info("="*60)
    
    # Test 1: atr_multi vs vol_thresh
    results1 = optimizer.param_sensitivity_heatmap(
        dfs,
        param1_name='atr_multi',
        param1_range=[0.1, 0.2, 0.3, 0.4, 0.5],
        param2_name='vol_thresh',
        param2_range=[0.8, 1.0, 1.2, 1.4, 1.5],
        metrics_to_plot=['sharpe_ratio', 'win_rate', 'max_drawdown']
    )
    
    # Test 2: tp_rr vs min_confidence
    results2 = optimizer.param_sensitivity_heatmap(
        dfs,
        param1_name='tp_rr',
        param1_range=[1.8, 2.0, 2.2, 2.4, 2.6],
        param2_name='min_confidence',
        param2_range=[0.4, 0.5, 0.6, 0.7, 0.8],
        metrics_to_plot=['sharpe_ratio', 'calmar_ratio']
    )
    
    logger.info("\n✅ Sensitivity analysis completo")
    return results1, results2


def run_efficient_frontier(optimizer, dfs):
    """Ejecuta Efficient Frontier"""
    logger.info("\n" + "="*60)
    logger.info("📊 EFFICIENT FRONTIER - RISK/SHARPE TRADE-OFFS")
    logger.info("="*60)
    
    param_ranges = {
        'atr_multi': (0.1, 0.5),
        'vol_thresh': (0.8, 1.5),
        'tp_rr': (1.8, 2.6)
    }
    
    risks, sharpes, param_combos = optimizer.efficient_frontier_params(
        dfs,
        param_ranges,
        n_points=30
    )
    
    logger.info("\n✅ Efficient Frontier completo")
    return risks, sharpes, param_combos


def run_bayesian_optimization(optimizer, dfs):
    """Ejecuta optimización Bayesiana"""
    logger.info("\n" + "="*60)
    logger.info("🤖 BAYESIAN OPTIMIZATION - SHARPE MAXIMIZATION")
    logger.info("="*60)
    
    param_bounds = [
        (0.1, 0.5),   # atr_multi
        (0.8, 1.5),   # vol_thresh
        (1.8, 2.6)    # tp_rr
    ]
    
    param_names = ['atr_multi', 'vol_thresh', 'tp_rr']
    
    # Optimize for Sharpe
    result_sharpe = optimizer.bayes_opt_sharpe(
        dfs,
        param_bounds,
        param_names,
        n_calls=50,
        target_metric='sharpe_ratio'
    )
    
    # Optimize for Calmar
    result_calmar = optimizer.bayes_opt_sharpe(
        dfs,
        param_bounds,
        param_names,
        n_calls=50,
        target_metric='calmar_ratio'
    )
    
    logger.info("\n✅ Bayesian optimization completo")
    return result_sharpe, result_calmar


def run_walk_forward(optimizer, dfs):
    """Ejecuta Walk-Forward validation"""
    logger.info("\n" + "="*60)
    logger.info("🚶 WALK-FORWARD VALIDATION - OOS TESTING")
    logger.info("="*60)
    
    param_bounds = [
        (0.1, 0.5),   # atr_multi
        (0.8, 1.5),   # vol_thresh
        (1.8, 2.6)    # tp_rr
    ]
    
    param_names = ['atr_multi', 'vol_thresh', 'tp_rr']
    
    results_df = optimizer.walk_forward_eval(
        dfs,
        n_periods=6,
        train_split=0.7,
        param_bounds=param_bounds,
        param_names=param_names
    )
    
    logger.info("\n✅ Walk-forward validation completo")
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Run strategy optimization')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sensitivity', 'frontier', 'bayes', 'walk_forward', 'all'],
        default='all',
        help='Modo de optimización'
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 STRATEGY OPTIMIZER - BTC IFVG Multi-TF")
    logger.info("="*60)
    
    # Load data
    dfs = load_sample_data()
    
    # Initialize optimizer
    optimizer = StrategyOptimizer(capital=10000, risk_pct=0.01)
    
    # Run optimization based on mode
    if args.mode == 'sensitivity' or args.mode == 'all':
        run_sensitivity_analysis(optimizer, dfs)
    
    if args.mode == 'frontier' or args.mode == 'all':
        run_efficient_frontier(optimizer, dfs)
    
    if args.mode == 'bayes' or args.mode == 'all':
        run_bayesian_optimization(optimizer, dfs)
    
    if args.mode == 'walk_forward' or args.mode == 'all':
        run_walk_forward(optimizer, dfs)
    
    logger.info("\n" + "="*60)
    logger.info("🎉 OPTIMIZACIÓN COMPLETA")
    logger.info("="*60)
    logger.info(f"📁 Resultados guardados en: {RESULTS_DIR / 'optimization'}")
    logger.info("\nArchivos generados:")
    logger.info("  - sensitivity_*.png (heatmaps)")
    logger.info("  - efficient_frontier_*.png")
    logger.info("  - bayes_opt_*.json (best params)")
    logger.info("  - walk_forward_*.png, *.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
