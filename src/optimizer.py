#!/usr/bin/env python3
"""
Strategy Optimizer - Sharpe Maximization & Efficient Frontier
============================================================
Implementa optimización de parámetros usando:
- Sensitivity Analysis (Heatmaps)
- Efficient Frontier adaptado a parámetros
- Bayesian Optimization (Sharpe/Calmar)
- Walk-Forward Validation

Metodología:
- Sharpe = (return - rf) / std_return, rf=4% anual
- Efficient Frontier: Trade-off params para max Sharpe min DD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json

# Optimization libraries
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import scipy.optimize as sco

# Local imports
from .backtester import AdvancedBacktester
from config.mtf_config import RESULTS_DIR

logger = logging.getLogger(__name__)

# Create optimizer results directory
OPT_RESULTS_DIR = RESULTS_DIR / 'optimization'
OPT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class StrategyOptimizer:
    """
    Optimizador de estrategia con múltiples metodologías
    """

    def __init__(self, capital: float = 10000, risk_pct: float = 0.01):
        """
        Inicializa el optimizador

        Args:
            capital: Capital inicial
            risk_pct: Porcentaje de riesgo por trade
        """
        self.capital = capital
        self.risk_pct = risk_pct
        self.backtester = AdvancedBacktester(capital=capital)
        self.results_cache = {}

    def _calculate_sharpe(
        self,
        returns: pd.Series,
        rf_annual: float = 0.04,
        periods_per_year: int = 72576  # 5-min bars in a trading year (252*24*12)
    ) -> float:
        """
        Calcula Sharpe Ratio con fórmula correcta

        Args:
            returns: Serie de returns por trade
            rf_annual: Risk-free rate anual (default 4%)
            periods_per_year: Períodos por año para anualizar (252 días trading)

        Returns:
            Sharpe ratio anualizado
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Calcular excess returns (restar risk-free rate por período)
        rf_per_period = rf_annual / periods_per_year
        excess_returns = returns - rf_per_period

        # Sharpe = E[R - Rf] / σ[R - Rf] * sqrt(periods_per_year)
        # Fórmula correcta: anualizar ratio, no componentes por separado
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

        return sharpe

    def _calculate_calmar(
        self,
        equity_curve: pd.Series,
        periods_per_year: int = 72576  # Correcto: 252 días trading
    ) -> float:
        """
        Calcula Calmar Ratio con anualización correcta

        Args:
            equity_curve: Curva de equity
            periods_per_year: Períodos por año (252 días trading)
            periods_per_year: Períodos por año

        Returns:
            Calmar ratio (CAGR / Max Drawdown)
        """
        if len(equity_curve) < 2:
            return 0.0

        # CAGR (Compound Annual Growth Rate) correcto
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        periods = len(equity_curve)
        years = periods / periods_per_year
        
        if years > 0 and equity_curve.iloc[0] > 0:
            # CAGR = (Final / Initial)^(1/years) - 1
            annual_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
        else:
            annual_return = 0.0

        # Max Drawdown (ya es negativo de drawdowns.min())
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_dd = abs(drawdowns.min())  # Convertir a positivo para división

        if max_dd == 0:
            return 0.0

        # Calmar = CAGR / Max Drawdown
        calmar = annual_return / max_dd

        return calmar

    def _run_backtest_with_params(
        self,
        dfs: Dict[str, pd.DataFrame],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ejecuta backtest con parámetros específicos

        Args:
            dfs: DataFrames multi-TF
            params: Parámetros de estrategia

        Returns:
            Dict con métricas
        """
        # Cache key
        cache_key = str(sorted(params.items()))
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]

        try:
            result = self.backtester.run_optimized_backtest(dfs, params)

            metrics = result['metrics']
            trades = result.get('trades', [])
            equity_curve = result.get('equity_curve', pd.Series([self.capital]))

            # Calcular returns por trade
            if len(trades) > 0:
                returns = pd.Series([t.get('pnl_pct', 0) for t in trades])
            else:
                returns = pd.Series([0])

            # Calcular Sharpe y Calmar
            sharpe = self._calculate_sharpe(returns)
            calmar = self._calculate_calmar(equity_curve)

            # Agregar a métricas
            metrics['sharpe_ratio'] = sharpe
            metrics['calmar_ratio'] = calmar

            # Cache result
            self.results_cache[cache_key] = metrics

            return metrics

        except Exception as e:
            logger.error(f"Error en backtest con params {params}: {e}")
            return {
                'total_return': 0,
                'win_rate': 0,
                'max_drawdown': 1,
                'sharpe_ratio': -10,
                'calmar_ratio': -10,
                'total_trades': 0
            }

    def param_sensitivity_heatmap(
        self,
        dfs: Dict[str, pd.DataFrame],
        param1_name: str = 'atr_multi',
        param1_range: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
        param2_name: str = 'vol_thresh',
        param2_range: List[float] = [0.8, 1.0, 1.2, 1.4, 1.5],
        base_params: Optional[Dict[str, Any]] = None,
        metrics_to_plot: List[str] = ['sharpe_ratio', 'win_rate', 'max_drawdown']
    ) -> Dict[str, pd.DataFrame]:
        """
        Genera heatmaps de sensibilidad de parámetros

        Args:
            dfs: DataFrames multi-TF
            param1_name: Nombre del primer parámetro
            param1_range: Rango de valores para param1
            param2_name: Nombre del segundo parámetro
            param2_range: Rango de valores para param2
            base_params: Parámetros base (otros params fijos)
            metrics_to_plot: Lista de métricas para plotear

        Returns:
            Dict con DataFrames de resultados por métrica
        """
        logger.info(f"🔍 Ejecutando sensitivity analysis: {param1_name} vs {param2_name}")

        # Base params
        if base_params is None:
            base_params = {
                'atr_multi': 0.3,
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

        # Grid search
        results = {metric: pd.DataFrame(index=param1_range, columns=param2_range)
                   for metric in metrics_to_plot}

        total_combos = len(param1_range) * len(param2_range)
        combo_count = 0

        for p1_val in param1_range:
            for p2_val in param2_range:
                combo_count += 1

                # Crear params
                params = base_params.copy()
                params[param1_name] = p1_val
                params[param2_name] = p2_val

                logger.info(
                    f"   Combo {combo_count}/{total_combos}: {param1_name}={p1_val}, {param2_name}={p2_val}")

                # Run backtest
                result = self.backtester.run_optimized_backtest(dfs, params)
                metrics = result['metrics']

                # Almacenar resultados
                for metric in metrics_to_plot:
                    value = metrics.get(metric, 0)
                    results[metric].loc[p1_val, p2_val] = value

        # Convertir a float
        for metric in metrics_to_plot:
            results[metric] = results[metric].astype(float)

        # Plotear heatmaps
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 5))

        if len(metrics_to_plot) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics_to_plot):
            sns.heatmap(
                results[metric],
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                ax=axes[idx],
                cbar_kws={'label': metric}
            )
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[idx].set_xlabel(param2_name)
            axes[idx].set_ylabel(param1_name)

        plt.tight_layout()

        # Guardar
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sensitivity_{param1_name}_vs_{param2_name}_{timestamp}.png'
        filepath = OPT_RESULTS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Heatmap guardado: {filepath}")

        plt.close()

        # Identificar mejor combo
        best_sharpe_idx = results['sharpe_ratio'].stack(
        ).idxmax() if 'sharpe_ratio' in results else None
        if best_sharpe_idx:
            logger.info(
                f"🎯 Mejor combo (Sharpe): {param1_name}={best_sharpe_idx[0]}, {param2_name}={best_sharpe_idx[1]}")
            logger.info(f"   Sharpe: {results['sharpe_ratio'].loc[best_sharpe_idx]:.3f}")

        return results

    def efficient_frontier_params(
        self,
        dfs: Dict[str, pd.DataFrame],
        param_ranges: Dict[str, Tuple[float, float]],
        n_points: int = 20,
        base_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
        """
        Genera Efficient Frontier para parámetros

        Adapta Markowitz portfolio theory:
        - Risk (DD%) vs Sharpe
        - Encuentra frontera óptima de trade-offs

        Args:
            dfs: DataFrames multi-TF
            param_ranges: Dict de (min, max) por parámetro a optimizar
            n_points: Número de puntos en la frontera
            base_params: Parámetros base

        Returns:
            Tuple (risks, sharpes, param_combinations)
        """
        logger.info(f"📊 Calculando Efficient Frontier con {n_points} puntos")

        if base_params is None:
            base_params = {
                'va_percent': 0.7,
                'vp_rows': 120,
                'min_confidence': 0.6,
                'ema_fast_5m': 18,
                'ema_slow_5m': 48,
                'ema_fast_15m': 18,
                'ema_slow_15m': 48,
                'ema_fast_1h': 95,
                'ema_slow_1h': 210
            }

        risks = []
        sharpes = []
        param_combinations = []

        # Generar combinaciones aleatorias (Monte Carlo style)
        for i in range(n_points):
            # Sample params
            params = base_params.copy()

            for param_name, (min_val, max_val) in param_ranges.items():
                params[param_name] = np.random.uniform(min_val, max_val)

            logger.info(f"   Punto {i+1}/{n_points}: {params}")

            # Run backtest
            result = self.backtester.run_optimized_backtest(dfs, params)
            metrics = result['metrics']

            risk = metrics.get('max_drawdown', 1.0)
            sharpe = metrics.get('sharpe_ratio', 0.0)

            risks.append(risk)
            sharpes.append(sharpe)
            param_combinations.append(params)

        # Plot Efficient Frontier
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(risks, sharpes, c=sharpes, cmap='RdYlGn', s=100, alpha=0.6)
        plt.colorbar(scatter, label='Sharpe Ratio')

        # Marcar punto óptimo (max Sharpe)
        best_idx = np.argmax(sharpes)
        plt.scatter(risks[best_idx], sharpes[best_idx], c='red', s=200, marker='*',
                    label=f'Óptimo (Sharpe={sharpes[best_idx]:.2f}, DD={risks[best_idx]:.1%})')

        plt.xlabel('Risk (Max Drawdown %)')
        plt.ylabel('Sharpe Ratio')
        plt.title('Efficient Frontier - Parameter Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Guardar
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'efficient_frontier_{timestamp}.png'
        filepath = OPT_RESULTS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Efficient Frontier guardado: {filepath}")

        plt.close()

        # Log mejor resultado
        logger.info(f"🎯 Punto óptimo encontrado:")
        logger.info(f"   Sharpe: {sharpes[best_idx]:.3f}")
        logger.info(f"   Max DD: {risks[best_idx]:.1%}")
        logger.info(f"   Params: {param_combinations[best_idx]}")

        return risks, sharpes, param_combinations

    def bayes_opt_sharpe(
        self,
        dfs: Dict[str, pd.DataFrame],
        param_bounds: List[Tuple[float, float]],
        param_names: List[str] = ['atr_multi', 'vol_thresh', 'tp_rr'],
        n_calls: int = 50,
        base_params: Optional[Dict[str, Any]] = None,
        target_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimización Bayesiana para maximizar Sharpe/Calmar

        Args:
            dfs: DataFrames multi-TF
            param_bounds: Lista de (min, max) por parámetro
            param_names: Nombres de parámetros a optimizar
            n_calls: Número de llamadas al optimizer
            base_params: Parámetros base fijos
            target_metric: 'sharpe_ratio' o 'calmar_ratio'

        Returns:
            Dict con best_params y best_score
        """
        logger.info(f"🤖 Iniciando Bayesian Optimization ({n_calls} calls)")
        logger.info(f"   Target metric: {target_metric}")
        logger.info(f"   Params: {param_names}")

        if base_params is None:
            base_params = {
                'va_percent': 0.7,
                'vp_rows': 120,
                'min_confidence': 0.6,
                'ema_fast_5m': 18,
                'ema_slow_5m': 48,
                'ema_fast_15m': 18,
                'ema_slow_15m': 48,
                'ema_fast_1h': 95,
                'ema_slow_1h': 210
            }

        # Define objective function
        def objective(param_values):
            # Crear params
            params = base_params.copy()
            for name, value in zip(param_names, param_values):
                params[name] = float(value)  # Asegurar que es float, no lista

            # Run backtest
            result = self.backtester.run_optimized_backtest(dfs, params)
            metrics = result['metrics']

            # Retornar métrica negativa (gp_minimize minimiza)
            score = metrics.get(target_metric, 0)

            logger.info(
                f"   Eval: {dict(zip(param_names, param_values))} → {target_metric}={score:.3f}")

            return -score  # Negativo porque gp_minimize minimiza

        # Create search space
        space = [Real(low, high, name=name)
                 for (low, high), name in zip(param_bounds, param_names)]

        # Run optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=42,
            verbose=False
        )

        # Extract best params
        best_params = base_params.copy()
        for name, value in zip(param_names, result.x):
            best_params[name] = float(value)

        best_score = -result.fun  # Convertir de vuelta a positivo

        logger.info(f"✅ Optimización completa!")
        logger.info(f"🎯 Mejor {target_metric}: {best_score:.3f}")
        logger.info(f"   Best params: {best_params}")

        # Guardar resultado
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dict = {
            'timestamp': timestamp,
            'target_metric': target_metric,
            'best_score': best_score,
            'best_params': best_params,
            'n_calls': n_calls
        }

        filename = f'bayes_opt_{target_metric}_{timestamp}.json'
        filepath = OPT_RESULTS_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)

        logger.info(f"💾 Resultado guardado: {filepath}")

        return result_dict

    def walk_forward_eval(
        self,
        dfs: Dict[str, pd.DataFrame],
        n_periods: int = 6,
        train_split: float = 0.7,
        param_bounds: List[Tuple[float, float]] = [(0.1, 0.5), (0.8, 1.5), (1.8, 2.6)],
        param_names: List[str] = ['atr_multi', 'vol_thresh', 'tp_rr']
    ) -> pd.DataFrame:
        """
        Walk-Forward Optimization

        Args:
            dfs: DataFrames multi-TF completos
            n_periods: Número de períodos (default 6)
            train_split: Porcentaje para training (default 70%)
            param_bounds: Bounds para optimización
            param_names: Parámetros a optimizar

        Returns:
            DataFrame con resultados por período
        """
        logger.info(f"🚶 Iniciando Walk-Forward Optimization ({n_periods} períodos)")

        df_5m = dfs['5m']
        total_bars = len(df_5m)
        period_size = total_bars // n_periods

        results = []

        for period_idx in range(n_periods):
            start_idx = period_idx * period_size
            end_idx = min((period_idx + 1) * period_size, total_bars)

            train_size = int((end_idx - start_idx) * train_split)
            train_end = start_idx + train_size

            logger.info(f"\n📅 Período {period_idx + 1}/{n_periods}")
            logger.info(f"   Train: {start_idx} → {train_end} ({train_size} bars)")
            logger.info(f"   Test: {train_end} → {end_idx} ({end_idx - train_end} bars)")

            # Split data
            train_dfs = {
                '5m': df_5m.iloc[start_idx:train_end],
                '15m': dfs['15m'].iloc[start_idx // 3:train_end // 3],
                '1h': dfs['1h'].iloc[start_idx // 12:train_end // 12]
            }

            test_dfs = {
                '5m': df_5m.iloc[train_end:end_idx],
                '15m': dfs['15m'].iloc[train_end // 3:end_idx // 3],
                '1h': dfs['1h'].iloc[train_end // 12:end_idx // 12]
            }

            # Optimize on train
            logger.info("   🔧 Optimizando en train set...")
            opt_result = self.bayes_opt_sharpe(
                train_dfs,
                param_bounds,
                param_names,
                n_calls=20,  # Reducido para velocidad
                target_metric='calmar_ratio'
            )

            best_params = opt_result['best_params']
            train_score = opt_result['best_score']

            # Test on OOS
            logger.info("   📊 Testing en OOS set...")
            result = self.backtester.run_optimized_backtest(test_dfs, best_params)
            test_metrics = result['metrics']

            test_win_rate = test_metrics.get('win_rate', 0)
            test_sharpe = test_metrics.get('sharpe_ratio', 0)
            test_calmar = test_metrics.get('calmar_ratio', 0)
            test_trades = test_metrics.get('total_trades', 0)

            # Degradation
            degradation = train_score - test_calmar

            logger.info(f"   ✅ Resultados:")
            logger.info(f"      Train Calmar: {train_score:.3f}")
            logger.info(f"      Test Calmar: {test_calmar:.3f}")
            logger.info(f"      Test Win Rate: {test_win_rate:.1%}")
            logger.info(f"      Test Sharpe: {test_sharpe:.3f}")
            logger.info(f"      Degradation: {degradation:.3f}")

            # Guardar resultado
            results.append({
                'period': period_idx + 1,
                'train_start': start_idx,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': end_idx,
                'train_calmar': train_score,
                'test_calmar': test_calmar,
                'test_sharpe': test_sharpe,
                'test_win_rate': test_win_rate,
                'test_trades': test_trades,
                'degradation': degradation,
                'best_params': str(best_params)
            })

        # Crear DataFrame
        results_df = pd.DataFrame(results)

        # Estadísticas
        avg_test_calmar = results_df['test_calmar'].mean()
        avg_degradation = results_df['degradation'].mean()
        avg_win_rate = results_df['test_win_rate'].mean()

        logger.info(f"\n📊 RESUMEN WALK-FORWARD:")
        logger.info(f"   Avg Test Calmar: {avg_test_calmar:.3f}")
        logger.info(f"   Avg Win Rate: {avg_win_rate:.1%}")
        logger.info(f"   Avg Degradation: {avg_degradation:.3f}")

        # Plot resultados
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Calmar Train vs Test
        x = results_df['period']
        axes[0].plot(x, results_df['train_calmar'], 'o-', label='Train Calmar', linewidth=2)
        axes[0].plot(x, results_df['test_calmar'], 's-', label='Test Calmar (OOS)', linewidth=2)
        axes[0].axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Target (2.0)')
        axes[0].set_xlabel('Period')
        axes[0].set_ylabel('Calmar Ratio')
        axes[0].set_title('Walk-Forward: Train vs Test Performance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Win Rate y Trades
        ax2 = axes[1]
        ax2.bar(x, results_df['test_win_rate'], alpha=0.6, label='Win Rate')
        ax2.axhline(y=0.58, color='green', linestyle='--', alpha=0.5, label='Target (58%)')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Win Rate')
        ax2.set_title('Walk-Forward: OOS Win Rate por Período')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Guardar
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'walk_forward_{timestamp}.png'
        filepath = OPT_RESULTS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Walk-Forward plot guardado: {filepath}")

        plt.close()

        # Guardar CSV
        csv_filename = f'walk_forward_results_{timestamp}.csv'
        csv_filepath = OPT_RESULTS_DIR / csv_filename
        results_df.to_csv(csv_filepath, index=False)
        logger.info(f"💾 Resultados CSV: {csv_filepath}")

        return results_df


# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🚀 Optimizer Module - Testing")
    print("=" * 60)

    # Este script necesita datos reales para funcionar
    # Ver ejemplo de uso en scripts/run_optimization.py

    print("\n✅ Optimizer listo para usar")
    print("\nMétodos disponibles:")
    print("  - param_sensitivity_heatmap(): Grid search + heatmaps")
    print("  - efficient_frontier_params(): Risk/Sharpe frontier")
    print("  - bayes_opt_sharpe(): Bayesian optimization")
    print("  - walk_forward_eval(): Walk-forward validation")
