"""
Advanced Backtesting Framework
Framework avanzado para backtesting de estrategias de trading
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import DataFetcher
from src.indicators import TechnicalIndicators
from src.backtest_engine import BacktestEngine

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/advanced_backtesting.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AdvancedBacktestFramework:
    """Framework avanzado para backtesting"""

    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.indicators = TechnicalIndicators()
        self.backtest_engine = BacktestEngine()

    def run_walk_forward_analysis(self,
                                strategy_class,
                                symbol: str = "BTCUSD",
                                timeframe: str = "1H",
                                start_date: str = "2023-01-01",
                                end_date: str = None,
                                train_window: int = 90,  # días
                                test_window: int = 30,   # días
                                step_size: int = 7,      # días
                                **strategy_params):
        """
        Ejecutar análisis walk-forward

        Args:
            strategy_class: Clase de la estrategia a testear
            symbol: Símbolo del activo
            timeframe: Timeframe de los datos
            start_date: Fecha de inicio
            end_date: Fecha de fin (default: hoy)
            train_window: Ventana de entrenamiento en días
            test_window: Ventana de prueba en días
            step_size: Paso entre ventanas en días
            **strategy_params: Parámetros de la estrategia
        """

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Starting walk-forward analysis for {symbol}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Train window: {train_window} days, Test window: {test_window} days")

        # Cargar datos históricos
        data = self.data_fetcher.fetch_crypto_data(symbol, timeframe, start_date, end_date)
        if data.empty:
            raise ValueError("No data available for the specified period")

        # Calcular indicadores
        data = self.indicators.calculate_all_indicators(data)

        # Generar ventanas walk-forward
        windows = self._generate_walk_forward_windows(
            data, train_window, test_window, step_size
        )

        results = []
        for i, (train_data, test_data) in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")

            # Optimizar parámetros en datos de entrenamiento
            best_params = self._optimize_strategy_parameters(
                strategy_class, train_data, **strategy_params
            )

            # Evaluar en datos de prueba
            test_result = self._evaluate_strategy_on_window(
                strategy_class, test_data, best_params
            )

            results.append({
                'window': i+1,
                'train_period': {
                    'start': train_data.index[0].strftime('%Y-%m-%d'),
                    'end': train_data.index[-1].strftime('%Y-%m-%d')
                },
                'test_period': {
                    'start': test_data.index[0].strftime('%Y-%m-%d'),
                    'end': test_data.index[-1].strftime('%Y-%m-%d')
                },
                'best_params': best_params,
                'test_results': test_result
            })

        # Calcular estadísticas agregadas
        summary = self._calculate_walk_forward_summary(results)

        return {
            'results': results,
            'summary': summary,
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'total_windows': len(windows),
                'train_window_days': train_window,
                'test_window_days': test_window,
                'step_size_days': step_size,
                'execution_time': datetime.now().isoformat()
            }
        }

    def _generate_walk_forward_windows(self, data: pd.DataFrame,
                                     train_window: int, test_window: int, step_size: int):
        """Generar ventanas walk-forward"""
        windows = []
        start_date = data.index[0]
        end_date = data.index[-1]

        current_train_end = start_date + timedelta(days=train_window)

        while current_train_end + timedelta(days=test_window) <= end_date:
            train_end = current_train_end
            test_end = train_end + timedelta(days=test_window)

            train_data = data.loc[start_date:train_end]
            test_data = data.loc[train_end:train_end + timedelta(days=test_window)]

            if len(train_data) > 100 and len(test_data) > 50:  # Mínimo de datos
                windows.append((train_data, test_data))

            current_train_end += timedelta(days=step_size)

        return windows

    def _optimize_strategy_parameters(self, strategy_class, train_data: pd.DataFrame, **param_ranges):
        """Optimizar parámetros de estrategia"""
        # Implementación simplificada - usar valores por defecto
        # En una implementación completa, usaríamos optimización bayesiana o grid search
        return {k: v[0] if isinstance(v, list) else v for k, v in param_ranges.items()}

    def _evaluate_strategy_on_window(self, strategy_class, test_data: pd.DataFrame, params: Dict):
        """Evaluar estrategia en ventana de prueba"""
        try:
            strategy = strategy_class(**params)
            result = self.backtest_engine.run_backtest(
                strategy=strategy,
                data=test_data,
                initial_capital=10000,
                risk_per_trade=0.02,
                max_open_trades=3
            )
            return result
        except Exception as e:
            logger.error(f"Error evaluating strategy: {e}")
            return {'error': str(e)}

    def _calculate_walk_forward_summary(self, results: List[Dict]):
        """Calcular estadísticas agregadas del walk-forward"""
        if not results:
            return {}

        # Extraer métricas de cada ventana
        returns = []
        sharpes = []
        win_rates = []
        max_drawdowns = []

        for result in results:
            test_res = result.get('test_results', {})
            if 'total_return' in test_res:
                returns.append(test_res['total_return'])
                sharpes.append(test_res.get('sharpe_ratio', 0))
                win_rates.append(test_res.get('win_rate', 0))
                max_drawdowns.append(test_res.get('max_drawdown', 0))

        if not returns:
            return {'error': 'No valid results found'}

        return {
            'total_windows': len(results),
            'successful_windows': len(returns),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_sharpe': np.mean(sharpes),
            'avg_win_rate': np.mean(win_rates),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'profit_factor': np.mean([r for r in returns if r > 0]) / abs(np.mean([r for r in returns if r < 0])) if any(r < 0 for r in returns) else float('inf'),
            'win_windows': sum(1 for r in returns if r > 0),
            'loss_windows': sum(1 for r in returns if r < 0)
        }

    def run_monte_carlo_analysis(self, strategy_class, data: pd.DataFrame,
                               num_simulations: int = 1000, **strategy_params):
        """Ejecutar análisis de Monte Carlo"""
        logger.info(f"Running Monte Carlo analysis with {num_simulations} simulations")

        results = []
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = []
            for i in range(num_simulations):
                # Crear variación aleatoria de parámetros
                varied_params = self._vary_parameters(strategy_params)
                future = executor.submit(self._run_single_monte_carlo, strategy_class, data, varied_params)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Monte Carlo simulation failed: {e}")

        # Analizar resultados
        analysis = self._analyze_monte_carlo_results(results)

        return {
            'simulations': results,
            'analysis': analysis,
            'metadata': {
                'num_simulations': num_simulations,
                'execution_time': datetime.now().isoformat()
            }
        }

    def _vary_parameters(self, params: Dict, variation: float = 0.1):
        """Variar parámetros aleatoriamente"""
        varied = {}
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Variar ±10%
                variation_amount = value * variation
                varied[key] = np.random.uniform(value - variation_amount, value + variation_amount)
            else:
                varied[key] = value
        return varied

    def _run_single_monte_carlo(self, strategy_class, data: pd.DataFrame, params: Dict):
        """Ejecutar una simulación individual de Monte Carlo"""
        try:
            strategy = strategy_class(**params)
            result = self.backtest_engine.run_backtest(
                strategy=strategy,
                data=data,
                initial_capital=10000,
                risk_per_trade=0.02,
                max_open_trades=3
            )
            return result
        except Exception as e:
            return {'error': str(e)}

    def _analyze_monte_carlo_results(self, results: List[Dict]):
        """Analizar resultados de Monte Carlo"""
        valid_results = [r for r in results if 'total_return' in r]

        if not valid_results:
            return {'error': 'No valid Monte Carlo results'}

        returns = [r['total_return'] for r in valid_results]
        sharpes = [r.get('sharpe_ratio', 0) for r in valid_results]

        return {
            'total_simulations': len(results),
            'valid_simulations': len(valid_results),
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'return_percentiles': {
                '5th': np.percentile(returns, 5),
                '25th': np.percentile(returns, 25),
                '50th': np.percentile(returns, 50),
                '75th': np.percentile(returns, 75),
                '95th': np.percentile(returns, 95)
            },
            'sharpe_stats': {
                'mean': np.mean(sharpes),
                'std': np.std(sharpes),
                'min': np.min(sharpes),
                'max': np.max(sharpes)
            },
            'probability_profit': sum(1 for r in returns if r > 0) / len(returns),
            'expected_return': np.mean(returns),
            'value_at_risk_95': np.percentile(returns, 5)  # 5th percentile
        }

def main():
    """Función principal para pruebas"""
    print("Advanced Backtesting Framework")
    print("==============================")

    framework = AdvancedBacktestFramework()

    # Ejemplo de uso - Walk Forward Analysis
    print("Ejecutando análisis walk-forward...")
    try:
        # Importar estrategia IFVG
        from strategies.ifvg_strategy import IFVGStrategy

        results = framework.run_walk_forward_analysis(
            strategy_class=IFVGStrategy,
            symbol="BTCUSD",
            timeframe="1H",
            start_date="2024-01-01",
            end_date="2024-06-01",
            train_window=60,
            test_window=15,
            step_size=7,
            atr_period=[100, 200, 300],
            atr_multiplier=[0.25, 0.5, 0.75]
        )

        print("Análisis completado exitosamente!")
        print(f"Resultados: {len(results['results'])} ventanas procesadas")

        # Guardar resultados
        with open('results/walk_forward_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("Resultados guardados en results/walk_forward_analysis.json")

    except Exception as e:
        print(f"Error en análisis: {e}")

if __name__ == "__main__":
    main()