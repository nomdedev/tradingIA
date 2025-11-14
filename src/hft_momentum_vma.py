"""
HFT Momentum Kalman VMA Strategy

Estrategia HFT de momentum usando Kalman filter para Variable Moving Average.
Optimizada para alta frecuencia con control de slippage y risk parity sizing.

Características:
- Kalman filter para VMA calculation
- Momentum signals con VMA crossover
- Risk parity sizing basado en volatilidad
- HFT optimizations: slippage control, latency simulation
- Walk-forward testing con re-optimización
- A/B testing vs simple momentum
- Robustness y anti-overfit analysis
"""

import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import talib
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Configuración
RESULTS_DIR = Path("results/hft_momentum_vma")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class KalmanFilter:
    """
    Kalman Filter para estimación de VMA (Variable Moving Average)
    """

    def __init__(self, process_noise=1e-5, measurement_noise=1e-3):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.x_hat = None  # State estimate
        self.P = None      # Error covariance
        self.first_run = True

    def update(self, measurement):
        """
        Actualizar filtro con nueva medición

        Args:
            measurement: Nueva observación del precio

        Returns:
            VMA estimada
        """
        if self.first_run:
            # Inicialización
            self.x_hat = measurement
            self.P = 1.0
            self.first_run = False
            return self.x_hat

        # Prediction step
        x_hat_minus = self.x_hat
        P_minus = self.P + self.process_noise

        # Update step
        K = P_minus / (P_minus + self.measurement_noise)  # Kalman gain
        self.x_hat = x_hat_minus + K * (measurement - x_hat_minus)
        self.P = (1 - K) * P_minus

        return self.x_hat


class HFTMomentumVMAStrategy(Strategy):
    """
    HFT Momentum Strategy usando Kalman VMA
    """

    # Parámetros optimizables
    kalman_process_noise = 1e-5
    kalman_measurement_noise = 1e-3
    momentum_threshold = 0.001  # Threshold para señal de momentum
    exit_threshold = 0.0005     # Threshold para exit
    risk_parity_weight = 0.3    # Peso para risk parity sizing
    max_holding_period = 5      # Máximo período de holding (HFT)
    slippage_bps = 2            # Slippage en basis points
    latency_ms = 50             # Latencia simulada en ms

    def init(self):
        """Inicializar indicadores y Kalman filter"""
        # Kalman filter para VMA
        self.kalman = KalmanFilter(
            process_noise=self.kalman_process_noise,
            measurement_noise=self.kalman_measurement_noise
        )

        # VMA usando Kalman filter
        self.vma = self.I(self._kalman_vma, self.data.Close)

        # Momentum signal (rate of change de VMA)
        self.momentum = self.I(self._calculate_momentum, self.vma)

        # Volatilidad para risk parity
        self.volatility = self.I(
            talib.ATR,
            self.data.High,
            self.data.Low,
            self.data.Close,
            timeperiod=10)

        # Contador de períodos en posición
        self.holding_counter = 0

        # Slippage y latency tracking
        self.slippage_costs = []
        self.latency_delays = []

    def _kalman_vma(self, price):
        """Calcular VMA usando Kalman filter"""
        vma_values = []
        for p in price:
            vma = self.kalman.update(p)
            vma_values.append(vma)
        return np.array(vma_values)

    def _calculate_momentum(self, vma):
        """Calcular momentum signal"""
        # Rate of change del VMA
        momentum = np.diff(vma, prepend=vma[0])
        return momentum

    def next(self):
        """Lógica de trading HFT"""
        if len(self.vma) < 2:
            return

        current_momentum = self.momentum[-1]
        current_volatility = self.volatility[-1] if not np.isnan(self.volatility[-1]) else 0.01

        # Simular latency (skip some signals)
        if np.random.random() < (self.latency_ms / 1000):  # Probabilidad de delay
            self.latency_delays.append(1)
            return

        # Risk parity sizing
        position_size = self.risk_parity_weight / current_volatility if current_volatility > 0 else 0.1
        position_size = min(position_size, 1.0)  # Cap at 100%

        # Entry signals
        long_signal = current_momentum > self.momentum_threshold
        short_signal = current_momentum < -self.momentum_threshold

        # Exit signals
        exit_signal = abs(current_momentum) < self.exit_threshold

        # Holding period limit (HFT)
        self.holding_counter = self.holding_counter + 1 if self.position else 0
        force_exit = self.holding_counter >= self.max_holding_period

        # Exit positions
        if (exit_signal or force_exit) and self.position:
            # Simular slippage en exit
            slippage = self._calculate_slippage()
            self.slippage_costs.append(slippage)

            self.position.close()
            self.holding_counter = 0
            return

        # Entry long
        if long_signal and not self.position:
            # Simular slippage en entry
            slippage = self._calculate_slippage()
            self.slippage_costs.append(slippage)

            # Calcular stop loss basado en volatilidad
            entry_price = self.data.Close[-1] * (1 + slippage / 10000)  # Aplicar slippage
            sl_price = entry_price * (1 - 2 * current_volatility)

            self.buy(size=position_size, sl=sl_price)
            self.holding_counter = 0

        # Entry short
        elif short_signal and not self.position:
            # Simular slippage en entry
            slippage = self._calculate_slippage()
            self.slippage_costs.append(slippage)

            # Calcular stop loss basado en volatilidad
            entry_price = self.data.Close[-1] * (1 - slippage / 10000)  # Aplicar slippage
            sl_price = entry_price * (1 + 2 * current_volatility)

            self.sell(size=position_size, sl=sl_price)
            self.holding_counter = 0

    def _calculate_slippage(self):
        """Calcular slippage simulado"""
        # Slippage basado en volatilidad del mercado
        base_slippage = self.slippage_bps
        vol_multiplier = np.random.normal(1.0, 0.2)  # Variación aleatoria
        return base_slippage * vol_multiplier


def calculate_metrics(bt):
    """Calcular métricas avanzadas incluyendo HFT metrics"""
    trades = bt.trades

    if len(trades) == 0:
        return {
            'sharpe_ratio': 0,
            'calmar_ratio': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'total_return': 0,
            'total_trades': 0,
            'avg_trade': 0,
            'sortino_ratio': 0,
            'var_95': 0,
            'ulcer_index': 0,
            'avg_holding_period': 0,
            'avg_slippage_bps': 0,
            'latency_impact': 0
        }

    # Returns diarios
    returns = trades['ReturnPct'].values
    equity_curve = (1 + returns).cumprod()

    # Sharpe Ratio (anualizado)
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(daily_returns) > 0 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0

    # Calmar Ratio
    max_dd = (equity_curve / equity_curve.expanding().max() - 1).min()
    if max_dd < 0:
        calmar = -np.mean(daily_returns) / abs(max_dd) * \
            np.sqrt(252) if len(daily_returns) > 0 else 0
    else:
        calmar = 0

    # Win Rate
    win_rate = (returns > 0).mean()

    # Profit Factor
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if len(losing_trades) > 0 else float('inf')

    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0:
        sortino = np.mean(daily_returns) / np.std(downside_returns) * np.sqrt(252)
    else:
        sortino = float('inf')

    # VaR 95%
    var_95 = np.percentile(returns, 5)

    # Ulcer Index
    drawdowns = 1 - equity_curve / equity_curve.expanding().max()
    ulcer = np.sqrt((drawdowns ** 2).mean())

    # HFT-specific metrics
    holding_periods = (trades['ExitTime'] - trades['EntryTime']
                       ).dt.total_seconds() / 60  # en minutos
    avg_holding_period = np.mean(holding_periods) if len(holding_periods) > 0 else 0

    # Slippage (simulado)
    strategy_instance = bt._strategy
    avg_slippage = np.mean(strategy_instance.slippage_costs) if hasattr(
        strategy_instance, 'slippage_costs') and strategy_instance.slippage_costs else 0

    # Latency impact
    latency_signals = len(
        strategy_instance.latency_delays) if hasattr(
        strategy_instance,
        'latency_delays') else 0
    total_signals = len(trades) + latency_signals
    latency_impact = latency_signals / total_signals if total_signals > 0 else 0

    return {
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'total_return': equity_curve[-1] - 1,
        'total_trades': len(trades),
        'avg_trade': np.mean(returns),
        'sortino_ratio': sortino,
        'var_95': var_95,
        'ulcer_index': ulcer,
        'avg_holding_period': avg_holding_period,
        'avg_slippage_bps': avg_slippage,
        'latency_impact': latency_impact
    }


def walk_forward_test(df, n_periods=6, train_hours=24, test_hours=6):
    """
    Walk-forward optimization and testing (HFT timeframe)

    Args:
        df: DataFrame con datos OHLCV
        n_periods: Número de periodos walk-forward
        train_hours: Horas para training/optimización
        test_hours: Horas para testing OOS
    """
    results = []
    period_length = pd.Timedelta(hours=train_hours + test_hours)

    for i in range(n_periods):
        # Definir periodo (HFT usa horas en lugar de meses)
        end_date = df.index[-1] - pd.Timedelta(hours=i * test_hours)
        start_date = end_date - period_length

        if start_date < df.index[0]:
            break

        period_data = df.loc[start_date:end_date]

        # Split train/test
        split_date = end_date - pd.Timedelta(hours=test_hours)
        train_data = period_data.loc[:split_date]
        test_data = period_data.loc[split_date:]

        print(
            f"Period {i+1}: Train {train_data.index[0]} to {train_data.index[-1]}, Test {test_data.index[0]} to {test_data.index[-1]}")

        # Optimización en train
        best_params = optimize_strategy(train_data)

        # Test OOS con mejores parámetros
        bt = Backtest(test_data, HFTMomentumVMAStrategy, cash=10000,
                      commission=0.0002)  # Lower commission for HFT
        result = bt.run(**best_params)

        # Calcular métricas
        metrics = calculate_metrics(bt)

        results.append({
            'period': i + 1,
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'params': best_params,
            'metrics': metrics,
            'backtest_result': result
        })

    return results


def optimize_strategy(train_data):
    """Optimización de parámetros para HFT"""
    def objective(momentum_thresh):
        """Función objetivo"""
        try:
            params = {
                'momentum_threshold': momentum_thresh,
                'exit_threshold': momentum_thresh * 0.5,
                'risk_parity_weight': 0.3,
                'max_holding_period': 5,
                'slippage_bps': 2,
                'latency_ms': 50
            }
            bt = Backtest(train_data, HFTMomentumVMAStrategy, cash=10000, commission=0.0002)
            bt.run(**params)
            metrics = calculate_metrics(bt)
            return -metrics['sharpe_ratio']  # Negativo para minimización
        except Exception:
            return 1000

    # Optimizar momentum threshold
    result = minimize_scalar(objective, bounds=(0.0005, 0.005), method='bounded')

    best_params = {
        'kalman_process_noise': 1e-5,
        'kalman_measurement_noise': 1e-3,
        'momentum_threshold': result.x,
        'exit_threshold': result.x * 0.5,
        'risk_parity_weight': 0.3,
        'max_holding_period': 5,
        'slippage_bps': 2,
        'latency_ms': 50
    }

    print(
        f"Mejores parámetros encontrados: momentum_threshold={result.x:.4f}, Sharpe: {-result.fun:.2f}")
    return best_params


def ab_test_vs_simple_momentum(df, simple_momentum_signals):
    """A/B test vs Simple Momentum strategy"""
    # Run HFT Kalman VMA strategy
    bt_hft = Backtest(df, HFTMomentumVMAStrategy, cash=10000, commission=0.0002)
    result_hft = bt_hft.run()

    # Simular Simple Momentum strategy
    simple_returns = simple_momentum_signals * 0.008  # Placeholder

    # Comparación estadística
    hft_returns = result_hft['Return [%]'].pct_change().dropna()
    t_stat = (np.mean(hft_returns) - np.mean(simple_returns)) / np.sqrt(np.var(hft_returns) / \
              len(hft_returns) + np.var(simple_returns) / len(simple_returns))
    p_value = 2 * (1 - abs(t_stat) / np.sqrt(2))  # Aproximación simple

    # Superiority percentage
    superiority = (hft_returns > simple_returns).mean()

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'hft_superiority': superiority,
        'hft_metrics': calculate_metrics(bt_hft),
        'simple_momentum_returns': simple_returns
    }


def robustness_analysis(walk_forward_results):
    """Análisis de robustez para HFT"""
    sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    sortinos = [r['metrics']['sortino_ratio'] for r in walk_forward_results]
    ulcers = [r['metrics']['ulcer_index'] for r in walk_forward_results]
    latencies = [r['metrics']['latency_impact'] for r in walk_forward_results]

    return {
        'sharpe_stability': np.std(sharpes),
        'avg_sortino': np.mean(sortinos),
        'avg_ulcer': np.mean(ulcers),
        'avg_latency_impact': np.mean(latencies),
        'sortino_robust': np.mean(sortinos) > 1.5,
        'ulcer_robust': np.mean(ulcers) < 10,
        'latency_robust': np.mean(latencies) < 0.3,  # <30% signals lost to latency
        'overall_robust': (np.std(sharpes) < 0.2 and np.mean(sortinos) > 1.5 and
                           np.mean(ulcers) < 10 and np.mean(latencies) < 0.3)
    }


def sensitivity_analysis(df, base_params, n_tests=5):
    """Análisis de sensibilidad para HFT"""
    sensitivities = {}

    for param_name, base_value in base_params.items():
        if param_name in ['kalman_process_noise', 'kalman_measurement_noise', 'momentum_threshold']:
            param_sensitivities = []

            # Test ±10% variation
            for factor in [0.9, 1.0, 1.1]:
                test_params = base_params.copy()
                test_params[param_name] = base_value * factor

                try:
                    bt = Backtest(df, HFTMomentumVMAStrategy, cash=10000, commission=0.0002)
                    bt.run(**test_params)
                    metrics = calculate_metrics(bt)
                    param_sensitivities.append(metrics['sharpe_ratio'])
                except Exception:
                    param_sensitivities.append(0)

            sensitivities[param_name] = {
                'values': param_sensitivities,
                'std': np.std(param_sensitivities),
                'stable': np.std(param_sensitivities) < 0.2
            }

    return sensitivities


def anti_overfit_analysis(walk_forward_results, ab_results):
    """Análisis anti-overfit para HFT"""
    # Bonferroni correction
    n_tests = 5
    bonferroni_alpha = 0.05 / n_tests

    # Test significance with correction
    p_corrected = ab_results['p_value'] * n_tests
    significant_corrected = p_corrected < 0.05

    # Degradation analysis
    oos_sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    degradation = np.std(oos_sharpes) < 0.2

    return {
        'bonferroni_alpha': bonferroni_alpha,
        'p_corrected': p_corrected,
        'significant_corrected': significant_corrected,
        'degradation_low': degradation,
        'overfit_risk': 'low' if significant_corrected and degradation else 'high'
    }


def generate_report(walk_forward_results, ab_results, robustness, sensitivity, anti_overfit):
    """Generar reporte completo para HFT"""
    # Crear directorio de resultados
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Métricas resumen
    oos_sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    oos_win_rates = [r['metrics']['win_rate'] for r in walk_forward_results]
    avg_holding_periods = [r['metrics']['avg_holding_period'] for r in walk_forward_results]
    avg_slippages = [r['metrics']['avg_slippage_bps'] for r in walk_forward_results]

    summary = {
        'strategy': 'HFT Momentum Kalman VMA',
        'total_periods': len(walk_forward_results),
        'avg_oos_sharpe': np.mean(oos_sharpes),
        'std_oos_sharpe': np.std(oos_sharpes),
        'avg_oos_win_rate': np.mean(oos_win_rates),
        'avg_holding_period_min': np.mean(avg_holding_periods),
        'avg_slippage_bps': np.mean(avg_slippages),
        'ab_test_significant': ab_results['significant'],
        'ab_superiority': ab_results['hft_superiority'],
        'robustness_overall': robustness['overall_robust'],
        'sensitivity_stable': all(s['stable'] for s in sensitivity.values()),
        'anti_overfit_risk': anti_overfit['overfit_risk'],
        'recommendation': 'Deploy' if (np.mean(oos_sharpes) > 1.5 and
                                       ab_results['significant'] and
                                       robustness['overall_robust'] and
                                       anti_overfit['overfit_risk'] == 'low') else 'Further Testing'
    }

    # Guardar resultados
    import json
    with open(RESULTS_DIR / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Guardar trades detallados
    all_trades = []
    for result in walk_forward_results:
        trades_df = result['backtest_result']._trades
        if len(trades_df) > 0:
            trades_df['period'] = result['period']
            all_trades.append(trades_df)

    if all_trades:
        trades_combined = pd.concat(all_trades)
        trades_combined.to_csv(RESULTS_DIR / 'trades.csv')

    # Generar gráficos
    generate_plots(walk_forward_results, summary)

    return summary


def generate_plots(walk_forward_results, summary):
    """Generar gráficos de análisis HFT"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Sharpe ratio por periodo
    periods = [r['period'] for r in walk_forward_results]
    sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    axes[0, 0].plot(periods, sharpes, 'o-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=1.5, color='r', linestyle='--', alpha=0.7, label='Target 1.5')
    axes[0, 0].set_title('OOS Sharpe Ratio por Periodo (HFT)')
    axes[0, 0].set_xlabel('Periodo Walk-Forward')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Average holding period
    holding_periods = [r['metrics']['avg_holding_period'] for r in walk_forward_results]
    axes[0, 1].plot(periods, holding_periods, 's-', color='green', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=5, color='r', linestyle='--', alpha=0.7, label='Max 5 min')
    axes[0, 1].set_title('Average Holding Period (min)')
    axes[0, 1].set_xlabel('Periodo Walk-Forward')
    axes[0, 1].set_ylabel('Holding Period (min)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Slippage analysis
    slippages = [r['metrics']['avg_slippage_bps'] for r in walk_forward_results]
    axes[1, 0].plot(periods, slippages, '^-', color='orange', linewidth=2, markersize=8)
    axes[1, 0].axhline(y=2, color='r', linestyle='--', alpha=0.7, label='Target <2bps')
    axes[1, 0].set_title('Average Slippage (bps)')
    axes[1, 0].set_xlabel('Periodo Walk-Forward')
    axes[1, 0].set_ylabel('Slippage (bps)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Latency impact
    latencies = [r['metrics']['latency_impact'] for r in walk_forward_results]
    axes[1, 1].plot(periods, latencies, 'd-', color='purple', linewidth=2, markersize=8)
    axes[1, 1].axhline(y=0.3, color='r', linestyle='--', alpha=0.7, label='Max 30%')
    axes[1, 1].set_title('Latency Impact (% signals lost)')
    axes[1, 1].set_xlabel('Periodo Walk-Forward')
    axes[1, 1].set_ylabel('Latency Impact')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'hft_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_complete_analysis(df_hft, simple_momentum_signals=None):
    """
    Ejecutar análisis completo de la estrategia HFT

    Args:
        df_hft: DataFrame con datos HFT OHLCV
        simple_momentum_signals: Señales simple momentum para comparación A/B
    """
    print("Ejecutando analisis completo HFT Momentum Kalman VMA")
    print(f"Datos HFT: {len(df_hft)} velas de {df_hft.index[0]} a {df_hft.index[-1]}")

    # Walk-forward testing (HFT timeframe)
    print("\nEjecutando Walk-Forward Optimization (HFT)...")
    wf_results = walk_forward_test(df_hft)

    # A/B test vs Simple Momentum
    ab_results = None
    if simple_momentum_signals is not None:
        print("\nEjecutando A/B Test vs Simple Momentum...")
        ab_results = ab_test_vs_simple_momentum(df_hft, simple_momentum_signals)

    # Robustness analysis
    print("\nEjecutando Robustness Analysis...")
    robustness = robustness_analysis(wf_results)

    # Sensitivity analysis
    print("\nEjecutando Sensitivity Analysis...")
    if wf_results:
        base_params = wf_results[-1]['params']
        sensitivity = sensitivity_analysis(df_hft, base_params)
    else:
        sensitivity = {}

    # Anti-overfit analysis
    print("\nEjecutando Anti-Overfit Analysis...")
    anti_overfit = anti_overfit_analysis(wf_results, ab_results or {'p_value': 1.0})

    # Generar reporte
    print("\nGenerando Reporte Final...")
    summary = generate_report(wf_results, ab_results, robustness, sensitivity, anti_overfit)

    print("\nAnalisis HFT completado!")
    print(f"Sharpe OOS promedio: {summary['avg_oos_sharpe']:.2f}")
    print(f"Win Rate promedio: {summary['avg_oos_win_rate']:.1%}")
    print(f"Holding Period promedio: {summary['avg_holding_period_min']:.1f} min")
    print(f"Slippage promedio: {summary['avg_slippage_bps']:.1f} bps")
    print(f"Recomendacion: {summary['recommendation']}")

    return {
        'walk_forward_results': wf_results,
        'ab_results': ab_results,
        'robustness': robustness,
        'sensitivity': sensitivity,
        'anti_overfit': anti_overfit,
        'summary': summary
    }


if __name__ == "__main__":
    # Ejemplo de uso
    print("HFT Momentum Kalman VMA Strategy")
    print("Para usar: run_complete_analysis(df_hft, simple_momentum_signals)")
    print("Resultados se guardan en results/hft_momentum_vma/")
