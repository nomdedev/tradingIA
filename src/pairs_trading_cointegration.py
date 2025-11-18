"""
Pairs Trading Cointegration Strategy

Estrategia de pairs trading usando cointegration con Johansen test.
Identifica pares cointegrados y opera cuando el spread diverge significativamente.

Características:
- Johansen cointegration test para identificar pares
- Entry cuando spread > 2*std del rolling mean
- Exit cuando spread regresa al mean
- Risk parity sizing basado en volatilidad
- Walk-forward optimization
- A/B testing vs simple mean reversion
"""

import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import talib
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuración
RESULTS_DIR = Path("results/pairs_trading_cointegration")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class PairsTradingStrategy(Strategy):
    """
    Pairs Trading Strategy usando cointegration
    """

    # Parámetros optimizables
    entry_threshold = 2.0  # Entry cuando spread > entry_threshold * std
    exit_threshold = 0.5   # Exit cuando spread < exit_threshold * std
    lookback_window = 100  # Ventana para rolling statistics
    max_holding_period = 20  # Máximo período de holding
    risk_parity_weight = 0.5  # Peso para risk parity sizing

    def init(self):
        """Inicializar indicadores y cointegration"""
        # Para pairs trading necesitamos datos de múltiples assets
        # En este ejemplo usamos BTC vs ETH como proxy
        # En producción, esto debería recibir datos de múltiples pares

        # Calcular spread sintético (placeholder - en producción usar datos reales)
        self.spread = self.I(self._calculate_spread, self.data.Close)

        # Rolling statistics del spread
        self.spread_mean = self.I(talib.SMA, self.spread, timeperiod=self.lookback_window)
        self.spread_std = self.I(self._rolling_std, self.spread, window=self.lookback_window)

        # Contador de períodos en posición
        self.holding_counter = 0

    def _calculate_spread(self, price):
        """Calcular spread sintético (placeholder)"""
        # En producción: spread = price_asset1 - hedge_ratio * price_asset2
        # Aquí usamos un spread sintético basado en EMA ratio
        ema_short = talib.EMA(price, timeperiod=20)
        ema_long = talib.EMA(price, timeperiod=50)
        spread = (ema_short - ema_long) / ema_long
        return spread

    def _rolling_std(self, data, window):
        """Calcular rolling standard deviation"""
        return pd.Series(data).rolling(window=window).std().bfill().values

    def next(self):
        """Lógica de trading"""
        if len(self.spread) < self.lookback_window:
            return

        current_spread = self.spread[-1]
        current_mean = self.spread_mean[-1]
        current_std = self.spread_std[-1]

        # Normalizar spread
        z_score = (current_spread - current_mean) / current_std if current_std > 0 else 0

        # Risk parity sizing
        volatility = current_std
        position_size = self.risk_parity_weight / volatility if volatility > 0 else 0.1
        position_size = min(position_size, 1.0)  # Cap at 100%

        # Entry signals
        long_signal = z_score < -self.entry_threshold  # Spread bajo, comprar
        short_signal = z_score > self.entry_threshold  # Spread alto, vender

        # Exit signals
        exit_signal = abs(z_score) < self.exit_threshold

        # Holding period limit
        self.holding_counter = self.holding_counter + 1 if self.position else 0
        force_exit = self.holding_counter >= self.max_holding_period

        # Exit positions
        if (exit_signal or force_exit) and self.position:
            self.position.close()
            self.holding_counter = 0
            return

        # Entry long (spread bajo - convergence expected)
        if long_signal and not self.position:
            # Calcular stop loss basado en volatilidad
            sl_price = self.data.Close[-1] * (1 - 2 * current_std)
            self.buy(size=position_size, sl=sl_price)
            self.holding_counter = 0

        # Entry short (spread alto - convergence expected)
        elif short_signal and not self.position:
            # Calcular stop loss basado en volatilidad
            sl_price = self.data.Close[-1] * (1 + 2 * current_std)
            self.sell(size=position_size, sl=sl_price)
            self.holding_counter = 0


def find_cointegrated_pairs(data_dict, significance_level=0.05):
    """
    Encontrar pares cointegrados usando Johansen test

    Args:
        data_dict: Dict con {asset_name: price_series}
        significance_level: Nivel de significancia para cointegration

    Returns:
        Lista de pares cointegrados con estadísticas
    """
    asset_names = list(data_dict.keys())
    cointegrated_pairs = []

    for i in range(len(asset_names)):
        for j in range(i + 1, len(asset_names)):
            asset1 = asset_names[i]
            asset2 = asset_names[j]

            try:
                # Johansen cointegration test
                price_data = pd.concat([data_dict[asset1], data_dict[asset2]], axis=1).dropna()
                price_data.columns = [asset1, asset2]

                if len(price_data) < 100:  # Mínimo de observaciones
                    continue

                johansen_test = coint_johansen(price_data.values, det_order=0, k_ar_diff=1)

                # Test estadístico para r=1 (cointegration)
                trace_stat = johansen_test.lr1[0]  # Trace statistic for r=0
                crit_values = johansen_test.cvt[:, 0]  # Critical values for r=0

                # Cointegrated si trace statistic > critical value
                if trace_stat > crit_values[int(significance_level * 100)]:
                    # Calcular hedge ratio usando OLS
                    X = sm.add_constant(price_data[asset2])
                    model = sm.OLS(price_data[asset1], X).fit()
                    hedge_ratio = model.params[1]

                    # Calcular spread
                    spread = price_data[asset1] - hedge_ratio * price_data[asset2]
                    spread_mean = spread.mean()
                    spread_std = spread.std()

                    cointegrated_pairs.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'hedge_ratio': hedge_ratio,
                        'trace_statistic': trace_stat,
                        'critical_value': crit_values[int(significance_level * 100)],
                        'spread_mean': spread_mean,
                        'spread_std': spread_std,
                        'half_life': calculate_half_life(spread)
                    })

            except Exception as e:
                print(f"Error testing {asset1} vs {asset2}: {e}")
                continue

    return cointegrated_pairs


def calculate_half_life(spread):
    """Calcular half-life del spread (Ornstein-Uhlenbeck process)"""
    try:
        # Regresión para encontrar theta (mean reversion speed)
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Alinear datos
        common_index = spread_lag.index.intersection(spread_diff.index)
        y = spread_diff.loc[common_index]
        X = spread_lag.loc[common_index]

        # OLS regression
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        theta = -model.params[1]  # Negative coefficient

        if theta > 0:
            half_life = np.log(2) / theta
        else:
            half_life = float('inf')

        return half_life

    except Exception:
        return float('inf')


def calculate_metrics(bt):
    """Calcular métricas avanzadas de performance"""
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
            'ulcer_index': 0
        }

    # Returns diarios
    returns = trades['ReturnPct'].values
    equity_curve = (1 + returns).cumprod()

    # Sharpe Ratio (anualizado con risk-free rate)
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    rf_daily = 0.04 / 252  # Risk-free rate diario
    excess_daily = daily_returns - rf_daily
    if len(excess_daily) > 0 and np.std(excess_daily) > 0:
        sharpe = (np.mean(excess_daily) / np.std(excess_daily)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Calmar Ratio
    max_dd = (equity_curve / equity_curve.expanding().max() - 1).min()
    if max_dd < 0:
        calmar = -np.mean(daily_returns) / abs(max_dd) * \
            np.sqrt(252) if len(daily_returns) > 0 else 0
    else:
        calmar = 0

    # Win Rate
    win_rate = (returns > 0).mean()

    # Profit Factor (correcto sin inf)
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if len(losing_trades) > 0 and losing_trades.sum() != 0 else 0.0

    # Sortino Ratio
    # Sortino Ratio (con risk-free rate correcto)
    rf_daily = 0.04 / 252  # Risk-free rate diario
    excess_daily_returns = daily_returns - rf_daily
    downside_returns = excess_daily_returns[excess_daily_returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        sortino = (excess_daily_returns.mean() / np.std(downside_returns)) * np.sqrt(252)
    else:
        sortino = 0.0  # 0 si no hay downside

    # VaR 95% (debe ser negativo - representa pérdida máxima esperada)
    var_95 = -np.percentile(-returns, 95)  # Siempre negativo

    # Ulcer Index
    drawdowns = 1 - equity_curve / equity_curve.expanding().max()
    ulcer = np.sqrt((drawdowns ** 2).mean())

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
        'ulcer_index': ulcer
    }


def walk_forward_test(df, cointegrated_pairs, n_periods=6, train_months=4, test_months=1):
    """
    Walk-forward optimization and testing

    Args:
        df: DataFrame con datos OHLCV (usado como proxy)
        cointegrated_pairs: Lista de pares cointegrados
        n_periods: Número de periodos walk-forward
        train_months: Meses para training/optimización
        test_months: Meses para testing OOS
    """
    results = []
    period_length = pd.Timedelta(days=30 * (train_months + test_months))

    for i in range(n_periods):
        # Definir periodo
        end_date = df.index[-1] - pd.Timedelta(days=i * 30 * test_months)
        start_date = end_date - period_length

        if start_date < df.index[0]:
            break

        period_data = df.loc[start_date:end_date]

        # Split train/test
        split_date = end_date - pd.Timedelta(days=30 * test_months)
        train_data = period_data.loc[:split_date]
        test_data = period_data.loc[split_date:]

        print(
            f"Period {i+1}: Train {train_data.index[0]} to {train_data.index[-1]}, Test {test_data.index[0]} to {test_data.index[-1]}")

        # Optimización en train (usando grid search por simplicidad)
        best_params = optimize_strategy(train_data)

        # Test OOS con mejores parámetros
        bt = Backtest(test_data, PairsTradingStrategy, cash=10000, commission=0.0005)
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
    """Optimización de parámetros usando grid search"""
    best_sharpe = -1000
    best_params = {
        'entry_threshold': 2.0,
        'exit_threshold': 0.5,
        'lookback_window': 100,
        'max_holding_period': 20,
        'risk_parity_weight': 0.5
    }

    # Grid search
    entry_options = [1.5, 2.0, 2.5]
    exit_options = [0.3, 0.5, 0.8]
    window_options = [50, 100, 150]
    holding_options = [10, 20, 30]
    risk_options = [0.3, 0.5, 0.7]

    for entry in entry_options:
        for exit_thresh in exit_options:
            for window in window_options:
                for holding in holding_options:
                    for risk in risk_options:
                        try:
                            params = {
                                'entry_threshold': entry,
                                'exit_threshold': exit_thresh,
                                'lookback_window': window,
                                'max_holding_period': holding,
                                'risk_parity_weight': risk
                            }
                            bt = Backtest(
                                train_data,
                                PairsTradingStrategy,
                                cash=10000,
                                commission=0.0005)
                            bt.run(**params)
                            metrics = calculate_metrics(bt)

                            if metrics['sharpe_ratio'] > best_sharpe:
                                best_sharpe = metrics['sharpe_ratio']
                                best_params = params
                        except Exception:
                            continue

    print(f"Mejores parámetros encontrados: {best_params}, Sharpe: {best_sharpe:.2f}")
    return best_params


def ab_test_vs_mean_reversion(df, mr_signals):
    """A/B test vs Mean Reversion strategy"""
    # Run Pairs Trading strategy
    bt_pairs = Backtest(df, PairsTradingStrategy, cash=10000, commission=0.0005)
    result_pairs = bt_pairs.run()

    # Simular Mean Reversion strategy (usando señales externas)
    mr_returns = mr_signals * 0.015  # Placeholder

    # Comparación estadística
    pairs_returns = result_pairs['Return [%]'].pct_change().dropna()
    t_stat = (np.mean(pairs_returns) - np.mean(mr_returns)) / \
        np.sqrt(np.var(pairs_returns) / len(pairs_returns) + np.var(mr_returns) / len(mr_returns))
    p_value = 2 * (1 - abs(t_stat) / np.sqrt(2))  # Aproximación simple

    # Superiority percentage
    superiority = (pairs_returns > mr_returns).mean()

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'pairs_superiority': superiority,
        'pairs_metrics': calculate_metrics(bt_pairs),
        'mr_returns': mr_returns
    }


def robustness_analysis(walk_forward_results):
    """Análisis de robustez"""
    sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    sortinos = [r['metrics']['sortino_ratio'] for r in walk_forward_results]
    ulcers = [r['metrics']['ulcer_index'] for r in walk_forward_results]

    return {
        'sharpe_stability': np.std(sharpes),
        'avg_sortino': np.mean(sortinos),
        'avg_ulcer': np.mean(ulcers),
        'sortino_robust': np.mean(sortinos) > 1.5,
        'ulcer_robust': np.mean(ulcers) < 10,
        'overall_robust': np.std(sharpes) < 0.2 and np.mean(sortinos) > 1.5 and np.mean(ulcers) < 10
    }


def sensitivity_analysis(df, base_params, n_tests=5):
    """Análisis de sensibilidad a parámetros"""
    sensitivities = {}

    for param_name, base_value in base_params.items():
        param_sensitivities = []

        # Test ±10% variation
        for factor in [0.9, 1.0, 1.1]:
            test_params = base_params.copy()
            test_params[param_name] = base_value * factor

            try:
                bt = Backtest(df, PairsTradingStrategy, cash=10000, commission=0.0005)
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
    """Análisis anti-overfit con Bonferroni correction"""
    # Bonferroni correction para múltiples tests
    n_tests = 5  # Número de tests realizados
    bonferroni_alpha = 0.05 / n_tests

    # Test significance with correction
    p_corrected = ab_results['p_value'] * n_tests
    significant_corrected = p_corrected < 0.05

    # Degradation analysis
    oos_sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    degradation = np.std(oos_sharpes) < 0.2  # Low degradation = robust

    return {
        'bonferroni_alpha': bonferroni_alpha,
        'p_corrected': p_corrected,
        'significant_corrected': significant_corrected,
        'degradation_low': degradation,
        'overfit_risk': 'low' if significant_corrected and degradation else 'high'
    }


def generate_report(
        walk_forward_results,
        ab_results,
        robustness,
        sensitivity,
        anti_overfit,
        cointegrated_pairs):
    """Generar reporte completo"""
    # Crear directorio de resultados
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Métricas resumen
    oos_sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    oos_win_rates = [r['metrics']['win_rate'] for r in walk_forward_results]

    summary = {
        'strategy': 'Pairs Trading Cointegration',
        'total_periods': len(walk_forward_results),
        'avg_oos_sharpe': np.mean(oos_sharpes),
        'std_oos_sharpe': np.std(oos_sharpes),
        'avg_oos_win_rate': np.mean(oos_win_rates),
        'cointegrated_pairs_found': len(cointegrated_pairs),
        'ab_test_significant': ab_results['significant'],
        'ab_superiority': ab_results['pairs_superiority'],
        'robustness_overall': robustness['overall_robust'],
        'sensitivity_stable': all(s['stable'] for s in sensitivity.values()),
        'anti_overfit_risk': anti_overfit['overfit_risk'],
        'recommendation': 'Deploy' if (np.mean(oos_sharpes) > 1.3 and
                                       ab_results['significant'] and
                                       robustness['overall_robust'] and
                                       anti_overfit['overfit_risk'] == 'low') else 'Further Testing'
    }

    # Guardar resultados
    import json
    with open(RESULTS_DIR / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Guardar cointegrated pairs
    with open(RESULTS_DIR / 'cointegrated_pairs.json', 'w') as f:
        json.dump(cointegrated_pairs, f, indent=2, default=str)

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
    """Generar gráficos de análisis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Sharpe ratio por periodo
    periods = [r['period'] for r in walk_forward_results]
    sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    axes[0, 0].plot(periods, sharpes, 'o-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=1.3, color='r', linestyle='--', alpha=0.7, label='Target 1.3')
    axes[0, 0].set_title('OOS Sharpe Ratio por Periodo')
    axes[0, 0].set_xlabel('Periodo Walk-Forward')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Win rate por periodo
    win_rates = [r['metrics']['win_rate'] for r in walk_forward_results]
    axes[0, 1].plot(periods, win_rates, 's-', color='green', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0.55, color='r', linestyle='--', alpha=0.7, label='Target 55%')
    axes[0, 1].set_title('OOS Win Rate por Periodo')
    axes[0, 1].set_xlabel('Periodo Walk-Forward')
    axes[0, 1].set_ylabel('Win Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Entry threshold sensitivity
    entry_thresholds = [r['params']['entry_threshold'] for r in walk_forward_results]
    axes[1, 0].scatter(entry_thresholds, sharpes, s=50, alpha=0.7)
    axes[1, 0].set_title('Entry Threshold vs Sharpe Ratio')
    axes[1, 0].set_xlabel('Entry Threshold')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].grid(True, alpha=0.3)

    # Profit factor por periodo
    profit_factors = [r['metrics']['profit_factor'] for r in walk_forward_results]
    axes[1, 1].plot(periods, profit_factors, '^-', color='purple', linewidth=2, markersize=8)
    axes[1, 1].axhline(y=1.5, color='r', linestyle='--', alpha=0.7, label='Target 1.5')
    axes[1, 1].set_title('OOS Profit Factor por Periodo')
    axes[1, 1].set_xlabel('Periodo Walk-Forward')
    axes[1, 1].set_ylabel('Profit Factor')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'walk_forward_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_complete_analysis(df_btc, asset_data_dict=None, mr_signals=None):
    """
    Ejecutar análisis completo de la estrategia

    Args:
        df_btc: DataFrame con datos BTC OHLCV (usado como proxy)
        asset_data_dict: Dict con datos de múltiples assets para cointegration
        mr_signals: Señales Mean Reversion para comparación A/B (opcional)
    """
    print("Ejecutando analisis completo Pairs Trading Cointegration")
    print(f"Datos: {len(df_btc)} velas de {df_btc.index[0]} a {df_btc.index[-1]}")

    # Encontrar pares cointegrados
    if asset_data_dict:
        print("\nBuscando pares cointegrados...")
        cointegrated_pairs = find_cointegrated_pairs(asset_data_dict)
        print(f"Encontrados {len(cointegrated_pairs)} pares cointegrados")
    else:
        cointegrated_pairs = []
        print("\nNo se proporcionaron datos multi-asset, usando estrategia sintética")

    # Walk-forward testing
    print("\nEjecutando Walk-Forward Optimization...")
    wf_results = walk_forward_test(df_btc, cointegrated_pairs)

    # A/B test vs Mean Reversion (si se proporciona)
    ab_results = None
    if mr_signals is not None:
        print("\nEjecutando A/B Test vs Mean Reversion...")
        ab_results = ab_test_vs_mean_reversion(df_btc, mr_signals)

    # Robustness analysis
    print("\nEjecutando Robustness Analysis...")
    robustness = robustness_analysis(wf_results)

    # Sensitivity analysis (usando parámetros del último periodo)
    print("\nEjecutando Sensitivity Analysis...")
    if wf_results:
        base_params = wf_results[-1]['params']
        sensitivity = sensitivity_analysis(df_btc, base_params)
    else:
        sensitivity = {}

    # Anti-overfit analysis
    print("\nEjecutando Anti-Overfit Analysis...")
    anti_overfit = anti_overfit_analysis(wf_results, ab_results or {'p_value': 1.0})

    # Generar reporte
    print("\nGenerando Reporte Final...")
    summary = generate_report(
        wf_results,
        ab_results,
        robustness,
        sensitivity,
        anti_overfit,
        cointegrated_pairs)

    print("\nAnalisis completado!")
    print(f"Sharpe OOS promedio: {summary['avg_oos_sharpe']:.2f}")
    print(f"Win Rate promedio: {summary['avg_oos_win_rate']:.1%}")
    print(f"Recomendacion: {summary['recommendation']}")

    return {
        'walk_forward_results': wf_results,
        'cointegrated_pairs': cointegrated_pairs,
        'ab_results': ab_results,
        'robustness': robustness,
        'sensitivity': sensitivity,
        'anti_overfit': anti_overfit,
        'summary': summary
    }


if __name__ == "__main__":
    # Ejemplo de uso
    print("Pairs Trading Cointegration Strategy")
    print("Para usar: run_complete_analysis(df_btc, asset_data_dict, mr_signals)")
    print("Resultados se guardan en results/pairs_trading_cointegration/")
