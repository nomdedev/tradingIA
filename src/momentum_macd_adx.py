"""
Momentum MACD + ADX Strategy

Estrategia de momentum usando MACD + ADX con filtros de volumen y trend HTF.
Sharpe OOS objetivo: 1.5, Win Rate: 55-65%.

Características:
- MACD hist > 0 (12/26/9)
- ADX > 25 (strong trend)
- Vol > 1.2 * SMA21
- HTF EMA210 uptrend
- Entry long/short con momentum
- SL = 1.5 * ATR, TP = 2.2R
"""

import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from backtesting.lib import resample_apply
import talib
import json
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Placeholder para optimización (implementar con grid search si no hay scikit-optimize)
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("scikit-optimize no disponible, usando grid search")

# Configuración
RESULTS_DIR = Path("results/momentum_macd_adx")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class MACDADxStrategy(Strategy):
    """
    Momentum Strategy usando MACD + ADX
    """

    # Parámetros optimizables
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    adx_period = 14
    adx_thresh = 25.0
    vol_multiplier = 1.2
    atr_period = 14
    sl_atr_mult = 1.5
    tp_rr_mult = 2.2

    def init(self):
        """Inicializar indicadores"""
        # MACD
        self.macd, self.macdsignal, self.macdhist = self.I(
            talib.MACD, self.data.Close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # ADX
        self.adx = self.I(
            talib.ADX,
            self.data.High,
            self.data.Low,
            self.data.Close,
            timeperiod=self.adx_period)

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=21)

        # ATR para SL/TP
        self.atr = self.I(
            talib.ATR,
            self.data.High,
            self.data.Low,
            self.data.Close,
            timeperiod=self.atr_period)

        # HTF EMA210 (resampleado a 1h)
        def ema_wrapper(data):
            return talib.EMA(data, timeperiod=210)
        self.ema210_htf = resample_apply('1H', ema_wrapper, self.data.Close)

    def next(self):
        """Lógica de trading"""
        # Condiciones de entrada
        macd_bullish = self.macdhist[-1] > 0
        macd_bearish = self.macdhist[-1] < 0
        adx_trend = self.adx[-1] > self.adx_thresh
        vol_confirm = self.data.Volume[-1] > (self.vol_multiplier * self.vol_sma[-1])
        htf_uptrend = self.ema210_htf[-1] > self.ema210_htf[-2] if len(
            self.ema210_htf) > 1 else True
        htf_downtrend = self.ema210_htf[-1] < self.ema210_htf[-2] if len(
            self.ema210_htf) > 1 else False

        # Entry Long
        if (macd_bullish and adx_trend and vol_confirm and htf_uptrend and
                not self.position.is_long and not self.position.is_short):
            # Calcular SL/TP
            entry_price = self.data.Close[-1]
            atr_value = self.atr[-1]
            sl_price = entry_price - (self.sl_atr_mult * atr_value)
            tp_price = entry_price + (self.tp_rr_mult * (entry_price - sl_price))

            # Entrar posición long
            self.buy(sl=sl_price, tp=tp_price)

        # Entry Short
        elif (macd_bearish and adx_trend and vol_confirm and htf_downtrend and
              not self.position.is_long and not self.position.is_short):
            # Calcular SL/TP
            entry_price = self.data.Close[-1]
            atr_value = self.atr[-1]
            sl_price = entry_price + (self.sl_atr_mult * atr_value)
            tp_price = entry_price - (self.tp_rr_mult * (sl_price - entry_price))

            # Entrar posición short
            self.sell(sl=sl_price, tp=tp_price)


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


def walk_forward_test(df, n_periods=6, train_months=4, test_months=1):
    """
    Walk-forward optimization and testing

    Args:
        df: DataFrame con datos OHLCV
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

        # Optimización bayesiana en train
        best_params = optimize_strategy(train_data)

        # Test OOS con mejores parámetros
        bt = Backtest(test_data, MACDADxStrategy, cash=10000, commission=0.0005)
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


@use_named_args([
    Real(20, 30, name='adx_thresh'),
    Real(1.1, 1.4, name='vol_multiplier'),
    Real(1.3, 1.7, name='sl_atr_mult'),
    Real(2.0, 2.5, name='tp_rr_mult')
])
def objective(**params):
    """Función objetivo para optimización bayesiana"""
    try:
        bt = Backtest(_train_data, MACDADxStrategy, cash=10000, commission=0.0005)
        bt.run(**params)

        # Maximizar Sharpe ratio
        metrics = calculate_metrics(bt)
        return -metrics['sharpe_ratio']  # Negativo porque gp_minimize minimiza
    except Exception:
        return 1000  # Penalización por error


def optimize_strategy(train_data):
    """Optimización bayesiana de parámetros"""
    global _train_data
    _train_data = train_data

    if OPTIMIZATION_AVAILABLE:
        # Usar optimización bayesiana si está disponible
        space = [
            Real(20, 30, name='adx_thresh'),
            Real(1.1, 1.4, name='vol_multiplier'),
            Real(1.3, 1.7, name='sl_atr_mult'),
            Real(2.0, 2.5, name='tp_rr_mult')
        ]

        # Optimización
        res = gp_minimize(objective, space, n_calls=40, random_state=42)

        # Extraer mejores parámetros
        best_params = {
            'adx_thresh': res.x[0],
            'vol_multiplier': res.x[1],
            'sl_atr_mult': res.x[2],
            'tp_rr_mult': res.x[3]
        }
    else:
        # Grid search alternativo
        print("Usando grid search (scikit-optimize no disponible)")
        best_sharpe = -1000
        best_params = {
            'adx_thresh': 25.0,
            'vol_multiplier': 1.2,
            'sl_atr_mult': 1.5,
            'tp_rr_mult': 2.2
        }

        # Grid simple
        adx_options = [22, 25, 28]
        vol_options = [1.1, 1.2, 1.3]
        sl_options = [1.3, 1.5, 1.7]
        tp_options = [2.0, 2.2, 2.4]

        for adx in adx_options:
            for vol in vol_options:
                for sl in sl_options:
                    for tp in tp_options:
                        try:
                            params = {
                                'adx_thresh': adx,
                                'vol_multiplier': vol,
                                'sl_atr_mult': sl,
                                'tp_rr_mult': tp
                            }
                            bt = Backtest(
                                train_data, MACDADxStrategy, cash=10000, commission=0.0005)
                            bt.run(**params)
                            metrics = calculate_metrics(bt)

                            if metrics['sharpe_ratio'] > best_sharpe:
                                best_sharpe = metrics['sharpe_ratio']
                                best_params = params
                        except Exception:
                            continue

        print(f"Mejores parámetros encontrados: {best_params}, Sharpe: {best_sharpe:.2f}")

    return best_params


def ab_test_vs_rsi_bb(df, rsi_bb_signals):
    """A/B test vs RSI + Bollinger Bands strategy"""
    # Run MACD+ADX strategy
    bt_macd = Backtest(df, MACDADxStrategy, cash=10000, commission=0.0005)
    result_macd = bt_macd.run()

    # Simular RSI+BB strategy (usando señales externas)
    # Esto es un placeholder - necesitarías implementar RSI+BB strategy
    rsi_bb_returns = rsi_bb_signals * 0.025  # Placeholder

    # Comparación estadística
    macd_returns = result_macd['Return [%]'].pct_change().dropna()
    t_stat = (np.mean(macd_returns) - np.mean(rsi_bb_returns)) / np.sqrt(np.var(macd_returns) / \
              len(macd_returns) + np.var(rsi_bb_returns) / len(rsi_bb_returns))
    p_value = 2 * (1 - abs(t_stat) / np.sqrt(2))  # Aproximación simple

    # Superiority percentage
    superiority = (macd_returns > rsi_bb_returns).mean()

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'macd_superiority': superiority,
        'macd_metrics': calculate_metrics(bt_macd),
        'rsi_bb_returns': rsi_bb_returns
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
                bt = Backtest(df, MACDADxStrategy, cash=10000, commission=0.0005)
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


def generate_report(walk_forward_results, ab_results, robustness, sensitivity, anti_overfit):
    """Generar reporte completo"""
    # Crear directorio de resultados
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Métricas resumen
    oos_sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    oos_win_rates = [r['metrics']['win_rate'] for r in walk_forward_results]

    summary = {
        'strategy': 'Momentum MACD + ADX',
        'total_periods': len(walk_forward_results),
        'avg_oos_sharpe': np.mean(oos_sharpes),
        'std_oos_sharpe': np.std(oos_sharpes),
        'avg_oos_win_rate': np.mean(oos_win_rates),
        'ab_test_significant': ab_results['significant'],
        'ab_superiority': ab_results['macd_superiority'],
        'robustness_overall': robustness['overall_robust'],
        'sensitivity_stable': all(s['stable'] for s in sensitivity.values()),
        'anti_overfit_risk': anti_overfit['overfit_risk'],
        'recommendation': 'Deploy' if (np.mean(oos_sharpes) > 1.3 and
                                       ab_results['significant'] and
                                       robustness['overall_robust'] and
                                       anti_overfit['overfit_risk'] == 'low') else 'Further Testing'
    }

    # Guardar resultados
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
    """Generar gráficos de análisis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Sharpe ratio por periodo
    periods = [r['period'] for r in walk_forward_results]
    sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    axes[0, 0].plot(periods, sharpes, 'o-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=1.5, color='r', linestyle='--', alpha=0.7, label='Target 1.5')
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

    # ADX threshold sensitivity
    adx_values = [r['params']['adx_thresh'] for r in walk_forward_results]
    axes[1, 0].scatter(adx_values, sharpes, s=50, alpha=0.7)
    axes[1, 0].set_title('ADX Threshold vs Sharpe Ratio')
    axes[1, 0].set_xlabel('ADX Threshold')
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


def run_complete_analysis(df_btc, rsi_bb_signals=None):
    """
    Ejecutar análisis completo de la estrategia

    Args:
        df_btc: DataFrame con datos BTC OHLCV
        rsi_bb_signals: Señales RSI+BB para comparación A/B (opcional)
    """
    print("Ejecutando analisis completo Momentum MACD + ADX")
    print(f"Datos: {len(df_btc)} velas de {df_btc.index[0]} a {df_btc.index[-1]}")

    # Walk-forward testing
    print("\nEjecutando Walk-Forward Optimization...")
    wf_results = walk_forward_test(df_btc)

    # A/B test vs RSI+BB (si se proporciona)
    ab_results = None
    if rsi_bb_signals is not None:
        print("\nEjecutando A/B Test vs RSI+BB...")
        ab_results = ab_test_vs_rsi_bb(df_btc, rsi_bb_signals)

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
    summary = generate_report(wf_results, ab_results, robustness, sensitivity, anti_overfit)

    print("\nAnalisis completado!")
    print(f"Sharpe OOS promedio: {summary['avg_oos_sharpe']:.2f}")
    print(f"Win Rate promedio: {summary['avg_oos_win_rate']:.1%}")
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
    print("Momentum MACD + ADX Strategy")
    print("Para usar: run_complete_analysis(df_btc, rsi_bb_signals)")
    print("Resultados se guardan en results/momentum_macd_adx/")
