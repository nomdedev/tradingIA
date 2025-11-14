"""
Mean Reversion IBS + Bollinger Bands Strategy

Estrategia de mean reversion usando Internal Bar Strength (IBS) + Bollinger Bands
con filtros de volumen y trend HTF. Sharpe OOS objetivo: 1.8, Win Rate: 69%.

Características:
- IBS < 0.3 (oversold)
- Close < BB lower (20/2SD)
- Vol > 1.5 * SMA21
- HTF EMA210_1h uptrend
- Entry long con confluencia 4/4
- SL = 1.5 * ATR14, TP = 2.2R
- Trailing stop +1R breakeven +0.5ATR
"""

import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
from backtesting.lib import resample_apply
import talib
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración
RESULTS_DIR = Path("results/mean_reversion_ibs_bb")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class IBSBBStrategy(Strategy):
    """
    Mean Reversion Strategy usando IBS + Bollinger Bands
    """

    # Parámetros optimizables
    bb_period = 20
    bb_std = 2.0
    ibs_thresh = 0.3
    vol_multiplier = 1.5
    atr_period = 14
    sl_atr_mult = 1.5
    tp_rr_mult = 2.2
    trail_trigger_rr = 1.0
    trail_offset_atr = 0.5

    def init(self):
        """Inicializar indicadores"""
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)

        # IBS (Internal Bar Strength)
        self.ibs = self.I(
            self._calculate_ibs,
            self.data.Open,
            self.data.High,
            self.data.Low,
            self.data.Close)

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
        self.ema210_htf = resample_apply('1H', talib.EMA, self.data.Close, timeperiod=210)

    def _calculate_ibs(self, open_, high, low, close):
        """Calcular Internal Bar Strength"""
        return (close - low) / (high - low)

    def next(self):
        """Lógica de trading"""
        # Confluencia completa (4/4)
        ibs_oversold = self.ibs[-1] < self.ibs_thresh
        bb_break = self.data.Close[-1] < self.bb_lower[-1]
        vol_confirm = self.data.Volume[-1] > (self.vol_multiplier * self.vol_sma[-1])
        htf_uptrend = self.ema210_htf[-1] > self.ema210_htf[-2] if len(
            self.ema210_htf) > 1 else True

        confluence = ibs_oversold and bb_break and vol_confirm and htf_uptrend

        # Entry Long
        if confluence and not self.position:
            # Calcular SL/TP
            entry_price = self.data.Close[-1]
            atr_value = self.atr[-1]
            sl_price = entry_price - (self.sl_atr_mult * atr_value)
            tp_price = entry_price + (self.tp_rr_mult * (entry_price - sl_price))

            # Entrar posición
            self.buy(sl=sl_price, tp=tp_price)

        # Trailing Stop Management
        elif self.position:
            # Activar trailing cuando llegue a +1R
            if self.position.pl > (self.trail_trigger_rr *
                                   abs(self.position.sl - self.position.entry_price)):
                # Calcular nuevo trailing stop
                trail_distance = self.trail_offset_atr * self.atr[-1]
                new_sl = max(self.position.sl, self.data.Close[-1] - trail_distance)
                self.position.sl = new_sl


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


def walk_forward_test(df, n_periods=8, train_months=3, test_months=1):
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
        bt = Backtest(test_data, IBSBBStrategy, cash=10000, commission=0.0005)
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
    Real(1.5, 2.5, name='bb_std'),
    Real(0.2, 0.4, name='ibs_thresh'),
    Real(1.2, 1.8, name='vol_multiplier'),
    Real(1.2, 1.8, name='sl_atr_mult'),
    Real(2.0, 3.0, name='tp_rr_mult')
])
def objective(**params):
    """Función objetivo para optimización bayesiana"""
    try:
        bt = Backtest(_train_data, IBSBBStrategy, cash=10000, commission=0.0005)
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

    # Espacio de búsqueda
    space = [
        Real(1.5, 2.5, name='bb_std'),
        Real(0.2, 0.4, name='ibs_thresh'),
        Real(1.2, 1.8, name='vol_multiplier'),
        Real(1.2, 1.8, name='sl_atr_mult'),
        Real(2.0, 3.0, name='tp_rr_mult')
    ]

    # Optimización
    res = gp_minimize(objective, space, n_calls=50, random_state=42)

    # Extraer mejores parámetros
    best_params = {
        'bb_std': res.x[0],
        'ibs_thresh': res.x[1],
        'vol_multiplier': res.x[2],
        'sl_atr_mult': res.x[3],
        'tp_rr_mult': res.x[4]
    }

    return best_params


def ab_test_vs_ifvg(df, ifvg_signals):
    """A/B test vs IFVG strategy"""
    # Run IBS+BB strategy
    bt_ibs = Backtest(df, IBSBBStrategy, cash=10000, commission=0.0005)
    result_ibs = bt_ibs.run()

    # Simular IFVG strategy (usando señales externas)
    # Esto es un placeholder - necesitarías implementar IFVG strategy
    ifvg_returns = ifvg_signals * 0.02  # Placeholder

    # Comparación estadística - usando numpy en lugar de scipy.stats
    ibs_returns = result_ibs['Return [%]'].pct_change().dropna()
    t_stat = (np.mean(ibs_returns) - np.mean(ifvg_returns)) / np.sqrt(np.var(ibs_returns) / \
              len(ibs_returns) + np.var(ifvg_returns) / len(ifvg_returns))
    p_value = 2 * (1 - abs(t_stat) / np.sqrt(2))  # Aproximación simple

    # Superiority percentage
    superiority = (ibs_returns > ifvg_returns).mean()

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'ibs_superiority': superiority,
        'ibs_metrics': calculate_metrics(bt_ibs),
        'ifvg_returns': ifvg_returns
    }


def anti_snooping_analysis(walk_forward_results):
    """Análisis anti-snooping con AIC y White's Reality Check"""

    # AIC calculation (simplified)
    returns = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    n = len(returns)
    k = 5  # parameters
    log_likelihood = -n / 2 * np.log(np.var(returns))
    aic = 2 * k - 2 * log_likelihood

    # Baseline (random strategy)
    baseline_aic = 2 * k - 2 * (-n / 2 * np.log(np.var(np.random.normal(0, 0.01, n))))

    # White's Reality Check (simplified)
    null_returns = []
    for _ in range(500):
        null_ret = np.random.normal(0, 0.01, n)
        null_returns.append(null_ret)

    null_means = [np.mean(nr) for nr in null_returns]
    adj_p = np.mean(np.array(null_means) >= np.mean(returns))

    return {
        'aic': aic,
        'baseline_aic': baseline_aic,
        'snooping_detected': aic > baseline_aic + 10,
        'whites_adj_p': adj_p,
        'bias_risk': 'high' if adj_p >= 0.05 else 'low'
    }


def generate_report(walk_forward_results, ab_results, snooping_analysis):
    """Generar reporte completo"""
    # Crear directorio de resultados
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Métricas resumen
    oos_sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]
    oos_win_rates = [r['metrics']['win_rate'] for r in walk_forward_results]

    summary = {
        'strategy': 'Mean Reversion IBS + BB',
        'total_periods': len(walk_forward_results),
        'avg_oos_sharpe': np.mean(oos_sharpes),
        'std_oos_sharpe': np.std(oos_sharpes),
        'avg_oos_win_rate': np.mean(oos_win_rates),
        'oos_degradation': np.std(oos_sharpes) < 0.2,  # Robustez
        'ab_test_significant': ab_results['significant'],
        'ab_superiority': ab_results['ibs_superiority'],
        'snooping_risk': snooping_analysis['bias_risk'],
        'recommendation': 'Deploy' if (np.mean(oos_sharpes) > 1.5 and
                                       ab_results['significant'] and
                                       snooping_analysis['bias_risk'] == 'low') else 'Further Testing'
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
    axes[0, 1].axhline(y=0.65, color='r', linestyle='--', alpha=0.7, label='Target 65%')
    axes[0, 1].set_title('OOS Win Rate por Periodo')
    axes[0, 1].set_xlabel('Periodo Walk-Forward')
    axes[0, 1].set_ylabel('Win Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Max Drawdown por periodo
    max_dds = [r['metrics']['max_drawdown'] for r in walk_forward_results]
    axes[1, 0].bar(periods, max_dds, color='red', alpha=0.7)
    axes[1, 0].axhline(y=-0.15, color='r', linestyle='--', alpha=0.7, label='Max DD -15%')
    axes[1, 0].set_title('OOS Max Drawdown por Periodo')
    axes[1, 0].set_xlabel('Periodo Walk-Forward')
    axes[1, 0].set_ylabel('Max Drawdown')
    axes[1, 0].legend()

    # Profit Factor por periodo
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

    # Heatmap de correlación de parámetros
    params_data = []
    for result in walk_forward_results:
        params = result['params']
        params['sharpe'] = result['metrics']['sharpe_ratio']
        params_data.append(params)

    if params_data:
        params_df = pd.DataFrame(params_data)
        corr_matrix = params_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlación de Parámetros y Performance')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'parameter_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


def run_complete_analysis(df_btc, ifvg_signals=None):
    """
    Ejecutar análisis completo de la estrategia

    Args:
        df_btc: DataFrame con datos BTC OHLCV
        ifvg_signals: Señales IFVG para comparación A/B (opcional)
    """
    print("🚀 Iniciando análisis completo Mean Reversion IBS + BB")
    print(f"📊 Datos: {len(df_btc)} velas de {df_btc.index[0]} a {df_btc.index[-1]}")

    # Walk-forward testing
    print("\n📈 Ejecutando Walk-Forward Optimization...")
    wf_results = walk_forward_test(df_btc)

    # A/B test vs IFVG (si se proporciona)
    ab_results = None
    if ifvg_signals is not None:
        print("\n⚔️ Ejecutando A/B Test vs IFVG...")
        ab_results = ab_test_vs_ifvg(df_btc, ifvg_signals)

    # Anti-snooping analysis
    print("\n🔍 Ejecutando Anti-Snooping Analysis...")
    snooping_results = anti_snooping_analysis(wf_results)

    # Generar reporte
    print("\n📋 Generando Reporte Final...")
    summary = generate_report(wf_results, ab_results, snooping_results)

    print("\nAnalisis completado!")
    print(f"Sharpe OOS promedio: {summary['avg_oos_sharpe']:.2f}")
    print(f"Win Rate promedio: {summary['avg_oos_win_rate']:.1%}")
    print(f"Recomendacion: {summary['recommendation']}")

    return {
        'walk_forward_results': wf_results,
        'ab_results': ab_results,
        'snooping_analysis': snooping_results,
        'summary': summary
    }


if __name__ == "__main__":
    # Ejemplo de uso
    print("Mean Reversion IBS + BB Strategy")
    print("Para usar: run_complete_analysis(df_btc, ifvg_signals)")
    print("Resultados se guardan en results/mean_reversion_ibs_bb/")
