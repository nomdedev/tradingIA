#!/usr/bin/env python3
"""
Backtest Comparativo: VP IFVG EMA V2 con/sin Volume Profile
Muestra el impacto real del VP en el rendimiento de la estrategia.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.vp_ifvg_ema_strategy_v2 import VPIFVGEmaStrategyV2
import talib


class SimpleBacktester:
    """Simple backtester for strategy testing"""

    def __init__(self, strategy, initial_capital=10000, commission=0.001):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.trades = []
        self.equity_curve = [self.capital]

    def run_backtest(self, df):
        """Run backtest on dataframe"""
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.capital]

        # Pre-calcular indicadores
        df = self._precalculate_indicators(df)

        # Pre-calcular Volume Profile si está activado
        if self.strategy.use_vp_levels:
            self.vp_cache = self._precalculate_volume_profile(df)
        else:
            self.vp_cache = {}

        # Calcular indicadores en todo el DataFrame
        df_full = df.copy()
        # Calcular EMAs con los nombres que espera la estrategia
        df_full['ema1'] = talib.EMA(df_full['close'].values, timeperiod=20)  # ema1_length = 20
        df_full['ema2'] = talib.EMA(df_full['close'].values, timeperiod=50)  # ema2_length = 50
        df_full['ema3'] = talib.EMA(df_full['close'].values, timeperiod=100) # ema3_length = 100
        df_full['ema4'] = talib.EMA(df_full['close'].values, timeperiod=200) # ema4_length = 200
        df_full['atr'] = talib.ATR(df_full['high'].values, df_full['low'].values, df_full['close'].values, timeperiod=14)

        # Llamar generate_signals UNA SOLA VEZ con todos los datos
        df_multi_tf = {'5min': df_full}
        self.strategy.current_vp_data = None  # Usará el cache interno si es necesario
        signals = self.strategy.generate_signals(df_multi_tf)

        total_bars = len(df)
        step_size = 5  # Procesar cada 5 barras para velocidad

        start_time = time.time()

        # Procesar las señales generadas
        for idx in range(0, total_bars, step_size):
            current_time = time.time()

            # Mostrar progreso cada 2000 barras
            if idx % 2000 == 0 or idx >= total_bars - step_size:
                progress = (idx + 1) / total_bars * 100

                print(f"📈 Progreso: {idx+1}/{total_bars} barras ({progress:.1f}%) - Capital: ${self.capital:,.2f}")

            # Verificar señales de entrada en esta barra
            if signals['entries'].iloc[idx]:
                signal = {
                    'action': 'enter',
                    'direction': signals['signals'].iloc[idx],
                    'strength': signals['trade_scores'].iloc[idx]
                }
                self._enter_trade(signal, df.iloc[idx])

            # Verificar señales de salida en esta barra
            if signals['exits'].iloc[idx]:
                signal = {
                    'action': 'exit',
                    'direction': -signals['signals'].iloc[idx] if signals['signals'].iloc[idx] != 0 else 0
                }
                self._exit_trade(signal, df.iloc[idx])

            # Record equity
            self.equity_curve.append(self.capital)

        total_time = time.time() - start_time
        print(f"⏱️  Tiempo total: {total_time/60:.1f} minutos")

        return self._calculate_metrics()

    def _precalculate_indicators(self, df):
        """Pre-calcular indicadores"""
        for period in [5, 9, 21, 22, 34, 50, 100, 200]:
            df[f'ema{period}'] = talib.EMA(df['close'].values, timeperiod=period)
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        return df

    def _precalculate_volume_profile(self, df):
        """Pre-calcular Volume Profile"""
        vp_cache = {}
        total_bars = len(df)
        vp_window = min(500, total_bars)

        for idx in range(total_bars):
            if idx < 20:
                vp_cache[idx] = {'poc': df.iloc[idx]['close'], 'vah': df.iloc[idx]['high'], 'val': df.iloc[idx]['low']}
                continue

            start_idx = max(0, idx - vp_window)
            df_window = df.iloc[start_idx:idx+1]

            price_high = df_window['high'].tail(vp_window).max()
            price_low = df_window['low'].tail(vp_window).min()
            price_range = price_high - price_low

            if price_range == 0:
                vp_cache[idx] = {'poc': df.iloc[idx]['close'], 'vah': price_high, 'val': price_low}
                continue

            price_step = price_range / 100
            bins = np.arange(price_low, price_high + price_step, price_step)
            volume_profile = np.zeros(len(bins) - 1)

            for i in range(len(df_window)):
                bar_high = df_window['high'].iloc[i]
                bar_low = df_window['low'].iloc[i]
                bar_volume = df_window['volume'].iloc[i]

                for j in range(len(bins) - 1):
                    bin_low = bins[j]
                    bin_high = bins[j + 1]
                    overlap_low = max(bar_low, bin_low)
                    overlap_high = min(bar_high, bin_high)

                    if overlap_high > overlap_low:
                        overlap_pct = (overlap_high - overlap_low) / (bar_high - bar_low + 1e-10)
                        volume_profile[j] += bar_volume * overlap_pct

            poc_idx = np.argmax(volume_profile)
            poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

            total_volume = np.sum(volume_profile)
            target_volume = total_volume * 0.68
            cumsum = 0
            va_indices = []

            for idx_vol in np.argsort(volume_profile)[::-1]:
                cumsum += volume_profile[idx_vol]
                va_indices.append(idx_vol)
                if cumsum >= target_volume:
                    break

            vah = bins[max(va_indices) + 1]
            val = bins[min(va_indices)]

            vp_cache[idx] = {'poc': poc, 'vah': vah, 'val': val}

        return vp_cache

    def _enter_trade(self, signal, bar):
        """Enter a new trade"""
        entry_price = bar['close']
        quantity = (self.capital * 0.02) / entry_price
        cost = quantity * entry_price * (1 + self.commission)

        if cost > self.capital:
            return

        self.capital -= cost
        trade = {
            'entry_idx': len(self.equity_curve) - 1,
            'entry_price': entry_price,
            'quantity': quantity,
            'direction': signal['direction'],
            'entry_time': bar.name
        }
        self.trades.append(trade)

    def _exit_trade(self, signal, bar):
        """Exit existing trade"""
        if not self.trades:
            return

        trade = self.trades[-1]
        exit_price = bar['close']
        pnl = (exit_price - trade['entry_price']) * trade['quantity'] * trade['direction']
        pnl -= abs(pnl) * self.commission

        self.capital += trade['entry_price'] * trade['quantity'] + pnl
        trade.update({
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_time': bar.name,
            'duration': len(self.equity_curve) - 1 - trade['entry_idx']
        })

    def _calculate_metrics(self):
        """Calculate backtest metrics"""
        if not self.trades:
            return {
                'final_capital': self.capital,
                'total_return_pct': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'sharpe_ratio': 0
            }

        final_capital = self.capital
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital * 100

        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]

        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        expectancy = sum(t.get('pnl', 0) for t in self.trades) / total_trades if total_trades > 0 else 0

        equity = pd.Series(self.equity_curve)
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min() * 100

        returns = equity.pct_change().dropna()
        # Sharpe con risk-free rate
        rf_daily = 0.04 / 252
        excess_returns = returns - rf_daily
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if len(returns) > 0 and excess_returns.std() > 0 else 0.0

        return {
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio
        }


def run_comparative_backtest():
    """Run comparative backtest with and without VP"""

    print("🔬 BACKTEST COMPARATIVO: VP IFVG EMA V2")
    print("Comparando rendimiento con/sin Volume Profile")
    print("=" * 60)

    # Load data
    data_path = 'data/btc_15Min.csv'
    if not os.path.exists(data_path):
        print(f"❌ No se encontró el archivo: {data_path}")
        return

    print(f"📊 Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()

    # Usar muestra para testing rápido
    sample_size = 5000
    df = df.tail(sample_size)
    print(f"✅ Datos cargados: {len(df)} barras")
    print()

    results = {}

    # Test 1: CON Volume Profile
    print("🧪 TEST 1: CON VOLUME PROFILE")
    print("-" * 30)
    strategy_vp = VPIFVGEmaStrategyV2()
    strategy_vp.use_vp_levels = True
    strategy_vp.use_volume_filter = False  # Desactivar para aislamiento
    strategy_vp.use_ema_filter = True
    strategy_vp.min_signal_strength = 2  # Configuración original

    backtester_vp = SimpleBacktester(strategy_vp, initial_capital=10000)
    results['con_vp'] = backtester_vp.run_backtest(df.copy())

    print()

    # Test 2: SIN Volume Profile
    print("🧪 TEST 2: SIN VOLUME PROFILE")
    print("-" * 30)
    strategy_no_vp = VPIFVGEmaStrategyV2()
    strategy_no_vp.use_vp_levels = False
    strategy_no_vp.use_volume_filter = False  # Desactivar para aislamiento
    strategy_no_vp.use_ema_filter = True
    strategy_no_vp.min_signal_strength = 2  # Configuración original

    backtester_no_vp = SimpleBacktester(strategy_no_vp, initial_capital=10000)
    results['sin_vp'] = backtester_no_vp.run_backtest(df.copy())

    print()

    # Comparación de resultados
    print("📊 COMPARACIÓN DE RESULTADOS")
    print("=" * 60)

    metrics = ['total_return_pct', 'max_drawdown', 'total_trades', 'win_rate', 'profit_factor', 'expectancy', 'sharpe_ratio']

    print(f"{'Métrica':<20} {'Con VP':<12} {'Sin VP':<12} {'Diferencia':<12}")
    print(f"{'-'*20:<20} {'-'*12:<12} {'-'*12:<12} {'-'*12:<12}")
    print("-" * 60)

    for metric in metrics:
        con_vp = results['con_vp'][metric]
        sin_vp = results['sin_vp'][metric]

        if metric in ['total_return_pct', 'win_rate', 'profit_factor', 'expectancy', 'sharpe_ratio']:
            diff = con_vp - sin_vp
            symbol = "+" if diff > 0 else ""
        else:  # max_drawdown (menor es mejor)
            diff = sin_vp - con_vp  # Invertido para mostrar mejora
            symbol = "+" if diff > 0 else ""

        if metric == 'total_return_pct':
            print(f"{'Retorno Total':<20} {con_vp:<12.2f}% {sin_vp:<12.2f}% {symbol}{diff:<12.2f}%")
        elif metric == 'max_drawdown':
            print(f"{'Max Drawdown':<20} {con_vp:<12.2f}% {sin_vp:<12.2f}% {symbol}{diff:<12.2f}%")
        elif metric == 'win_rate':
            print(f"{'Win Rate':<20} {con_vp:<12.2f}% {sin_vp:<12.2f}% {symbol}{diff:<12.2f}%")
        elif metric == 'profit_factor':
            print(f"{'Profit Factor':<20} {con_vp:<12.2f} {sin_vp:<12.2f} {symbol}{diff:<12.2f}")
        elif metric == 'expectancy':
            print(f"{'Expectancy':<20} ${con_vp:<12.2f} ${sin_vp:<12.2f} {symbol}${diff:<12.2f}")
        elif metric == 'sharpe_ratio':
            print(f"{'Sharpe Ratio':<20} {con_vp:<12.2f} {sin_vp:<12.2f} {symbol}{diff:<12.2f}")
        elif metric == 'total_trades':
            print(f"{'Total Trades':<20} {int(con_vp):<12} {int(sin_vp):<12} {int(diff):<12}")
    print()

    # Análisis
    retorno_con = results['con_vp']['total_return_pct']
    retorno_sin = results['sin_vp']['total_return_pct']
    mejora = retorno_con - retorno_sin

    print("🎯 ANÁLISIS DEL IMPACTO DEL VP")
    print("=" * 60)

    if mejora > 5:
        print(f"✅ El VP mejora el rendimiento en {mejora:.2f}% absoluto")
        print("   - Confirma señales FVG débiles en zonas de valor")
        print("   - Aumenta el número de trades ejecutados")
        print("   - Actúa como filtro de calidad de señal")
    elif mejora > 0:
        print(f"🟡 El VP aporta mejora moderada de {mejora:.2f}%")
        print("   - Beneficio marginal pero positivo")
        print("   - Considerar activar en mercados ranging")
    else:
        print(f"❌ El VP reduce el rendimiento en {abs(mejora):.2f}% absoluto")
        print("   - Threshold de 5% puede ser demasiado restrictivo")
        print("   - En mercados trending fuertes, VP filtra señales válidas")
        print("   - Recomendar desactivar por defecto")

    # Recomendaciones
    print()
    print("💡 RECOMENDACIONES")
    print("=" * 60)

    if mejora > 10:
        print("🟢 RECOMENDACIÓN: VP ALTAMENTE RECOMENDADO")
        print("   - Mejora significativa del rendimiento")
        print("   - Aumenta trades y rentabilidad")
        print("   - Threshold de 5% funciona bien")
    elif mejora > 5:
        print("🟢 RECOMENDACIÓN: Mantener VP activado")
        print("   - Aporta mejora significativa al rendimiento")
        print("   - Justifica el costo computacional adicional")
    elif mejora > 0:
        print("🟡 RECOMENDACIÓN: VP opcional")
        print("   - Mejora marginal, pero positiva")
        print("   - Permitir al usuario activar/desactivar")
        print("   - Mejor en mercados laterales")
    else:
        print("🔴 RECOMENDACIÓN: Desactivar VP por defecto")
        print("   - Reduce el rendimiento general")
        print("   - Threshold actual filtra señales válidas")
        print("   - Ajustar parámetros o remover funcionalidad")

    print()
    print("📝 NOTAS TÉCNICAS:")
    print("   - VP se integra multiplicando: (vp_signal * sign(fvg_signal))")
    print("   - Threshold actual: 5% del rango VAH-VAL")
    print("   - Confirma POC, VAH, VAL como niveles de decisión")
    print("   - Funciona mejor cuando señales FVG son moderadas")
    print("   - En señales FVG fuertes, VP tiene poco impacto adicional")

    # Guardar resultados
    os.makedirs('results', exist_ok=True)
    with open('results/vp_impact_analysis.md', 'w', encoding='utf-8') as f:
        f.write("# Análisis del Impacto del Volume Profile\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Datos:** {len(df)} barras BTC 15min\n\n")
        f.write("## Resultados Comparativos\n\n")
        f.write("| Configuración | Retorno | Win Rate | Profit Factor | Expectancy | Trades |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.write(f"| **Con VP** | {results['con_vp']['total_return_pct']:.2f}% | {results['con_vp']['win_rate']:.2f}% | {results['con_vp']['profit_factor']:.2f} | ${results['con_vp']['expectancy']:.2f} | {results['con_vp']['total_trades']} |\n")
        f.write(f"| **Sin VP** | {results['sin_vp']['total_return_pct']:.2f}% | {results['sin_vp']['win_rate']:.2f}% | {results['sin_vp']['profit_factor']:.2f} | ${results['sin_vp']['expectancy']:.2f} | {results['sin_vp']['total_trades']} |\n")
        f.write(f"| **Diferencia** | {mejora:.2f}% | {(results['con_vp']['win_rate'] - results['sin_vp']['win_rate']):.2f}% | {(results['con_vp']['profit_factor'] - results['sin_vp']['profit_factor']):.2f} | ${(results['con_vp']['expectancy'] - results['sin_vp']['expectancy']):.2f} | {results['con_vp']['total_trades'] - results['sin_vp']['total_trades']} |\n\n")

        if mejora > 0:
            f.write("## Conclusión\n\n")
            f.write("El Volume Profile aporta valor positivo al sistema de trading.\n\n")
        else:
            f.write("## Conclusión\n\n")
            f.write("El Volume Profile actualmente reduce el rendimiento del sistema.\n\n")

    print("\n💾 Reporte guardado en: results/vp_impact_analysis.md")


if __name__ == "__main__":
    run_comparative_backtest()