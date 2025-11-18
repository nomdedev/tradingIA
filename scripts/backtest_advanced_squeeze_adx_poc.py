#!/usr/bin/env python3
"""
Backtest Avanzado: Squeeze Momentum + ADX + POC Multi-Timeframe
Análisis de pendiente ADX, divergencias, y POC como niveles de decisión
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy
import talib


class AdvancedBacktester:
    """Advanced backtester with multi-timeframe analysis and POC levels"""

    def __init__(self, strategy, initial_capital=10000, commission=0.001):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.trades = []
        self.equity_curve = [self.capital]

    def _precalculate_volume_profile(self, df, lookback=500):
        """Pre-calcular Volume Profile para POC"""
        vp_cache = {}

        for idx in range(len(df)):
            if idx < 20:
                vp_cache[idx] = {'poc': df.iloc[idx]['close'], 'vah': df.iloc[idx]['high'], 'val': df.iloc[idx]['low']}
                continue

            start_idx = max(0, idx - lookback)
            df_window = df.iloc[start_idx:idx+1]

            price_high = df_window['high'].max()
            price_low = df_window['low'].min()
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

            vah = bins[max(va_indices) + 1] if va_indices else price_high
            val = bins[min(va_indices)] if va_indices else price_low

            vp_cache[idx] = {'poc': poc, 'vah': vah, 'val': val}

        return vp_cache

    def _calculate_adx_slope(self, df, period=14):
        """Calcular pendiente del ADX"""
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        adx_slope = np.gradient(adx)
        return adx_slope

    def _detect_adx_divergence(self, df, adx_period=14):
        """Detectar divergencias en ADX"""
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=adx_period)

        # Detectar divergencias simples (precio vs ADX)
        divergences = np.zeros(len(df))

        for i in range(20, len(df) - 1):  # Evitar índices fuera de rango
            # Buscar mínimos locales en precio y ADX
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1] and
                adx[i] > adx[i-1] and adx[i] > adx[i+1]):
                divergences[i] = 1  # Divergencia alcista

            # Buscar máximos locales en precio y ADX
            elif (df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1] and
                  adx[i] < adx[i-1] and adx[i] < adx[i+1]):
                divergences[i] = -1  # Divergencia bajista

        return divergences

    def run_backtest(self, df_5m, df_15m=None, df_1h=None):
        """Run backtest with multi-timeframe analysis"""
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.capital]

        # Preparar datos multi-timeframe
        df_multi_tf = {'5min': df_5m.copy()}

        if df_15m is not None:
            df_multi_tf['15min'] = df_15m.copy()
        if df_1h is not None:
            df_multi_tf['1h'] = df_1h.copy()

        # Pre-calcular indicadores adicionales
        for tf, df in df_multi_tf.items():
            # ADX slope y divergencias
            df['adx_slope'] = self._calculate_adx_slope(df)
            df['adx_divergence'] = self._detect_adx_divergence(df)

        # Pre-calcular Volume Profile para POC
        print("🔄 Pre-calculando Volume Profile para POC...")
        vp_cache = self._precalculate_volume_profile(df_5m)

        total_bars = len(df_5m)
        step_size = 5

        start_time = time.time()

        # Procesar señales (simplificado)
        for idx in range(len(df_5m)):
            try:
                # Usar VP pre-calculado
                vp_data = vp_cache.get(idx, {'poc': df_5m.iloc[idx]['close'], 'vah': df_5m.iloc[idx]['high'], 'val': df_5m.iloc[idx]['low']})

                # Agregar análisis POC a la estrategia
                self.strategy.current_poc = vp_data['poc']
                self.strategy.current_adx_slope = df_5m.iloc[idx]['adx_slope']
                self.strategy.current_adx_divergence = df_5m.iloc[idx]['adx_divergence']

                signals = self.strategy.generate_signals(df_multi_tf)

                # Debug: mostrar progreso
                if idx % 1000 == 0:
                    entries_so_far = sum(1 for i in range(idx+1) if i < len(signals['entries']) and signals['entries'].iloc[i] == 1)
                    print(f"   Progreso [{idx}/{len(df_5m)}]: {entries_so_far} entradas encontradas")

                # Verificar señales de entrada
                if idx < len(signals['entries']) and signals['entries'].iloc[idx] == 1:
                    current_price = df_5m.iloc[idx]['close']
                    poc_alignment = self._check_poc_alignment(current_price, vp_data, signals['signals'].iloc[idx])

                    signal = {
                        'action': 'enter',
                        'direction': signals['signals'].iloc[idx],
                        'strength': signals['trade_scores'].iloc[idx] if idx < len(signals['trade_scores']) else 1,
                        'poc_confirmed': poc_alignment
                    }
                    self._enter_trade(signal, df_5m.iloc[idx])

                # Verificar señales de salida
                if idx < len(signals['exits']) and signals['exits'].iloc[idx] == 1:
                    signal = {
                        'action': 'exit',
                        'direction': -signals['signals'].iloc[idx] if idx < len(signals['signals']) and signals['signals'].iloc[idx] != 0 else 0
                    }
                    self._exit_trade(signal, df_5m.iloc[idx])

            except Exception as e:
                continue

        total_time = time.time() - start_time
        print(".1f")

        return self._calculate_metrics()

    def _check_poc_alignment(self, price, vp_data, signal_direction):
        """Verificar si el precio está alineado con POC para la dirección de la señal (menos restrictivo)"""
        poc = vp_data['poc']

        if signal_direction == 1:  # Long signal
            return price >= poc  # Precio en o por encima del POC para longs (menos restrictivo)
        elif signal_direction == -1:  # Short signal
            return price <= poc  # Precio en o por debajo del POC para shorts (menos restrictivo)

        return False

    def _enter_trade(self, signal, bar):
        """Enter a new trade"""
        entry_price = bar['close']
        quantity = (self.capital * 0.02) / entry_price  # 2% risk per trade

        if quantity <= 0:
            return

        cost = quantity * entry_price * (1 + self.commission)

        if cost > self.capital:
            return

        self.capital -= cost
        trade = {
            'entry_idx': len(self.equity_curve) - 1,
            'entry_price': entry_price,
            'quantity': quantity,
            'direction': signal['direction'],
            'entry_time': bar.name,
            'poc_confirmed': signal.get('poc_confirmed', False)
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
                'sharpe_ratio': 0,
                'poc_trades': 0,
                'poc_win_rate': 0
            }

        final_capital = self.capital
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital * 100

        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        poc_trades = [t for t in self.trades if t.get('poc_confirmed', False)]

        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        poc_win_rate = len([t for t in poc_trades if t.get('pnl', 0) > 0]) / len(poc_trades) * 100 if poc_trades else 0

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
            'sharpe_ratio': sharpe_ratio,
            'poc_trades': len(poc_trades),
            'poc_win_rate': poc_win_rate
        }


def load_multi_timeframe_data():
    """Load data for multiple timeframes"""
    data_files = {
        '5m': 'data/btc_5Min.csv',
        '15m': 'data/btc_15Min.csv',
        '1h': 'data/btc_1H.csv'
    }

    dataframes = {}

    for tf, file_path in data_files.items():
        if os.path.exists(file_path):
            print(f"📊 Cargando datos {tf}: {file_path}")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df.columns = df.columns.str.lower()

            # Usar muestra para testing rápido
            sample_size = 5000 if tf == '5m' else 2000
            df = df.tail(sample_size)

            dataframes[tf] = df
            print(f"   ✅ {len(df)} barras cargadas")
        else:
            print(f"   ❌ Archivo no encontrado: {file_path}")

    return dataframes


def run_advanced_backtest():
    """Run advanced backtest with Squeeze + ADX + POC"""

    print("🚀 BACKTEST AVANZADO: Squeeze Momentum + ADX + POC Multi-Timeframe")
    print("=" * 80)

    # Load multi-timeframe data
    dataframes = load_multi_timeframe_data()

    if '5m' not in dataframes:
        print("❌ Datos de 5 minutos no encontrados")
        return

    df_5m = dataframes['5m']
    df_15m = dataframes.get('15m')
    df_1h = dataframes.get('1h')

    print(f"\n📊 Datos cargados:")
    print(f"   5m: {len(df_5m)} barras")
    if df_15m is not None:
        print(f"   15m: {len(df_15m)} barras")
    if df_1h is not None:
        print(f"   1h: {len(df_1h)} barras")

    results = {}

    # Test 1: Squeeze + ADX + POC completo
    print("\n🧪 TEST 1: Squeeze + ADX + POC + Multi-Timeframe")
    print("-" * 50)

    strategy_full = SqueezeMomentumADXTTMStrategy()
    strategy_full.use_multitimeframe = True
    strategy_full.use_poc_filter = True
    strategy_full.use_adx_slope = True
    strategy_full.use_adx_divergence = True

    backtester_full = AdvancedBacktester(strategy_full, initial_capital=10000)
    results['full'] = backtester_full.run_backtest(df_5m, df_15m, df_1h)

    # Test 2: Solo Squeeze + ADX (sin POC ni multi-timeframe avanzado)
    print("\n🧪 TEST 2: Solo Squeeze + ADX (baseline)")
    print("-" * 50)

    strategy_basic = SqueezeMomentumADXTTMStrategy()
    strategy_basic.use_multitimeframe = False
    strategy_basic.use_poc_filter = False
    strategy_basic.use_adx_slope = False
    strategy_basic.use_adx_divergence = False

    backtester_basic = AdvancedBacktester(strategy_basic, initial_capital=10000)
    results['basic'] = backtester_basic.run_backtest(df_5m)

    # Comparación de resultados
    print("\n📊 COMPARACIÓN DE RESULTADOS")
    print("=" * 80)

    metrics = ['total_return_pct', 'max_drawdown', 'total_trades', 'win_rate', 'profit_factor', 'expectancy', 'sharpe_ratio', 'poc_trades', 'poc_win_rate']

    print(f"{'Métrica':<20} {'Completo':<12} {'Básico':<12} {'Diferencia':<12}")
    print(f"{'-'*20:<20} {'-'*12:<12} {'-'*12:<12} {'-'*12:<12}")

    for metric in metrics:
        full_val = results['full'][metric]
        basic_val = results['basic'][metric]

        if metric in ['total_return_pct', 'win_rate', 'profit_factor', 'expectancy', 'sharpe_ratio', 'poc_win_rate']:
            diff = full_val - basic_val
            symbol = "+" if diff > 0 else ""
        else:
            diff = basic_val - full_val  # Para drawdown (menor es mejor)
            symbol = "+" if diff > 0 else ""

        if metric in ['total_return_pct', 'max_drawdown', 'win_rate', 'profit_factor', 'expectancy', 'sharpe_ratio', 'poc_win_rate']:
            print(f"{metric:<20} {full_val:<12.2f} {basic_val:<12.2f} {symbol}{diff:<12.2f}")
        else:
            print(f"{metric:<20} {full_val:<12} {basic_val:<12} {symbol}{diff:<12}")

    print()

    # Análisis específico
    retorno_full = results['full']['total_return_pct']
    retorno_basic = results['basic']['total_return_pct']
    mejora = retorno_full - retorno_basic

    print("🎯 ANÁLISIS DEL IMPACTO")
    print("=" * 80)

    if mejora > 10:
        print(f"🟢 MEJORA SIGNIFICATIVA: +{mejora:.2f}%")
        print("   - POC filtra entradas de baja calidad")
        print("   - Multi-timeframe añade confirmación de tendencia")
        print("   - ADX slope/divergence mejoran timing")
    elif mejora > 0:
        print(f"🟡 MEJORA MODERADA: +{mejora:.2f}%")
        print("   - Beneficio marginal pero positivo")
        print("   - POC útil en mercados ranging")
    else:
        print(f"🔴 SIN MEJORA: {mejora:.2f}%")
        print("   - Posible sobre-filtrado")
        print("   - Ajustar parámetros de POC/ADX")

    # Análisis POC específico
    poc_trades = results['full']['poc_trades']
    poc_win_rate = results['full']['poc_win_rate']
    total_trades = results['full']['total_trades']

    print("\n📍 ANÁLISIS POC ESPECÍFICO")
    print(".1f")
    print(".1f")
    if poc_win_rate > results['full']['win_rate']:
        print("   ✅ POC mejora la calidad de las entradas")
    else:
        print("   ⚠️ POC no mejora significativamente la calidad")

    # Recomendaciones
    print("\n💡 RECOMENDACIONES")
    print("=" * 80)

    if mejora > 5:
        print("🟢 SISTEMA RECOMENDADO")
        print("   - Implementar POC + multi-timeframe + ADX avanzado")
        print("   - Configurar alerts para entradas con POC confirmado")
    else:
        print("🟡 SISTEMA OPCIONAL")
        print("   - Usar como filtro adicional, no crítico")
        print("   - Permitir usuario activar/desactivar")

    print("\n🔧 CONFIGURACIÓN ÓPTIMA:")
    print("   - POC lookback: 500 barras")
    print("   - ADX slope: activado para pendiente positiva")
    print("   - Multi-timeframe: 5m + 15m + 1h")
    print("   - Threshold POC: precio > POC para longs, precio < POC para shorts")

    # Guardar resultados
    os.makedirs('results', exist_ok=True)
    with open('results/advanced_squeeze_adx_poc_analysis.md', 'w', encoding='utf-8') as f:
        f.write("# Análisis Avanzado: Squeeze + ADX + POC\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Configuraciones Comparadas\n\n")
        f.write("- **Full**: Squeeze + ADX + POC + Multi-timeframe + ADX slope/divergence\n")
        f.write("- **Basic**: Solo Squeeze + ADX básico\n\n")
        f.write("## Resultados\n\n")
        f.write("| Configuración | Retorno | Trades | Win Rate | POC Trades | POC Win Rate |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.write(f"| **Full** | {results['full']['total_return_pct']:.2f}% | {results['full']['total_trades']} | {results['full']['win_rate']:.2f}% | {results['full']['poc_trades']} | {results['full']['poc_win_rate']:.2f}% |\n")
        f.write(f"| **Basic** | {results['basic']['total_return_pct']:.2f}% | {results['basic']['total_trades']} | {results['basic']['win_rate']:.2f}% | - | - |\n")
        f.write(f"| **Diferencia** | {mejora:.2f}% | {results['full']['total_trades'] - results['basic']['total_trades']} | {(results['full']['win_rate'] - results['basic']['win_rate']):.2f}% | - | - |\n\n")

        if mejora > 0:
            f.write("## Conclusión\n\n")
            f.write("El sistema avanzado aporta valor adicional a la estrategia básica.\n\n")
        else:
            f.write("## Conclusión\n\n")
            f.write("El sistema avanzado no mejora significativamente los resultados.\n\n")

    print("\n💾 Reporte guardado en: results/advanced_squeeze_adx_poc_analysis.md")
if __name__ == "__main__":
    run_advanced_backtest()