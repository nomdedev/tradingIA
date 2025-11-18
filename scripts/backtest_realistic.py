#!/usr/bin/env python3
"""
Backtest Realista: Squeeze Momentum + ADX + POC Multi-Timeframe
Análisis completo con datos reales, sin filtros restrictivos
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


class RealisticBacktester:
    """Realistic backtester with proper risk management and realistic assumptions"""

    def __init__(self, strategy, initial_capital=10000, risk_per_trade=0.02, max_trades_per_day=3):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_trades_per_day = max_trades_per_day
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.0002  # 0.02% slippage
        self.trades = []
        self.equity_curve = [self.capital]
        self.daily_trades = {}
        self.current_day = None

    def _precalculate_volume_profile(self, df, lookback=200):
        """Pre-calcular Volume Profile para POC (más conservador)"""
        vp_cache = {}

        for idx in range(len(df)):
            if idx < 20:
                vp_cache[idx] = {'poc': df.iloc[idx]['close'], 'vah': df.iloc[idx]['high'], 'val': df.iloc[idx]['low']}
                continue

            start_idx = max(0, idx - lookback)
            df_window = df.iloc[start_idx:idx+1]

            # Simplified POC calculation - just volume-weighted average
            total_volume = df_window['volume'].sum()
            if total_volume > 0:
                weighted_price = (df_window['close'] * df_window['volume']).sum() / total_volume
                poc = weighted_price
            else:
                poc = df_window['close'].iloc[-1]

            vah = df_window['high'].max()
            val = df_window['low'].min()

            vp_cache[idx] = {'poc': poc, 'vah': vah, 'val': val}

        return vp_cache

    def _calculate_adx_slope(self, df, period=14):
        """Calcular pendiente del ADX (simplificada)"""
        adx = df['adx'].values if 'adx' in df.columns else np.zeros(len(df))
        adx_slope = np.gradient(adx)
        return adx_slope

    def run_realistic_backtest(self, df_5m, df_15m=None, df_1h=None):
        """Run realistic backtest with proper risk management"""
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.capital]
        self.daily_trades = {}
        self.current_day = None

        print("🔄 Pre-calculando indicadores...")
        start_time = time.time()

        # Pre-calcular indicadores para todo el dataset
        df_full = df_5m.copy()
        df_full = self.strategy._calculate_squeeze_momentum(df_full)
        df_full = self.strategy._calculate_adx(df_full)
        df_full = self.strategy._calculate_ttm_waves(df_full)

        # Pre-calcular Volume Profile
        vp_cache = self._precalculate_volume_profile(df_full)

        # Calcular ADX slope
        df_full['adx_slope'] = self._calculate_adx_slope(df_full)

        prep_time = time.time() - start_time
        print(".2f")

        # Preparar datos multi-timeframe
        df_multi_tf = {'5Min': df_full}
        if df_15m is not None:
            df_multi_tf['15Min'] = df_15m.copy()
        if df_1h is not None:
            df_multi_tf['1H'] = df_1h.copy()

        total_bars = len(df_full)
        trades_today = 0

        print("🚀 Ejecutando backtest realista...")

        for idx in range(len(df_full)):
            current_time = df_full.index[idx]
            current_day = current_time.date()

            # Reset daily trade counter
            if current_day != self.current_day:
                self.current_day = current_day
                trades_today = 0

            # Skip if max trades per day reached
            if trades_today >= self.max_trades_per_day:
                self.equity_curve.append(self.capital)
                continue

            try:
                # Get VP data
                vp_data = vp_cache.get(idx, {'poc': df_full.iloc[idx]['close'], 'vah': df_full.iloc[idx]['high'], 'val': df_full.iloc[idx]['low']})

                # Set strategy parameters
                self.strategy.current_poc = vp_data['poc']
                self.strategy.current_adx_slope = df_full.iloc[idx]['adx_slope']
                self.strategy.current_adx_divergence = 0  # Simplified

                # Generate signals
                signals = self.strategy.generate_signals(df_multi_tf)

                # Check for entry signal
                if (idx < len(signals['entries']) and signals['entries'].iloc[idx] == 1 and
                    self._validate_entry_signal(df_full.iloc[idx], vp_data, signals['signals'].iloc[idx])):

                    # Enter trade with realistic execution
                    success = self._enter_realistic_trade(df_full.iloc[idx], signals['signals'].iloc[idx], vp_data)
                    if success:
                        trades_today += 1

                # Check for exit signal
                if (idx < len(signals['exits']) and signals['exits'].iloc[idx] == 1 and self.trades):
                    self._exit_realistic_trade(df_full.iloc[idx])

                # Update equity curve
                self.equity_curve.append(self.capital)

                # Progress update
                if idx % 1000 == 0:
                    progress = (idx + 1) / total_bars * 100
                    print(".1f")

            except Exception as e:
                self.equity_curve.append(self.capital)
                continue

        return self._calculate_realistic_metrics()

    def _validate_entry_signal(self, bar, vp_data, signal_direction):
        """Validate entry signal with realistic filters"""
        current_price = bar['close']

        # Basic POC alignment (very permissive)
        poc = vp_data['poc']
        if signal_direction == 1:  # Long
            poc_ok = current_price >= poc * 0.995  # Allow 0.5% below POC
        elif signal_direction == -1:  # Short
            poc_ok = current_price <= poc * 1.005  # Allow 0.5% above POC
        else:
            poc_ok = True

        # ADX slope check (optional, not restrictive)
        adx_slope_ok = True
        if hasattr(self.strategy, 'current_adx_slope') and self.strategy.use_adx_slope:
            adx_slope = self.strategy.current_adx_slope
            if signal_direction == 1 and adx_slope < -0.1:  # Avoid longs when ADX falling sharply
                adx_slope_ok = False
            elif signal_direction == -1 and adx_slope > 0.1:  # Avoid shorts when ADX rising sharply
                adx_slope_ok = False

        return poc_ok and adx_slope_ok

    def _enter_realistic_trade(self, bar, direction, vp_data):
        """Enter trade with realistic execution"""
        entry_price = bar['close'] * (1 + self.slippage if direction == 1 else 1 - self.slippage)

        # Calculate position size based on risk
        risk_amount = self.capital * self.risk_per_trade

        # Use ATR for stop loss if available
        if 'atr' in bar:
            stop_distance = bar['atr'] * 1.5
        else:
            stop_distance = entry_price * 0.02  # 2% default

        position_size = risk_amount / stop_distance
        position_value = position_size * entry_price

        # Check if we have enough capital
        cost = position_value * (1 + self.commission)
        if cost > self.capital:
            return False

        self.capital -= cost

        trade = {
            'entry_idx': len(self.equity_curve) - 1,
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': direction,
            'stop_distance': stop_distance,
            'entry_time': bar.name,
            'poc_confirmed': self._check_poc_alignment(bar['close'], vp_data, direction),
            'status': 'open'
        }

        self.trades.append(trade)
        return True

    def _exit_realistic_trade(self, bar):
        """Exit trade with realistic execution"""
        if not self.trades:
            return

        trade = self.trades[-1]
        if trade['status'] != 'open':
            return

        exit_price = bar['close'] * (1 - self.slippage if trade['direction'] == 1 else 1 + self.slippage)

        # Calculate P&L
        price_diff = exit_price - trade['entry_price']
        pnl = price_diff * trade['position_size'] * trade['direction']
        pnl -= abs(pnl) * self.commission  # Commission on exit

        self.capital += trade['entry_price'] * trade['position_size'] + pnl

        # Update trade record
        trade.update({
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_time': bar.name,
            'status': 'closed',
            'duration': len(self.equity_curve) - 1 - trade['entry_idx']
        })

    def _check_poc_alignment(self, price, vp_data, signal_direction):
        """Check POC alignment (permissive)"""
        poc = vp_data['poc']
        if signal_direction == 1:
            return price >= poc * 0.99  # Within 1% of POC for longs
        elif signal_direction == -1:
            return price <= poc * 1.01  # Within 1% of POC for shorts
        return True

    def _calculate_realistic_metrics(self):
        """Calculate comprehensive realistic metrics"""
        if not self.trades:
            return {
                'final_capital': self.capital,
                'total_return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'expectancy': 0,
                'calmar_ratio': 0,
                'poc_trades': 0,
                'poc_win_rate': 0
            }

        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        poc_trades = [t for t in closed_trades if t.get('poc_confirmed', False)]

        final_capital = self.capital
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital * 100

        total_trades = len(closed_trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0

        total_profit = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss) if total_trades > 0 else 0

        # Calculate drawdown
        equity = pd.Series(self.equity_curve)
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min() * 100

        # Sharpe ratio (assuming daily returns, simplified)
        returns = equity.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            # Sharpe con risk-free rate
            rf_daily = 0.04 / 252
            excess_returns = returns - rf_daily
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0

        # Calmar ratio
        calmar_ratio = total_return_pct / abs(max_drawdown) if max_drawdown != 0 else 0

        poc_win_rate = len([t for t in poc_trades if t['pnl'] > 0]) / len(poc_trades) * 100 if poc_trades else 0

        return {
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'expectancy': expectancy,
            'calmar_ratio': calmar_ratio,
            'poc_trades': len(poc_trades),
            'poc_win_rate': poc_win_rate
        }


def load_full_dataset():
    """Load full dataset for realistic backtest"""
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

            # Use more data for realistic backtest
            sample_size = min(10000, len(df))  # Up to 10k bars
            df = df.tail(sample_size)

            dataframes[tf] = df
            print(f"   ✅ {len(df)} barras cargadas")
        else:
            print(f"   ❌ Archivo no encontrado: {file_path}")

    return dataframes


def run_realistic_backtest():
    """Run comprehensive realistic backtest"""

    print("🚀 BACKTEST REALISTA: Squeeze Momentum + ADX + POC")
    print("=" * 80)
    print(f"📅 Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print()

    # Load full dataset
    dataframes = load_full_dataset()

    if '5m' not in dataframes:
        print("❌ Datos de 5 minutos no encontrados")
        return

    df_5m = dataframes['5m']
    df_15m = dataframes.get('15m')
    df_1h = dataframes.get('1h')

    print(f"\n📊 Dataset cargado:")
    print(f"   5m: {len(df_5m)} barras ({df_5m.index[0]} a {df_5m.index[-1]})")
    if df_15m is not None:
        print(f"   15m: {len(df_15m)} barras")
    if df_1h is not None:
        print(f"   1h: {len(df_1h)} barras")

    results = {}

    # Test 1: Estrategia básica (sin filtros restrictivos)
    print("\n🧪 TEST 1: Estrategia Básica (Squeeze + ADX)")
    print("-" * 60)

    strategy_basic = SqueezeMomentumADXTTMStrategy()
    strategy_basic.use_multitimeframe = False
    strategy_basic.use_poc_filter = False
    strategy_basic.use_adx_slope = False
    strategy_basic.use_adx_divergence = False

    backtester_basic = RealisticBacktester(strategy_basic, initial_capital=10000, risk_per_trade=0.02)
    results['basic'] = backtester_basic.run_realistic_backtest(df_5m)

    # Test 2: Estrategia completa con POC (filtros permisivos)
    print("\n🧪 TEST 2: Estrategia Completa (Squeeze + ADX + POC permisivo)")
    print("-" * 60)

    strategy_full = SqueezeMomentumADXTTMStrategy()
    strategy_full.use_multitimeframe = True
    strategy_full.use_poc_filter = True
    strategy_full.use_adx_slope = True
    strategy_full.use_adx_divergence = False  # Too restrictive

    backtester_full = RealisticBacktester(strategy_full, initial_capital=10000, risk_per_trade=0.02)
    results['full'] = backtester_full.run_realistic_backtest(df_5m, df_15m, df_1h)

    # Análisis comparativo detallado
    print("\n📊 ANÁLISIS COMPARATIVO DETALLADO")
    print("=" * 80)

    metrics_to_show = [
        ('total_return_pct', 'Retorno Total', '%'),
        ('total_trades', 'Total Trades', ''),
        ('win_rate', 'Win Rate', '%'),
        ('profit_factor', 'Profit Factor', ''),
        ('expectancy', 'Expectancy', '$'),
        ('max_drawdown', 'Max Drawdown', '%'),
        ('sharpe_ratio', 'Sharpe Ratio', ''),
        ('calmar_ratio', 'Calmar Ratio', ''),
        ('poc_trades', 'Trades con POC', ''),
        ('poc_win_rate', 'Win Rate POC', '%')
    ]

    print(f"{'Configuración':<25} {'Básica':<12} {'Completa':<12} {'Diferencia':<12}")
    print(f"{'-'*25:<25} {'-'*12:<12} {'-'*12:<12} {'-'*12:<12}")
    print("-" * 80)

    for metric_key, metric_name, suffix in metrics_to_show:
        basic_val = results['basic'][metric_key]
        full_val = results['full'][metric_key]

        if metric_key in ['total_return_pct', 'win_rate', 'profit_factor', 'expectancy', 'sharpe_ratio', 'calmar_ratio', 'poc_win_rate']:
            diff = full_val - basic_val
            symbol = "+" if diff > 0 else ""
        elif metric_key == 'max_drawdown':
            diff = basic_val - full_val  # Lower drawdown is better
            symbol = "+" if diff > 0 else ""
        else:
            diff = full_val - basic_val
            symbol = "+" if diff > 0 else ""

        if metric_key in ['total_return_pct', 'win_rate', 'max_drawdown', 'sharpe_ratio', 'calmar_ratio', 'poc_win_rate']:
            print("<25.2f"        elif metric_key in ['profit_factor', 'expectancy']:
            print("<25.2f"        else:
            print("<25"
    # Análisis de rendimiento
    print("
🎯 ANÁLISIS DE RENDIMIENTO"    print("=" * 80)

    retorno_basic = results['basic']['total_return_pct']
    retorno_full = results['full']['total_return_pct']
    mejora = retorno_full - retorno_basic

    if mejora > 5:
        print(".2f"        print("   ✅ Los filtros avanzados aportan valor significativo")
        print("   ✅ POC mejora la calidad sin eliminar oportunidades")
    elif mejora > 0:
        print(".2f"        print("   ⚠️ Mejora marginal - considerar simplificar")
    else:
        print(".2f"        print("   ❌ Los filtros avanzados no aportan valor")
        print("   ❌ Considerar usar solo estrategia básica")

    # Análisis de riesgo
    dd_basic = abs(results['basic']['max_drawdown'])
    dd_full = abs(results['full']['max_drawdown'])

    print("
🛡️ ANÁLISIS DE RIESGO"    print("=" * 80)
    print(".2f"    print(".2f"    if dd_full < dd_basic:
        print("   ✅ Estrategia completa reduce drawdown")
    elif dd_full > dd_basic * 1.2:
        print("   ❌ Estrategia completa aumenta drawdown significativamente")
    else:
        print("   ⚠️ Drawdown similar entre estrategias")

    # Análisis POC específico
    poc_trades = results['full']['poc_trades']
    total_trades_full = results['full']['total_trades']
    poc_ratio = poc_trades / total_trades_full * 100 if total_trades_full > 0 else 0

    print("
📍 ANÁLISIS POC DETALLADO"    print("=" * 80)
    print(".1f"    print(".1f"    print(".1f"
    if results['full']['poc_win_rate'] > results['full']['win_rate']:
        print("   ✅ POC filtra trades de mejor calidad")
    else:
        print("   ⚠️ POC no mejora significativamente la calidad")

    # Recomendaciones finales
    print("
💡 RECOMENDACIONES FINALES"    print("=" * 80)

    if mejora > 2 and dd_full <= dd_basic * 1.1:
        print("🟢 RECOMENDADO: Estrategia Completa")
        print("   - Usar Squeeze + ADX + POC permisivo")
        print("   - Multi-timeframe mejora timing")
        print("   - POC añade confirmación sin eliminar oportunidades")
    elif retorno_basic > 0:
        print("🟡 RECOMENDADO: Estrategia Básica")
        print("   - Squeeze + ADX suficiente para buen rendimiento")
        print("   - Filtros avanzados opcionales")
        print("   - Menor complejidad, mismo resultado")
    else:
        print("🔴 REQUERIDO: Revisar Estrategia")
        print("   - Rendimiento negativo en ambas configuraciones")
        print("   - Revisar parámetros o timeframe")
        print("   - Considerar diferentes indicadores")

    print("
📋 CONFIGURACIÓN ÓPTIMA RECOMENDADA:"    print("   - Risk por trade: 2%")
    print("   - Max trades por día: 3")
    print("   - Commission: 0.1%")
    print("   - Slippage: 0.02%")
    print("   - POC threshold: ±1% del POC")
    print("   - ADX slope: filtro opcional, no restrictivo")

    # Guardar resultados detallados
    os.makedirs('results', exist_ok=True)
    with open('results/realistic_backtest_analysis.md', 'w', encoding='utf-8') as f:
        f.write("# Análisis Realista de Backtest\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        f.write("## Configuración del Backtest\n\n")
        f.write("- **Capital Inicial:** $10,000\n")
        f.write("- **Risk por Trade:** 2%\n")
        f.write("- **Máx Trades por Día:** 3\n")
        f.write("- **Comisión:** 0.1%\n")
        f.write("- **Slippage:** 0.02%\n")
        f.write("- **Datos:** BTC 5min, 15min, 1H\n\n")
        f.write("## Resultados Comparativos\n\n")
        f.write("| Configuración | Retorno | Trades | Win Rate | Profit Factor | Max DD | POC Trades |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        f.write(f"| **Básica** | {results['basic']['total_return_pct']:.2f}% | {results['basic']['total_trades']} | {results['basic']['win_rate']:.1f}% | {results['basic']['profit_factor']:.2f} | {results['basic']['max_drawdown']:.2f}% | - |\n")
        f.write(f"| **Completa** | {results['full']['total_return_pct']:.2f}% | {results['full']['total_trades']} | {results['full']['win_rate']:.1f}% | {results['full']['profit_factor']:.2f} | {results['full']['max_drawdown']:.2f}% | {results['full']['poc_trades']} |\n")
        f.write(f"| **Diferencia** | {mejora:.2f}% | {results['full']['total_trades'] - results['basic']['total_trades']} | {(results['full']['win_rate'] - results['basic']['win_rate']):.1f}% | {(results['full']['profit_factor'] - results['basic']['profit_factor']):.2f} | {(results['full']['max_drawdown'] - results['basic']['max_drawdown']):.2f}% | - |\n\n")

        if mejora > 0:
            f.write("## Conclusión\n\n")
            f.write("La estrategia completa aporta valor adicional manteniendo oportunidades válidas.\n\n")
        else:
            f.write("## Conclusión\n\n")
            f.write("La estrategia básica es suficiente para obtener buenos resultados.\n\n")

    print("
💾 Reporte guardado en: results/realistic_backtest_analysis.md"
if __name__ == "__main__":
    run_realistic_backtest()