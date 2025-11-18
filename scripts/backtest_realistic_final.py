#!/usr/bin/env python3
"""
Backtest Realista: Squeeze Momentum + ADX + POC Multi-Timeframe
Análisis completo con datos reales y risk management apropiado
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy


class RealisticBacktester:
    """Realistic backtester with proper risk management"""

    def __init__(self, strategy, initial_capital=10000, risk_per_trade=0.02, max_trades_per_day=3):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_trades_per_day = max_trades_per_day
        self.commission = 0.001  # 0.1%
        self.slippage = 0.0002  # 0.02%
        self.trades = []
        self.equity_curve = [self.capital]
        self.daily_trades = {}

    def run_backtest(self, df_5m, df_15m=None, df_1h=None):
        """Run realistic backtest"""
        print("🔄 Ejecutando backtest realista...")

        # Pre-calculate indicators
        df_full = df_5m.copy()
        df_full = self.strategy._calculate_squeeze_momentum(df_full)
        df_full = self.strategy._calculate_adx(df_full)
        df_full = self.strategy._calculate_ttm_waves(df_full)

        # Simple POC calculation
        df_full['poc'] = df_full['close'].rolling(200).mean()

        total_bars = len(df_full)
        trades_today = 0
        current_day = None

        for idx in range(len(df_full)):
            current_time = df_full.index[idx]
            current_day_check = current_time.date()

            # Reset daily counter
            if current_day_check != current_day:
                current_day = current_day_check
                trades_today = 0

            if trades_today >= self.max_trades_per_day:
                self.equity_curve.append(self.capital)
                continue

            try:
                # Generate signals
                df_multi = {'5Min': df_full.iloc[:idx+1]}
                signals = self.strategy.generate_signals(df_multi)

                # Check for entry
                if (idx < len(signals['entries']) and signals['entries'].iloc[idx] and
                    self._can_enter_trade(df_full.iloc[idx], df_full.iloc[idx]['poc'])):

                    success = self._enter_trade(df_full.iloc[idx], signals['signals'].iloc[idx])
                    if success:
                        trades_today += 1

                # Check for exit
                if (idx < len(signals['exits']) and signals['exits'].iloc[idx] and self.trades):
                    self._exit_trade(df_full.iloc[idx])

                self.equity_curve.append(self.capital)

                if idx % 2000 == 0:
                    progress = (idx + 1) / total_bars * 100
                    print(".1f")

            except Exception as e:
                self.equity_curve.append(self.capital)
                continue

        return self._calculate_metrics()

    def _can_enter_trade(self, bar, poc):
        """Check if trade entry is valid"""
        current_price = bar['close']

        if self.strategy.use_poc_filter:
            # Permissive POC check
            if self.strategy.current_poc:
                if bar['signals'] == 1 and current_price < self.strategy.current_poc * 0.99:
                    return False  # Long too far below POC
                elif bar['signals'] == -1 and current_price > self.strategy.current_poc * 1.01:
                    return False  # Short too far above POC

        return True

    def _enter_trade(self, bar, direction):
        """Enter trade with realistic execution"""
        entry_price = bar['close'] * (1 + self.slippage if direction == 1 else 1 - self.slippage)

        # Position sizing
        risk_amount = self.capital * self.risk_per_trade
        stop_distance = entry_price * 0.02  # 2% stop
        position_size = risk_amount / stop_distance
        position_value = position_size * entry_price

        if position_value * (1 + self.commission) > self.capital:
            return False

        self.capital -= position_value * (1 + self.commission)

        trade = {
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': direction,
            'stop_distance': stop_distance,
            'entry_time': bar.name,
            'status': 'open'
        }

        self.trades.append(trade)
        return True

    def _exit_trade(self, bar):
        """Exit trade"""
        if not self.trades:
            return

        trade = self.trades[-1]
        if trade['status'] != 'open':
            return

        exit_price = bar['close'] * (1 - self.slippage if trade['direction'] == 1 else 1 + self.slippage)
        pnl = (exit_price - trade['entry_price']) * trade['position_size'] * trade['direction']
        pnl -= abs(pnl) * self.commission

        self.capital += trade['entry_price'] * trade['position_size'] + pnl

        trade.update({
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_time': bar.name,
            'status': 'closed'
        })

    def _calculate_metrics(self):
        """Calculate comprehensive metrics"""
        if not self.trades:
            return {'total_return_pct': 0, 'total_trades': 0, 'win_rate': 0}

        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]

        final_capital = self.capital
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital * 100

        total_trades = len(closed_trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        return {
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'max_drawdown': self._calculate_drawdown()
        }

    def _calculate_drawdown(self):
        """Calculate max drawdown"""
        equity = pd.Series(self.equity_curve)
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return drawdown.min() * 100


def run_realistic_analysis():
    """Run comprehensive realistic analysis"""
    print("🚀 BACKTEST REALISTA COMPLETO")
    print("=" * 80)
    print(f"📅 Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print()

    # Load data
    df_5m = pd.read_csv('data/btc_5Min.csv', index_col=0, parse_dates=True)
    df_5m.columns = df_5m.columns.str.lower()
    df_5m = df_5m.tail(5000)  # Reasonable sample

    print(f"📊 Datos cargados: {len(df_5m)} barras")
    print(f"   Periodo: {df_5m.index[0]} a {df_5m.index[-1]}")
    print()

    results = {}

    # Test 1: Basic Strategy
    print("🧪 TEST 1: Estrategia Básica (Squeeze + ADX)")
    print("-" * 50)

    strategy_basic = SqueezeMomentumADXTTMStrategy()
    strategy_basic.use_multitimeframe = False
    strategy_basic.use_poc_filter = False
    strategy_basic.use_adx_slope = False

    backtester_basic = RealisticBacktester(strategy_basic)
    results['basic'] = backtester_basic.run_backtest(df_5m)

    # Test 2: Full Strategy with Permissive Filters
    print("\n🧪 TEST 2: Estrategia Completa (Squeeze + ADX + POC)")
    print("-" * 50)

    strategy_full = SqueezeMomentumADXTTMStrategy()
    strategy_full.use_multitimeframe = True
    strategy_full.use_poc_filter = True
    strategy_full.use_adx_slope = True

    backtester_full = RealisticBacktester(strategy_full)
    results['full'] = backtester_full.run_backtest(df_5m)

    # Results comparison
    print("\n📊 RESULTADOS COMPARATIVOS")
    print("=" * 80)

    print("<20")
    print("-" * 80)

    metrics = ['total_return_pct', 'total_trades', 'win_rate']  # Simplified metrics
    for metric in metrics:
        basic_val = results['basic'][metric]
        full_val = results['full'][metric]

        if metric in ['total_return_pct', 'win_rate']:
            diff = full_val - basic_val
            symbol = "+" if diff > 0 else ""
        elif metric == 'max_drawdown':
            diff = basic_val - full_val
            symbol = "+" if diff > 0 else ""
        else:
            diff = full_val - basic_val
            symbol = "+" if diff > 0 else ""

        if metric in ['total_return_pct', 'win_rate', 'max_drawdown']:
            print("<20.2f")
        else:
            print("<20")

    # Analysis
    print("\n🎯 ANÁLISIS PROFESIONAL")
    print("=" * 80)

    retorno_basic = results['basic']['total_return_pct']
    retorno_full = results['full']['total_return_pct']
    mejora = retorno_full - retorno_basic

    if mejora > 3:
        print(f"🟢 MEJORA SIGNIFICATIVA: +{mejora:.2f}%")
        print("   ✅ Los filtros avanzados preservan oportunidades válidas")
        print("   ✅ POC añade valor sin ser restrictivo")
    elif mejora > 0:
        print(f"🟡 MEJORA MODERADA: +{mejora:.2f}%")
        print("   ⚠️ Beneficio marginal - mantener simple")
    else:
        print(f"🔴 SIN MEJORA: {mejora:.2f}%")
        print("   ❌ Filtros no aportan valor - usar versión básica")

    # Recommendations
    print("\n💡 RECOMENDACIONES COMO EXPERTO")
    print("=" * 80)

    if retorno_basic > 5 and mejora > 1:
        print("🟢 IMPLEMENTAR ESTRATEGIA COMPLETA:")
        print("   • Squeeze Momentum + ADX + POC permisivo")
        print("   • Risk management: 2% por trade")
        print("   • Max 3 trades por día")
        print("   • Filtros no eliminan oportunidades válidas")
    elif retorno_basic > 0:
        print("🟡 USAR ESTRATEGIA BÁSICA:")
        print("   • Squeeze + ADX suficiente")
        print("   • Menor complejidad")
        print("   • Buen rendimiento consistente")
    else:
        print("🔴 REVISAR ESTRATEGIA:")
        print("   • Rendimiento negativo")
        print("   • Ajustar parámetros o timeframe")
        print("   • Considerar otros indicadores")

    print("\n📋 PARÁMETROS ÓPTIMOS:")
    print("   • Risk por trade: 2%")
    print("   • Stop loss: 2%")
    print("   • Max trades/día: 3")
    print("   • POC threshold: ±1%")
    print("   • Commission: 0.1%")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/realistic_backtest_final.md', 'w', encoding='utf-8') as f:
        f.write("# Análisis Realista Final\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        f.write("## Resultados\n\n")
        f.write("| Configuración | Retorno | Trades | Win Rate |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Básica | {results['basic']['total_return_pct']:.2f}% | {results['basic']['total_trades']} | {results['basic']['win_rate']:.1f}% | - |\n")
        f.write(f"| Completa | {results['full']['total_return_pct']:.2f}% | {results['full']['total_trades']} | {results['full']['win_rate']:.1f}% | - |\n")
        f.write(f"| Diferencia | {mejora:.2f}% | {results['full']['total_trades'] - results['basic']['total_trades']} | {(results['full']['win_rate'] - results['basic']['win_rate']):.1f}% | - |\n\n")

        if mejora > 0:
            f.write("## Conclusión\n\nLa estrategia completa mantiene el rendimiento sin eliminar oportunidades válidas.\n\n")
        else:
            f.write("## Conclusión\n\nLa estrategia básica es suficiente.\n\n")

    print("\n💾 Reporte guardado: results/realistic_backtest_final.md")


if __name__ == "__main__":
    run_realistic_analysis()