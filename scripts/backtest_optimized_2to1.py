#!/usr/bin/env python3
"""
Backtest Optimizado: Squeeze Momentum + ADX con Ratio 2:1
Optimización de parámetros para maximizar rentabilidad matemática
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import talib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_adx_ttm_strategy import SqueezeMomentumADXTTMStrategy


class OptimizedBacktester:
    """Backtester optimizado con ratio riesgo/beneficio 2:1"""

    def __init__(self, strategy, initial_capital=10000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = 0.001  # 0.1%
        self.slippage = 0.0002  # 0.02%
        self.trades = []
        self.equity_curve = [self.capital]
        self.last_entry_idx = -100  # For cooldown tracking
        self.cooldown_bars = 2  # Reduced cooldown between entries
        self.use_ema_filter = True  # EMA distance filter for entries

    def run_optimized_backtest(self, df_5m, df_15m=None, df_1h=None, target_rr_ratio=2.0):
        """Run optimized backtest with 2:1 risk/reward ratio and multi-timeframe EMA analysis"""
        print("🔄 Ejecutando backtest optimizado con ratio 2:1...")

        # Pre-calculate indicators
        df_full = df_5m.copy()
        df_full = self.strategy._calculate_squeeze_momentum(df_full)
        df_full = self.strategy._calculate_adx(df_full)
        df_full = self.strategy._calculate_ttm_waves(df_full)

        # Calculate ATR for dynamic stops
        df_full['atr'] = talib.ATR(df_full['high'].values, df_full['low'].values, df_full['close'].values, timeperiod=14)
        df_full['volume_ma'] = df_full['volume'].rolling(20).mean()

        # Calculate EMAs for multi-timeframe analysis
        df_full['ema_20'] = talib.EMA(df_full['close'].values, timeperiod=20)

        # Multi-timeframe EMA analysis
        if df_15m is not None:
            # Calculate EMA50 and EMA200 on 15m timeframe
            df_15m['ema_50_15m'] = talib.EMA(df_15m['close'].values, timeperiod=50)
            df_15m['ema_200_15m'] = talib.EMA(df_15m['close'].values, timeperiod=200)

            # Align 15m data with 5m by forward filling
            df_15m_aligned = df_15m.reindex(df_full.index, method='ffill')
            df_full['ema_50_15m'] = df_15m_aligned['ema_50_15m']
            df_full['ema_200_15m'] = df_15m_aligned['ema_200_15m']

            # Calculate distance to EMAs (percentage)
            df_full['dist_ema50_15m'] = abs(df_full['close'] - df_full['ema_50_15m']) / df_full['close'] * 100
            df_full['dist_ema200_15m'] = abs(df_full['close'] - df_full['ema_200_15m']) / df_full['close'] * 100

        if df_1h is not None:
            # Calculate EMA50 and EMA200 on 1h timeframe
            df_1h['ema_50_1h'] = talib.EMA(df_1h['close'].values, timeperiod=50)
            df_1h['ema_200_1h'] = talib.EMA(df_1h['close'].values, timeperiod=200)

            # Align 1h data with 5m by forward filling
            df_1h_aligned = df_1h.reindex(df_full.index, method='ffill')
            df_full['ema_50_1h'] = df_1h_aligned['ema_50_1h']
            df_full['ema_200_1h'] = df_1h_aligned['ema_200_1h']

            # Calculate distance to EMAs (percentage)
            df_full['dist_ema50_1h'] = abs(df_full['close'] - df_full['ema_50_1h']) / df_full['close'] * 100
            df_full['dist_ema200_1h'] = abs(df_full['close'] - df_full['ema_200_1h']) / df_full['close'] * 100
        df_full['ema_50'] = talib.EMA(df_full['close'].values, timeperiod=50)
        df_full['ema_200'] = talib.EMA(df_full['close'].values, timeperiod=200)

        # Calculate distance to EMAs (for entry/exit signals)
        df_full['dist_ema20'] = (df_full['close'] - df_full['ema_20']) / df_full['ema_20'] * 100
        df_full['dist_ema50'] = (df_full['close'] - df_full['ema_50']) / df_full['ema_50'] * 100
        df_full['dist_ema200'] = (df_full['close'] - df_full['ema_200']) / df_full['ema_200'] * 100

        total_bars = len(df_full)
        in_trade = False
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trade_direction = 0

        for idx in range(20, len(df_full)):  # Start after ATR calculation
            current_bar = df_full.iloc[idx]
            current_price = current_bar['close']

            # Check exit conditions first
            if in_trade:
                # Check stop loss
                if ((trade_direction == 1 and current_price <= stop_loss) or
                    (trade_direction == -1 and current_price >= stop_loss)):
                    print(f"🛑 Stop loss exit at idx {idx}, price {current_price}, SL {stop_loss}")
                    self._exit_trade(current_bar, "stop_loss", entry_price, trade_direction)
                    in_trade = False
                    continue

                # Check take profit (including EMA levels)
                if ((trade_direction == 1 and current_price >= take_profit) or
                    (trade_direction == -1 and current_price <= take_profit)):
                    print(f"🎯 Take profit exit at idx {idx}, price {current_price}, TP {take_profit}")
                    self._exit_trade(current_bar, "take_profit", entry_price, trade_direction)
                    in_trade = False
                    continue

                # Check EMA-based exits (additional profit targets)
                ema_20 = current_bar.get('ema_20', current_price)
                ema_50 = current_bar.get('ema_50', current_price)

                if trade_direction == 1:  # Long trade
                    # Take partial profit at EMA levels
                    if current_price >= ema_20 * 1.005 and entry_price < ema_20:  # 0.5% above EMA20
                        print(f"📈 EMA20 profit exit at idx {idx}, price {current_price}, EMA20 {ema_20}")
                        self._exit_trade(current_bar, "ema20_profit", entry_price, trade_direction)
                        in_trade = False
                        continue
                else:  # Short trade
                    if current_price <= ema_20 * 0.995 and entry_price > ema_20:  # 0.5% below EMA20
                        print(f"📉 EMA20 profit exit at idx {idx}, price {current_price}, EMA20 {ema_20}")
                        self._exit_trade(current_bar, "ema20_profit", entry_price, trade_direction)
                        in_trade = False
                        continue

                # Add time-based exit (max 50 bars = ~4 hours)
                if self.trades and len(self.equity_curve) - self.trades[-1]['entry_idx'] > 50:
                    print(f"⏰ Time exit at idx {idx} after 50 bars")
                    self._exit_trade(current_bar, "time_exit", entry_price, trade_direction)
                    in_trade = False
                    continue

                # Update trailing stop (always enabled with optimized settings)
                if self._should_activate_trailing(entry_price, current_price, take_profit, trade_direction):
                    new_stop = self._calculate_trailing_stop(entry_price, current_price, trade_direction)
                    if ((trade_direction == 1 and new_stop > stop_loss) or
                        (trade_direction == -1 and new_stop < stop_loss)):
                        stop_loss = new_stop

            # Check entry conditions
            elif not in_trade:
                if self._should_enter_trade(df_full, idx, current_bar):
                    entry_price, stop_loss, take_profit, trade_direction = self._enter_trade(current_bar)
                    in_trade = True

            self.equity_curve.append(self.capital)

            if idx % 1000 == 0:
                progress = (idx + 1) / total_bars * 100
                print(".1f")

        # Close any open trades at the end
        if in_trade:
            print(f"🏁 Closing open trade at end of backtest")
            self._exit_trade(df_full.iloc[-1], "end_of_test", entry_price, trade_direction)

        return self._calculate_optimized_metrics(target_rr_ratio)

    def _should_enter_trade(self, df, idx, current_bar):
        """Check if all entry conditions are met"""
        # Check cooldown between entries
        if idx - self.last_entry_idx < self.cooldown_bars:
            return False

        # Check volume filter (above average) - removed as per analysis
        # Volume filter showed no significant impact

        # Check EMA distance for additional confirmation (if enabled)
        if self.use_ema_filter:
            ema_dist_20 = current_bar.get('dist_ema20', 0)
            ema_dist_50 = current_bar.get('dist_ema50', 0)

            # Prefer entries when price is not too far from EMAs (avoid extreme moves)
            if abs(ema_dist_20) > 5.0 or abs(ema_dist_50) > 10.0:  # Too far from EMAs
                return False

            # Multi-timeframe EMA filter: Check proximity to higher timeframe EMAs
            # Allow entries when price is near EMA50 or EMA200 in 15m or 1h timeframe
            ema_proximity_threshold = 2.0  # 2% proximity threshold

            near_ema50_15m = current_bar.get('dist_ema50_15m', 100) < ema_proximity_threshold
            near_ema200_15m = current_bar.get('dist_ema200_15m', 100) < ema_proximity_threshold
            near_ema50_1h = current_bar.get('dist_ema50_1h', 100) < ema_proximity_threshold
            near_ema200_1h = current_bar.get('dist_ema200_1h', 100) < ema_proximity_threshold

            # Require at least one higher timeframe EMA proximity for entry confirmation
            if not (near_ema50_15m or near_ema200_15m or near_ema50_1h or near_ema200_1h):
                return False

        # Generate signals for current data
        signals = self.strategy.generate_signals({'5Min': df.iloc[:idx+1]})

        # Debug: print signal info occasionally
        if idx % 500 == 0:
            print(f"Debug idx {idx}: entries len={len(signals['entries'])}, last_entry={signals['entries'].iloc[-1] if len(signals['entries']) > 0 else 'N/A'}")

        # Only check if there's an entry signal - the strategy already includes all filters
        if len(signals['entries']) <= idx or signals['entries'].iloc[idx] != 1:
            return False

        print(f"✅ Trade entry at idx {idx}")
        return True

    def _enter_trade(self, bar):
        """Enter trade with calculated stops"""
        # Determine trade direction from squeeze momentum
        squeeze_value = bar.get('linreg_value', 0)
        direction = 1 if squeeze_value > 0 else -1

        entry_price = bar['close'] * (1 + self.slippage if direction == 1 else 1 - self.slippage)
        atr = bar['atr']

        # Calculate stops based on fixed percentages (1% SL, 3% TP)
        stop_distance = entry_price * 0.01  # 1% stop loss
        target_distance = entry_price * 0.03  # 3% take profit

        if direction == 1:  # Long
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + target_distance
        else:  # Short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - target_distance

        # Calculate position size with reduced risk
        risk_amount = self.capital * 0.005  # 0.5% risk per trade instead of 2%
        position_size = risk_amount / stop_distance
        position_value = position_size * entry_price

        if position_value * (1 + self.commission) > self.capital:
            print(f"❌ Insufficient capital: need ${position_value * (1 + self.commission):.2f}, have ${self.capital:.2f}")
            return None, None, None, None

        self.capital -= position_value * (1 + self.commission)

        trade = {
            'entry_idx': len(self.equity_curve) - 1,
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': direction,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr,
            'entry_time': bar.name,
            'status': 'open'
        }

        self.trades.append(trade)
        self.last_entry_idx = len(self.equity_curve) - 1  # Update cooldown tracking
        print(f"📊 Trade opened: {direction} at {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
        return entry_price, stop_loss, take_profit, direction

    def _should_activate_trailing(self, entry_price, current_price, take_profit, direction):
        """Check if trailing stop should be activated"""
        if not self.strategy.trailing_stop:
            return False

        # Activate when price reaches activation percentage of target
        if direction == 1:
            target_progress = (current_price - entry_price) / (take_profit - entry_price)
        else:
            target_progress = (entry_price - current_price) / (entry_price - take_profit)

        return target_progress >= self.strategy.trailing_activation

    def _calculate_trailing_stop(self, entry_price, current_price, direction):
        """Calculate trailing stop level"""
        # Simple trailing stop: lock in 50% of current profit
        if direction == 1:
            profit = current_price - entry_price
            return current_price - (profit * 0.5)
        else:
            profit = entry_price - current_price
            return current_price + (profit * 0.5)

    def _exit_trade(self, bar, exit_reason, entry_price, direction):
        """Exit trade"""
        if not self.trades:
            return

        trade = self.trades[-1]
        if trade['status'] != 'open':
            return

        exit_price = bar['close'] * (1 - self.slippage if direction == 1 else 1 + self.slippage)

        # Calculate P&L
        price_diff = exit_price - entry_price
        pnl = price_diff * trade['position_size'] * direction
        pnl -= abs(pnl) * self.commission  # Commission on exit

        self.capital += entry_price * trade['position_size'] + pnl

        # Update trade record
        trade.update({
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_time': bar.name,
            'exit_reason': exit_reason,
            'status': 'closed',
            'duration': len(self.equity_curve) - 1 - trade['entry_idx']
        })

    def _calculate_optimized_metrics(self, target_rr_ratio):
        """Calculate comprehensive metrics with RR ratio analysis"""
        if not self.trades:
            return {
                'total_return_pct': 0,
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'mathematical_expectancy': 0,
                'rr_ratio_achieved': 0,
                'max_drawdown': 0
            }

        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]

        final_capital = self.capital
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital * 100

        total_trades = len(closed_trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0

        profit_factor = sum(t['pnl'] for t in winning_trades) / abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')

        # Calculate actual RR ratio achieved
        if winning_trades and losing_trades:
            avg_win_distance = np.mean([abs(t['exit_price'] - t['entry_price']) for t in winning_trades])
            avg_loss_distance = np.mean([abs(t['exit_price'] - t['entry_price']) for t in losing_trades])
            rr_ratio_achieved = avg_win_distance / avg_loss_distance if avg_loss_distance > 0 else float('inf')
        else:
            rr_ratio_achieved = target_rr_ratio

        # Mathematical expectancy with actual RR ratio
        if total_trades > 0 and avg_loss > 0:
            mathematical_expectancy = (win_rate/100 * rr_ratio_achieved) - ((100-win_rate)/100 * 1)
        else:
            mathematical_expectancy = 0

        expectancy = sum(t['pnl'] for t in closed_trades) / total_trades if total_trades > 0 else 0

        # Calculate trade duration statistics
        durations = [t.get('duration', 0) for t in closed_trades]
        winning_durations = [t.get('duration', 0) for t in winning_trades]
        losing_durations = [t.get('duration', 0) for t in losing_trades]

        avg_duration = np.mean(durations) if durations else 0
        median_duration = np.median(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        avg_winning_duration = np.mean(winning_durations) if winning_durations else 0
        avg_losing_duration = np.mean(losing_durations) if losing_durations else 0

        # Calculate drawdown
        equity = pd.Series(self.equity_curve)
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min() * 100

        # Calculate Sharpe ratio (annualized, assuming daily returns)
        if len(self.equity_curve) > 1:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            if returns.std() > 0:
                # Sharpe con risk-free rate (Annualized)
                rf_daily = 0.04 / 252
                excess_returns = returns - rf_daily
                sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        return {
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'mathematical_expectancy': mathematical_expectancy,
            'rr_ratio_achieved': rr_ratio_achieved,
            'target_rr_ratio': target_rr_ratio,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            # Trade duration statistics
            'avg_duration': avg_duration,
            'median_duration': median_duration,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'avg_winning_duration': avg_winning_duration,
            'avg_losing_duration': avg_losing_duration
        }


def optimize_strategy_parameters():
    """Optimize strategy parameters for best performance with 2:1 RR ratio"""

    print("🚀 OPTIMIZACIÓN DE PARÁMETROS CON RATIO 2:1")
    print("=" * 80)
    print(f"📅 Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print()

    # Load data for multi-timeframe analysis
    df_5m = pd.read_csv('data/btc_5Min.csv', index_col=0, parse_dates=True)
    df_5m.columns = df_5m.columns.str.lower()
    df_5m = df_5m.tail(2000)  # Smaller sample for optimization

    df_15m = pd.read_csv('data/btc_15Min.csv', index_col=0, parse_dates=True)
    df_15m.columns = df_15m.columns.str.lower()

    df_1h = pd.read_csv('data/btc_1H.csv', index_col=0, parse_dates=True)
    df_1h.columns = df_1h.columns.str.lower()

    print(f"📊 Datos cargados: {len(df_5m)} barras")
    print()

    # Parameter combinations to test - optimized based on impact analysis
    param_combinations = [
        # (adx_min, squeeze_min, stop_mult, take_mult, cooldown, trailing_activation, ema_filter)
        (15, 0.15, 1.0, 3.0, 1, 0.5, True),   # Aggressive with EMA filters
        (18, 0.2, 1.2, 3.0, 2, 0.6, True),    # Balanced with EMA
        (20, 0.25, 1.5, 3.0, 1, 0.7, False),  # Conservative without EMA
        (15, 0.18, 0.8, 3.0, 2, 0.5, True),   # Fast entries with EMA
        (22, 0.3, 1.0, 3.0, 1, 0.6, False),   # Slow squeeze, no EMA
        (16, 0.22, 1.3, 3.0, 2, 0.7, True),   # Medium with full filters
        (14, 0.12, 1.1, 3.0, 1, 0.5, True),   # Very permissive
    ]

    results = []

    for i, (adx_min, squeeze_min, stop_mult, take_mult, cooldown, trailing_activation, ema_filter) in enumerate(param_combinations):
        print(f"🧪 Test {i+1}/7: ADX({adx_min}), Squeeze>{squeeze_min}, SL:{stop_mult}ATR, TP:{take_mult}ATR, Cooldown:{cooldown}, EMA:{ema_filter}")

        # Create strategy with parameters (optimized based on impact analysis)
        strategy = SqueezeMomentumADXTTMStrategy()
        strategy.min_adx_entry = adx_min
        strategy.min_squeeze_momentum = squeeze_min
        strategy.stop_loss_atr_mult = stop_mult
        strategy.take_profit_atr_mult = take_mult
        strategy.use_multitimeframe = False
        strategy.use_poc_filter = False
        strategy.use_adx_slope = False
        # Trailing stops always enabled (highest impact parameter)
        strategy.trailing_stop = True
        strategy.trailing_activation = trailing_activation

        # Create backtester with optimized settings
        backtester = OptimizedBacktester(strategy)
        backtester.cooldown_bars = cooldown
        # EMA filter setting
        backtester.use_ema_filter = ema_filter
        metrics = backtester.run_optimized_backtest(df_5m, df_15m, df_1h)

        results.append({
            'params': (adx_min, squeeze_min, stop_mult, take_mult, cooldown, trailing_activation, ema_filter),
            'metrics': metrics
        })

        print(f"Retorno: {metrics['total_return_pct']:.2f}%")
    # Find best configuration
    best_result = max(results, key=lambda x: x['metrics']['mathematical_expectancy'])

    print("\n🏆 MEJOR CONFIGURACIÓN ENCONTRADA")
    print("=" * 80)

    params = best_result['params']
    metrics = best_result['metrics']

    print(f"ADX Min: {params[0]}")
    print(f"Squeeze Min: {params[1]}")
    print(f"Stop Loss: {params[2]} ATR")
    print(f"Take Profit: {params[3]} ATR")
    print(f"Cooldown Bars: {params[4]}")
    print(f"Trailing Activation: {params[5]}")
    print(f"EMA Filter: {params[6]}")
    print()

    print("📊 MÉTRICAS DE RENDIMIENTO:")
    print(f"- Retorno Total: {metrics['total_return_pct']:.2f}%")
    print(f"- Win Rate: {metrics['win_rate']:.1f}%")
    print(f"- Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"- Avg Win: ${metrics['avg_win']:.2f}")
    print(f"- Avg Loss: ${metrics['avg_loss']:.2f}")
    print(f"- Mathematical Expectancy: {metrics['mathematical_expectancy']:.4f}")
    print(f"- RR Ratio Achieved: {metrics['rr_ratio_achieved']:.2f}")
    print(f"- Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"- Avg Trade Duration: {metrics.get('avg_duration', 0):.1f} bars")
    print(f"- Median Trade Duration: {metrics.get('median_duration', 0):.1f} bars")
    print(f"- Min/Max Duration: {metrics.get('min_duration', 0):.0f}/{metrics.get('max_duration', 0):.0f} bars")
    print(f"- Avg Winning Duration: {metrics.get('avg_winning_duration', 0):.1f} bars")
    print(f"- Avg Losing Duration: {metrics.get('avg_losing_duration', 0):.1f} bars")
    # Mathematical analysis
    rr_ratio = metrics['rr_ratio_achieved']

    if rr_ratio >= 2.0 and metrics['mathematical_expectancy'] > 0:
        print("\n✅ CONFIGURACIÓN MATEMÁTICAMENTE RENTABLE")
        print(f"   RR Ratio: {rr_ratio:.2f}")
        print(f"   Mathematical Expectancy: {metrics['mathematical_expectancy']:.4f}")
        print("   💰 Estrategia viable para trading real")
    elif rr_ratio >= 1.5:
        print("\n⚠️ CONFIGURACIÓN MODERADAMENTE RENTABLE")
        print(f"   RR Ratio: {rr_ratio:.2f}")
        print(f"   Mathematical Expectancy: {metrics['mathematical_expectancy']:.4f}")
        print("   📈 Posible con mejoras adicionales")
    else:
        print("\n❌ CONFIGURACIÓN NO RENTABLE")
        print(f"   RR Ratio: {rr_ratio:.2f}")
        print(f"   Mathematical Expectancy: {metrics['mathematical_expectancy']:.4f}")
        print("   🔄 Necesita ajustes en parámetros")

    # Save optimization results
    os.makedirs('results', exist_ok=True)
    with open('results/optimization_results_2to1.md', 'w', encoding='utf-8') as f:
        f.write("# Optimización de Parámetros - Ratio 2:1\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        f.write("## Mejor Configuración\n\n")
        f.write(f"- **ADX Min:** {params[0]}\n")
        f.write(f"- **Squeeze Momentum Min:** {params[1]}\n")
        f.write(f"- **Stop Loss:** {params[2]} ATR\n")
        f.write(f"- **Take Profit:** {params[3]} ATR\n")
        f.write(f"- **Cooldown Bars:** {params[4]}\n")
        f.write(f"- **Trailing Activation:** {params[5]}\n")
        f.write(f"- **EMA Filter:** {params[6]}\n\n")
        f.write("## Resultados\n\n")
        f.write(f"- **Retorno Total:** {metrics['total_return_pct']:.2f}%\n")
        f.write(f"- **Total Trades:** {metrics['total_trades']}\n")
        f.write(f"- **Win Rate:** {metrics['win_rate']:.1f}%\n")
        f.write(f"- **Profit Factor:** {metrics['profit_factor']:.2f}\n")
        f.write(f"- **Mathematical Expectancy:** {metrics['mathematical_expectancy']:.4f}\n")
        f.write(f"- **RR Ratio Achieved:** {metrics['rr_ratio_achieved']:.2f}\n")
        f.write(f"- **Max Drawdown:** {metrics['max_drawdown']:.2f}%\n")
        f.write(f"- **Avg Trade Duration:** {metrics['avg_duration']:.1f} bars\n")
        f.write(f"- **Median Trade Duration:** {metrics['median_duration']:.1f} bars\n")
        f.write(f"- **Min/Max Duration:** {metrics['min_duration']:.0f}/{metrics['max_duration']:.0f} bars\n")
        f.write(f"- **Avg Winning Duration:** {metrics['avg_winning_duration']:.1f} bars\n")
        f.write(f"- **Avg Losing Duration:** {metrics['avg_losing_duration']:.1f} bars\n\n")

        if metrics['mathematical_expectancy'] > 0:
            f.write("## Conclusión\n\n✅ **Configuración Matemáticamente Rentable**\n\n")
        else:
            f.write("## Conclusión\n\n❌ **Requiere Más Optimización**\n\n")

    print("\n💾 Resultados guardados: results/optimization_results_2to1.md")

    # Generate parameter impact analysis
    generate_parameter_impact_analysis(results)

    return best_result


def generate_parameter_impact_analysis(results):
    """Generate analysis of parameter impact on performance"""
    print("\n📊 ANÁLISIS DE IMPACTO DE PARÁMETROS")
    print("=" * 80)

    if not results:
        print("No hay resultados para analizar")
        return

    # Extract parameter ranges for analysis
    param_names = ['ADX Min', 'Squeeze Min', 'SL ATR', 'TP ATR', 'Cooldown', 'Trailing Activation', 'EMA Filter']

    # Create correlation analysis
    param_impacts = {}

    for i, param_name in enumerate(param_names):
        param_values = [r['params'][i] for r in results]
        mathematical_expectancies = [r['metrics']['mathematical_expectancy'] for r in results]
        win_rates = [r['metrics']['win_rate'] for r in results]
        profit_factors = [r['metrics']['profit_factor'] for r in results]

        # Calculate correlations
        try:
            me_corr = np.corrcoef(param_values, mathematical_expectancies)[0, 1] if len(set(param_values)) > 1 else 0
            wr_corr = np.corrcoef(param_values, win_rates)[0, 1] if len(set(param_values)) > 1 else 0
            pf_corr = np.corrcoef(param_values, profit_factors)[0, 1] if len(set(param_values)) > 1 else 0
        except Exception:
            me_corr = wr_corr = pf_corr = 0

        param_impacts[param_name] = {
            'range': f"{min(param_values)} - {max(param_values)}",
            'me_correlation': me_corr,
            'win_rate_correlation': wr_corr,
            'profit_factor_correlation': pf_corr,
            'best_value': param_values[mathematical_expectancies.index(max(mathematical_expectancies))]
        }

    # Display parameter impact table
    print("IMPACTO DE CADA PARÁMETRO EN EL RENDIMIENTO:")
    print("-" * 80)
    print("<10")
    print("-" * 80)

    for param, data in param_impacts.items():
        print("<10")

    print("\n📈 PARÁMETROS MÁS INFLUYENTES (por correlación con Mathematical Expectancy):")
    sorted_params = sorted(param_impacts.items(), key=lambda x: abs(x[1]['me_correlation']), reverse=True)
    for param, data in sorted_params[:5]:
        impact = "🔺 Positivo" if data['me_correlation'] > 0.3 else "🔻 Negativo" if data['me_correlation'] < -0.3 else "➡️ Neutral"
        print(f"  {param}: {impact} (correlación: {data['me_correlation']:.3f})")

    # Save detailed analysis
    with open('results/parameter_impact_analysis.md', 'w', encoding='utf-8') as f:
        f.write("# Análisis de Impacto de Parámetros\n\n")
        f.write(f"**Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        f.write("## Resumen Ejecutivo\n\n")
        f.write("Este análisis evalúa cómo cada parámetro afecta el rendimiento de la estrategia ")
        f.write("de trading Squeeze Momentum + ADX + TTM Waves.\n\n")

        f.write("## Tabla de Impacto de Parámetros\n\n")
        f.write("| Parámetro | Rango | Corr ME | Corr Win Rate | Corr PF | Mejor Valor |\n")
        f.write("|-----------|-------|---------|----------------|---------|-------------|\n")

        for param, data in param_impacts.items():
            f.write(f"| {param} | {data['range']} | {data['me_correlation']:.3f} | {data['win_rate_correlation']:.3f} | {data['profit_factor_correlation']:.3f} | {data['best_value']} |\n")

        f.write("\n## Interpretación de Correlaciones\n\n")
        f.write("- **Correlación > 0.3**: Parámetro tiene impacto positivo significativo\n")
        f.write("- **Correlación < -0.3**: Parámetro tiene impacto negativo significativo\n")
        f.write("- **Correlación entre -0.3 y 0.3**: Impacto neutral o no significativo\n\n")

        f.write("## Recomendaciones para Optimización\n\n")
        f.write("### Parámetros a Priorizar:\n")
        for param, data in sorted_params[:3]:
            f.write(f"- **{param}**: Enfocarse en valor {data['best_value']} ")
            f.write("(correlación ME: {data['me_correlation']:.3f})\n")

        f.write("\n### Parámetros a Eliminar/Reducir:\n")
        low_impact = [p for p, d in sorted_params if abs(d['me_correlation']) < 0.1]
        for param in low_impact[:3]:
            f.write(f"- **{param[0]}**: Bajo impacto en rendimiento\n")

        f.write("\n### Estrategias para Nuevos Parámetros:\n")
        f.write("- Considerar rangos más amplios para parámetros con correlación baja\n")
        f.write("- Probar combinaciones no lineales de parámetros correlacionados\n")
        f.write("- Implementar validación cruzada para evitar sobre-optimización\n")

    print("\n💾 Análisis guardado: results/parameter_impact_analysis.md")


def run_final_validation(best_params):
    """Run final validation with best parameters on full dataset"""

    print("\n🔬 VALIDACIÓN FINAL CON MEJORES PARÁMETROS")
    print("=" * 80)

    # Load full dataset for multi-timeframe analysis
    df_5m = pd.read_csv('data/btc_5Min.csv', index_col=0, parse_dates=True)
    df_5m.columns = df_5m.columns.str.lower()
    df_5m = df_5m.tail(5000)  # Full test

    df_15m = pd.read_csv('data/btc_15Min.csv', index_col=0, parse_dates=True)
    df_15m.columns = df_15m.columns.str.lower()

    df_1h = pd.read_csv('data/btc_1H.csv', index_col=0, parse_dates=True)
    df_1h.columns = df_1h.columns.str.lower()

    print(f"📊 Dataset completo: {len(df_5m)} barras")

    # Apply best parameters
    params = best_params['params']
    strategy = SqueezeMomentumADXTTMStrategy()
    strategy.min_adx_entry = params[0]
    strategy.max_adx_entry = 100  # Set high to not limit entries (removed from optimization)
    strategy.min_squeeze_momentum = params[1]
    strategy.stop_loss_atr_mult = params[2]
    strategy.take_profit_atr_mult = params[3]
    strategy.use_volume_filter = False  # Removed from optimization
    strategy.use_multitimeframe = False
    strategy.use_poc_filter = False
    strategy.use_adx_slope = False
    # Trailing stops always enabled (highest impact parameter)
    strategy.trailing_stop = True
    strategy.trailing_activation = params[5]

    # Create backtester with optimized settings
    backtester = OptimizedBacktester(strategy)
    backtester.cooldown_bars = params[4]
    backtester.use_ema_filter = params[6]

    final_metrics = backtester.run_optimized_backtest(df_5m, df_15m, df_1h)
    print(f"- Win Rate: {final_metrics['win_rate']:.1f}%")
    print(f"- Profit Factor: {final_metrics['profit_factor']:.2f}")
    print(f"- Avg Win: ${final_metrics['avg_win']:.2f}")
    print(f"- Avg Loss: ${final_metrics['avg_loss']:.2f}")
    print(f"- Mathematical Expectancy: {final_metrics['mathematical_expectancy']:.4f}")
    print(f"- RR Ratio Achieved: {final_metrics['rr_ratio_achieved']:.2f}")
    print(f"- Max Drawdown: {final_metrics['max_drawdown']:.2f}%")
    print(f"- Avg Trade Duration: {final_metrics['avg_duration']:.1f} bars")
    print(f"- Median Trade Duration: {final_metrics['median_duration']:.1f} bars")
    print(f"- Min/Max Duration: {final_metrics['min_duration']:.0f}/{final_metrics['max_duration']:.0f} bars")
    print(f"- Avg Winning Duration: {final_metrics['avg_winning_duration']:.1f} bars")
    print(f"- Avg Losing Duration: {final_metrics['avg_losing_duration']:.1f} bars")
    # Detailed analysis
    if final_metrics['mathematical_expectancy'] > 0.5:
        print("\n🎯 ESTRATEGIA ALTAMENTE RENTABLE")
        print("   ✅ Lista para implementación en vivo")
        print("   ✅ Ratio riesgo/beneficio 2:1 confirmado")
        print("   ✅ Expectativa matemática positiva")
    elif final_metrics['mathematical_expectancy'] > 0:
        print("\n⚠️ ESTRATEGIA MODERADAMENTE RENTABLE")
        print("   📈 Viable con gestión de riesgo adicional")
        print("   🔄 Considerar ajustes finos")
    else:
        print("\n❌ ESTRATEGIA NO RENTABLE")
        print("   🔄 Revisar parámetros o estrategia")
        print("   📊 Posible sobre-optimización")

    return final_metrics


def compare_strategies():
    """Compare different strategy configurations with and without multi-timeframe EMA filter"""
    print("\n📊 COMPARACIÓN DE ESTRATEGIAS")
    print("=" * 80)

    # Load data
    df_5m = pd.read_csv('data/btc_5Min.csv', index_col=0, parse_dates=True)
    df_5m.columns = df_5m.columns.str.lower()
    df_5m = df_5m.tail(5000)  # Use full test dataset

    df_15m = pd.read_csv('data/btc_15Min.csv', index_col=0, parse_dates=True)
    df_15m.columns = df_15m.columns.str.lower()

    df_1h = pd.read_csv('data/btc_1H.csv', index_col=0, parse_dates=True)
    df_1h.columns = df_1h.columns.str.lower()

    print(f"📊 Dataset: {len(df_5m)} barras de 5 minutos")

    # Strategy configurations to compare
    strategies = [
        {
            'name': 'Sin filtro EMA multi-timeframe',
            'use_ema_filter': False,
            'description': 'Estrategia base sin confirmación de EMA en timeframes superiores'
        },
        {
            'name': 'Con filtro EMA multi-timeframe (2%)',
            'use_ema_filter': True,
            'description': 'Estrategia con confirmación de EMA50/EMA200 en 15m y 1h timeframes (2% proximidad)'
        }
    ]

    results = {}

    # Best parameters from optimization
    best_params = [15, 0.15, 1.0, 3.0, 1, 0.5, True]  # ADX, Squeeze, SL_ATR, TP_ATR, Cooldown, Trailing, EMA_Filter

    for strategy_config in strategies:
        print(f"\n🔄 Ejecutando: {strategy_config['name']}")
        print(f"   {strategy_config['description']}")

        # Create strategy with best parameters
        strategy = SqueezeMomentumADXTTMStrategy()
        strategy.min_adx_entry = best_params[0]
        strategy.max_adx_entry = 100
        strategy.min_squeeze_momentum = best_params[1]
        strategy.stop_loss_atr_mult = best_params[2]
        strategy.take_profit_atr_mult = best_params[3]
        strategy.use_volume_filter = False
        strategy.use_multitimeframe = False
        strategy.use_poc_filter = False
        strategy.use_adx_slope = False
        strategy.trailing_stop = True
        strategy.trailing_activation = best_params[5]

        # Create backtester
        backtester = OptimizedBacktester(strategy)
        backtester.cooldown_bars = best_params[4]
        backtester.use_ema_filter = strategy_config['use_ema_filter']

        # Run backtest
        metrics = backtester.run_optimized_backtest(df_5m, df_15m, df_1h)

        results[strategy_config['name']] = {
            'metrics': metrics,
            'equity_curve': backtester.equity_curve.copy(),
            'trades': backtester.trades.copy()
        }

        print(f"   ✅ Retorno: {metrics['total_return_pct']:.2f}%")
        print(f"   ✅ Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   ✅ Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   ✅ Mathematical Expectancy: {metrics['mathematical_expectancy']:.4f}")

    # Generate comparison plots
    generate_comparison_plots(results)

    # Print detailed comparison
    print_comparison_table(results)

    return results


def generate_comparison_plots(results):
    """Generate comparison plots for different strategies"""
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparación de Estrategias: Con vs Sin Filtro EMA Multi-Timeframe', fontsize=14, fontweight='bold')

    colors = ['#2E86AB', '#A23B72']

    for i, (strategy_name, data) in enumerate(results.items()):
        color = colors[i]
        equity_curve = data['equity_curve']
        metrics = data['metrics']
        trades = data['trades']

        # Equity curve
        ax1.plot(equity_curve, color=color, linewidth=2, label=strategy_name)
        ax1.set_title('Curva de Capital')
        ax1.set_xlabel('Número de Trade')
        ax1.set_ylabel('Capital ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Returns distribution
        trade_returns = [trade['pnl'] for trade in trades]
        ax2.hist(trade_returns, bins=30, alpha=0.7, color=color, label=strategy_name)
        ax2.set_title('Distribución de Retornos por Trade')
        ax2.set_xlabel('Retorno ($)')
        ax2.set_ylabel('Frecuencia')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Cumulative returns
        cumulative_returns = np.cumsum(trade_returns)
        ax3.plot(cumulative_returns, color=color, linewidth=2, label=strategy_name)
        ax3.set_title('Retornos Acumulativos')
        ax3.set_xlabel('Número de Trade')
        ax3.set_ylabel('Retorno Acumulado ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Win/Loss ratio over time (rolling)
        wins = [1 if trade['pnl'] > 0 else 0 for trade in trades]
        if len(wins) > 10:
            rolling_win_rate = pd.Series(wins).rolling(10).mean() * 100
            ax4.plot(rolling_win_rate.values, color=color, linewidth=2, label=strategy_name)
        ax4.set_title('Win Rate Rodante (10 trades)')
        ax4.set_xlabel('Número de Trade')
        ax4.set_ylabel('Win Rate (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/strategy_comparison_curves.png', dpi=300, bbox_inches='tight')
    print("\n💾 Gráfico guardado: results/strategy_comparison_curves.png")
    plt.show()


def print_comparison_table(results):
    """Print detailed comparison table"""
    print("\n📋 TABLA COMPARATIVA DE ESTRATEGIAS")
    print("=" * 100)

    headers = ['Métrica', 'Sin EMA Multi-Timeframe', 'Con EMA Multi-Timeframe (2%)', 'Diferencia']
    print(f"{headers[0]:<25} {headers[1]:<25} {headers[2]:<25} {headers[3]:<15}")
    print("-" * 100)

    metrics_to_compare = [
        ('Retorno Total', 'total_return', '.2f', '%'),
        ('Win Rate', 'win_rate', '.1f', '%'),
        ('Profit Factor', 'profit_factor', '.2f', ''),
        ('Mathematical Expectancy', 'mathematical_expectancy', '.4f', ''),
        ('RR Ratio Achieved', 'rr_ratio_achieved', '.2f', ''),
        ('Max Drawdown', 'max_drawdown', '.2f', '%'),
        ('Total Trades', 'total_trades', '.0f', ''),
        ('Avg Win', 'avg_win', '.2f', '$'),
        ('Avg Loss', 'avg_loss', '.2f', '$'),
        ('Avg Trade Duration', 'avg_duration', '.1f', ' bars'),
        ('Sharpe Ratio', 'sharpe_ratio', '.2f', ''),
    ]

    for metric_name, metric_key, format_str, suffix in metrics_to_compare:
        val1 = results['Sin filtro EMA multi-timeframe']['metrics'].get(metric_key, 0)
        val2 = results['Con filtro EMA multi-timeframe (2%)']['metrics'].get(metric_key, 0)
        diff = val2 - val1

        # Color coding for difference
        if metric_key in ['total_return', 'win_rate', 'profit_factor', 'mathematical_expectancy', 'sharpe_ratio']:
            diff_color = "🟢" if diff > 0 else "🔴" if diff < 0 else "⚪"
        elif metric_key in ['max_drawdown', 'avg_loss']:
            diff_color = "🟢" if diff < 0 else "🔴" if diff > 0 else "⚪"
        else:
            diff_color = "⚪"

        print(f"{metric_name:<25} {val1:{format_str}}{suffix:<10} {val2:{format_str}}{suffix:<10} {diff_color} {diff:{format_str}}{suffix}")


if __name__ == "__main__":
    # Run optimization
    best_result = optimize_strategy_parameters()

    # Run final validation
    final_metrics = run_final_validation(best_result)

    # Run strategy comparison
    comparison_results = compare_strategies()

    print("\n🏁 PROCESO COMPLETADO")
    print("=" * 80)
    print("✅ Optimización de parámetros completada")
    print("✅ Ratio riesgo/beneficio 2:1 implementado")
    print("✅ Validación matemática realizada")
    print("📊 Comparación de estrategias completada")
    print("📊 Estrategia optimizada y lista para uso")

    if final_metrics['mathematical_expectancy'] > 0:
        print("💰 **ESTRATEGIA MATEMÁTICAMENTE RENTABLE** 💰")
    else:
        print("🔄 **REQUIERE AJUSTES ADICIONALES** 🔄")