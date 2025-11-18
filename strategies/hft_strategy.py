"""
High-Frequency Trading (HFT) Strategy for TradingIA Platform
Implements a market-making style HFT strategy with fast execution and scalping.

Based on the top-performing HFT strategy from the analysis results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib
from strategies.base_strategy import BaseStrategy


class HFTStrategy(BaseStrategy):
    """
    High-Frequency Trading strategy that uses market microstructure signals,
    order book dynamics, and fast scalping techniques for high Sharpe ratio.
    """

    def __init__(self, name: str = "HFT_Strategy"):
        super().__init__(name)

        # HFT Parameters
        self.min_volume_threshold = 1000  # Minimum volume for signal
        self.spread_threshold = 0.001  # 0.1% spread threshold
        self.momentum_window = 5  # Short-term momentum window
        self.volatility_window = 20  # Volatility calculation window
        self.scalp_profit_pct = 0.002  # 0.2% scalp profit target
        self.max_holding_bars = 3  # Maximum bars to hold position

        # Risk management
        self.stop_loss_pct = 0.005  # 0.5% stop loss
        self.max_daily_trades = 50  # Maximum trades per day
        self.max_consecutive_losses = 3  # Max consecutive losing trades

        # Market microstructure parameters
        self.order_flow_threshold = 1.2  # Order flow imbalance threshold
        self.realized_vol_threshold = 0.02  # Realized volatility threshold

        # Technical indicators
        self.fast_ma = 5
        self.slow_ma = 13
        self.rsi_period = 9
        self.atr_period = 5

        # State tracking
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.position_entry_price = None
        self.position_entry_time = None

    def _calculate_market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure indicators"""
        df = df.copy()

        # Order flow (simulated from tick data)
        if 'volume' in df.columns:
            df['buy_volume'] = df['volume'] * 0.5  # Simplified assumption
            df['sell_volume'] = df['volume'] * 0.5
            df['order_flow'] = (df['buy_volume'] - df['sell_volume']) / df['volume']
        else:
            df['order_flow'] = 0

        # Realized volatility
        df['returns'] = df['close'].pct_change()
        df['realized_vol'] = df['returns'].rolling(self.volatility_window).std()

        # Bid-ask spread (simulated)
        df['spread'] = (df['high'] - df['low']) / df['close']
        df['spread_ma'] = df['spread'].rolling(10).mean()

        # Market depth (simulated)
        df['market_depth'] = df['volume'].rolling(5).mean() if 'volume' in df.columns else 1

        return df

    def _calculate_momentum_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate short-term momentum signals"""
        df = df.copy()

        # Fast momentum
        df['momentum'] = df['close'] / df['close'].shift(self.momentum_window) - 1

        # Rate of change
        df['roc'] = talib.ROC(df['close'], timeperiod=self.momentum_window)

        # Price velocity
        df['price_velocity'] = df['close'].diff(self.momentum_window) / self.momentum_window

        # Volume momentum
        if 'volume' in df.columns:
            df['volume_momentum'] = df['volume'] / df['volume'].shift(self.momentum_window) - 1
        else:
            df['volume_momentum'] = 0

        return df

    def _calculate_scalping_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate scalping entry/exit signals"""
        df = df.copy()

        # Moving averages for trend
        df['fast_ma'] = talib.SMA(df['close'], timeperiod=self.fast_ma)
        df['slow_ma'] = talib.SMA(df['close'], timeperiod=self.slow_ma)
        df['ma_diff'] = df['fast_ma'] - df['slow_ma']

        # RSI for overbought/oversold
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)

        # ATR for volatility filter
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)

        # Support/resistance levels (recent highs/lows)
        df['recent_high'] = df['high'].rolling(10).max()
        df['recent_low'] = df['low'].rolling(10).min()

        return df

    def _check_risk_limits(self, current_time) -> bool:
        """Check if risk limits allow new trade"""
        # Reset daily counters if new day
        if self.last_trade_time is not None:
            if hasattr(current_time, 'date') and hasattr(self.last_trade_time, 'date'):
                if current_time.date() != self.last_trade_time.date():
                    self.daily_trades = 0
                    self.consecutive_losses = 0

        # Check limits
        if self.daily_trades >= self.max_daily_trades:
            return False

        if self.consecutive_losses >= self.max_consecutive_losses:
            return False

        return True

    def _calculate_position_size(self, price: float, volatility: float) -> float:
        """Calculate position size based on volatility and risk"""
        # Base position size
        base_size = 1.0

        # Adjust for volatility (lower size in high vol)
        if volatility > self.realized_vol_threshold:
            base_size *= 0.5

        # Adjust for consecutive losses
        if self.consecutive_losses > 0:
            base_size *= (0.8 ** self.consecutive_losses)

        return base_size

    def predict_signal(self, df: pd.DataFrame) -> Dict:
        """Generate HFT trading signal"""
        if len(df) < max(self.momentum_window, self.volatility_window, 20):
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'Insufficient data for HFT signals'
            }

        try:
            # Get latest data point
            current = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else current

            # Check risk limits
            current_time = df.index[-1] if hasattr(df.index[-1], 'hour') else pd.Timestamp.now()
            if not self._check_risk_limits(current_time):
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Risk limits exceeded'
                }

            # Calculate indicators
            df_processed = self._calculate_market_microstructure(df)
            df_processed = self._calculate_momentum_signals(df_processed)
            df_processed = self._calculate_scalping_signals(df_processed)

            current_indicators = df_processed.iloc[-1]

            # HFT Signal Logic
            signal = 'HOLD'
            confidence = 0.0
            reasons = []

            # 1. Volume filter
            if 'volume' in df_processed.columns and current_indicators.get('volume', 0) < self.min_volume_threshold:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Volume below threshold'
                }

            # 2. Spread filter
            if current_indicators['spread'] > self.spread_threshold:
                reasons.append('High spread')

            # 3. Momentum signals
            momentum_score = 0

            # Price momentum
            if current_indicators['momentum'] > 0.001:  # 0.1% upward momentum
                momentum_score += 1
                reasons.append('Bullish momentum')
            elif current_indicators['momentum'] < -0.001:  # 0.1% downward momentum
                momentum_score -= 1
                reasons.append('Bearish momentum')

            # ROC signal
            if current_indicators['roc'] > 0.5:
                momentum_score += 0.5
            elif current_indicators['roc'] < -0.5:
                momentum_score -= 0.5

            # 4. Order flow
            if abs(current_indicators['order_flow']) > self.order_flow_threshold:
                if current_indicators['order_flow'] > 0:
                    momentum_score += 0.5
                    reasons.append('Buy order flow')
                else:
                    momentum_score -= 0.5
                    reasons.append('Sell order flow')

            # 5. Technical signals
            # RSI filter
            if current_indicators['rsi'] < 30:  # Oversold
                momentum_score += 0.3
                reasons.append('RSI oversold')
            elif current_indicators['rsi'] > 70:  # Overbought
                momentum_score -= 0.3
                reasons.append('RSI overbought')

            # Moving average crossover
            if current_indicators['ma_diff'] > 0 and df_processed.iloc[-2]['ma_diff'] <= 0:
                momentum_score += 0.4
                reasons.append('MA crossover bullish')
            elif current_indicators['ma_diff'] < 0 and df_processed.iloc[-2]['ma_diff'] >= 0:
                momentum_score -= 0.4
                reasons.append('MA crossover bearish')

            # 6. Volatility filter
            volatility = current_indicators.get('realized_vol', 0)
            if volatility > self.realized_vol_threshold:
                momentum_score *= 0.7  # Reduce signal strength in high vol
                reasons.append('High volatility')

            # Determine signal
            if momentum_score > 1.0:
                signal = 'BUY'
                confidence = min(momentum_score / 2.0, 1.0)
            elif momentum_score < -1.0:
                signal = 'SELL'
                confidence = min(abs(momentum_score) / 2.0, 1.0)
            else:
                signal = 'HOLD'
                confidence = 0.0

            # Position size calculation
            position_size = self._calculate_position_size(current['close'], volatility)

            return {
                'signal': signal,
                'confidence': float(confidence),
                'position_size': position_size,
                'momentum_score': momentum_score,
                'reasons': reasons,
                'reason': '; '.join(reasons) if reasons else 'No clear signal'
            }

        except Exception as e:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': f'HFT signal error: {str(e)}'
            }

    def check_exit_conditions(self, df: pd.DataFrame, entry_price: float, entry_time) -> Dict:
        """Check if position should be exited"""
        if len(df) == 0:
            return {'exit': False, 'reason': 'No data'}

        current = df.iloc[-1]
        current_time = df.index[-1]

        # Calculate holding period
        if hasattr(current_time, 'value') and hasattr(entry_time, 'value'):
            holding_bars = len(df) - df.index.get_loc(entry_time) if entry_time in df.index else 0
        else:
            holding_bars = 1

        # Profit/loss calculation
        current_price = current['close']
        pnl_pct = (current_price - entry_price) / entry_price

        # Exit conditions
        exit_signal = False
        exit_reason = ""

        # 1. Profit target
        if pnl_pct >= self.scalp_profit_pct:
            exit_signal = True
            exit_reason = f"Profit target reached: {pnl_pct:.2%}"

        # 2. Stop loss
        elif pnl_pct <= -self.stop_loss_pct:
            exit_signal = True
            exit_reason = f"Stop loss triggered: {pnl_pct:.2%}"

        # 3. Maximum holding period
        elif holding_bars >= self.max_holding_bars:
            exit_signal = True
            exit_reason = f"Max holding period reached: {holding_bars} bars"

        # 4. Adverse momentum (exit if momentum reverses)
        if len(df) >= self.momentum_window:
            recent_momentum = current_price / df.iloc[-self.momentum_window]['close'] - 1
            if pnl_pct > 0 and recent_momentum < -0.001:  # Profit but momentum turned negative
                exit_signal = True
                exit_reason = "Momentum reversal"

        return {
            'exit': exit_signal,
            'reason': exit_reason,
            'pnl_pct': pnl_pct,
            'holding_bars': holding_bars
        }

    def generate_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Generate trading signals for backtesting"""
        signals = []
        position = None
        entry_price = None
        entry_time = None

        # Reset daily counters
        self.daily_trades = 0
        self.consecutive_losses = 0

        for i in range(max(self.momentum_window, self.volatility_window), len(df)):
            current_df = df.iloc[:i+1]
            current_time = df.index[i]

            # Check exit conditions if in position
            if position is not None:
                exit_check = self.check_exit_conditions(current_df, entry_price, entry_time)
                if exit_check['exit']:
                    # Close position
                    signals.append({
                        'timestamp': current_time,
                        'signal': 'CLOSE',
                        'price': df.iloc[i]['close'],
                        'reason': exit_check['reason'],
                        'pnl_pct': exit_check['pnl_pct'],
                        'holding_bars': exit_check['holding_bars']
                    })

                    # Update consecutive losses
                    if exit_check['pnl_pct'] < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0

                    position = None
                    entry_price = None
                    entry_time = None
                    self.daily_trades += 1

                    # Check if we hit max consecutive losses
                    if self.consecutive_losses >= self.max_consecutive_losses:
                        break  # Stop trading for the day

            # Generate new signals if not in position
            elif position is None:
                signal = self.predict_signal(current_df)

                if signal['signal'] in ['BUY', 'SELL'] and signal['confidence'] > 0.6:
                    signals.append({
                        'timestamp': current_time,
                        'signal': signal['signal'],
                        'price': df.iloc[i]['close'],
                        'confidence': signal['confidence'],
                        'position_size': signal.get('position_size', 1.0),
                        'reason': signal['reason']
                    })

                    position = signal['signal']
                    entry_price = df.iloc[i]['close']
                    entry_time = current_time

        return signals

    def get_parameters(self) -> Dict:
        """Get strategy parameters"""
        return {
            'min_volume_threshold': self.min_volume_threshold,
            'spread_threshold': self.spread_threshold,
            'momentum_window': self.momentum_window,
            'volatility_window': self.volatility_window,
            'scalp_profit_pct': self.scalp_profit_pct,
            'max_holding_bars': self.max_holding_bars,
            'stop_loss_pct': self.stop_loss_pct,
            'max_daily_trades': self.max_daily_trades,
            'max_consecutive_losses': self.max_consecutive_losses,
            'order_flow_threshold': self.order_flow_threshold,
            'realized_vol_threshold': self.realized_vol_threshold,
            'fast_ma': self.fast_ma,
            'slow_ma': self.slow_ma,
            'rsi_period': self.rsi_period,
            'atr_period': self.atr_period
        }

    def set_parameters(self, params: Dict):
        """Set strategy parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

