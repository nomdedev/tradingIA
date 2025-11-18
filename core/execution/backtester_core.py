"""
Advanced Backtester Core for Multi-Timeframe Trading Strategies.

This module provides a comprehensive backtesting framework with:
- Walk-forward optimization
- Monte Carlo simulation
- Risk management and metrics calculation
- VectorBT integration for portfolio simulation
- FASE 1: Realistic execution modeling (market impact, order types, latency)
"""

import pandas as pd
import numpy as np
import logging
import traceback
import threading
import sys
import os
from skopt import gp_minimize
from skopt.space import Real, Integer
import vectorbt as vbt
from typing import Dict, List, Optional, Union, Tuple

# Add src to path for realistic execution components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from src.execution.market_impact import MarketImpactModel, VolumeProfileAnalyzer
    from src.execution.order_manager import OrderManager, OrderType, OrderSide
    from src.execution.latency_model import LatencyProfile
    from src.risk.kelly_sizer import KellyPositionSizer
    REALISTIC_EXECUTION_AVAILABLE = True
except ImportError as e:
    REALISTIC_EXECUTION_AVAILABLE = False
    logging.warning(f"Realistic execution components not available: {e}")


class BacktesterCore:
    """
    Advanced backtesting engine for trading strategies.

    Provides comprehensive backtesting capabilities including:
    - Simple backtesting with metrics calculation
    - Walk-forward optimization
    - Monte Carlo simulation for robustness testing
    - Risk management and realistic cost modeling
    - VectorBT integration for portfolio simulation
    """

    def __init__(self, initial_capital=10000, commission=0.001, slippage_pct=0.001,
                 enable_realistic_execution=False, latency_profile='retail_average',
                 enable_kelly_position_sizing=False, kelly_fraction=0.5, max_position_pct=0.10):
        """
        Initialize backtester with optional realistic execution modeling.

        Args:
            initial_capital: Starting capital
            commission: Base commission rate (used if realistic execution disabled)
            slippage_pct: Base slippage (used if realistic execution disabled)
            enable_realistic_execution: Enable FASE 1 realistic execution (market impact, latency)
            latency_profile: Latency profile ('co-located', 'retail_average', 'mobile', etc.)
            enable_kelly_position_sizing: Enable FASE 2 Kelly Criterion position sizing
            kelly_fraction: Kelly fraction (0.5 = half Kelly, conservative)
            max_position_pct: Maximum position size as % of capital
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.enable_realistic_execution = enable_realistic_execution
        self.enable_kelly_position_sizing = enable_kelly_position_sizing
        self.logger = logging.getLogger(__name__)
        self._cancel_flag = threading.Event()
        self._current_thread = None
        
        # Trade history tracking for Kelly statistics
        self.trade_history = pd.DataFrame(columns=[
            'timestamp', 'side', 'entry_price', 'exit_price', 'size', 
            'pnl', 'pnl_pct', 'hold_time', 'mae', 'mfe'
        ])
        self.current_capital = initial_capital  # Track capital dynamically

        # FASE 2: Initialize Kelly position sizer
        if enable_kelly_position_sizing and REALISTIC_EXECUTION_AVAILABLE:
            self.logger.info("🎯 Kelly position sizing enabled (FASE 2)")
            self.kelly_sizer = KellyPositionSizer(
                kelly_fraction=kelly_fraction,
                max_position_pct=max_position_pct
            )
            self.logger.info(f"   Kelly fraction: {kelly_fraction}")
            self.logger.info(f"   Max position: {max_position_pct*100}%")
        elif enable_kelly_position_sizing and not REALISTIC_EXECUTION_AVAILABLE:
            self.logger.warning("⚠️ Kelly position sizing requested but components not available")
            self.enable_kelly_position_sizing = False

        # FASE 1: Initialize realistic execution components
        if enable_realistic_execution and REALISTIC_EXECUTION_AVAILABLE:
            self.logger.info("🚀 Realistic execution enabled (FASE 1)")
            self.market_impact_model = MarketImpactModel()
            self.volume_analyzer = VolumeProfileAnalyzer()
            self.latency_model = LatencyProfile.get_profile(latency_profile)
            self.latency_profile_name = latency_profile
            self.logger.info(f"   Latency profile: {latency_profile}")
        elif enable_realistic_execution and not REALISTIC_EXECUTION_AVAILABLE:
            self.logger.warning("⚠️ Realistic execution requested but components not available")
            self.logger.warning("   Falling back to simple execution model")
            self.enable_realistic_execution = False
        else:
            self.logger.info("Simple execution model (legacy)")

    def _record_trade(self, timestamp, side: str, entry_price: float, 
                     exit_price: float, size: float, hold_time: float = 0.0,
                     mae: float = 0.0, mfe: float = 0.0):
        """
        Record a completed trade in trade history.
        
        Args:
            timestamp: Trade timestamp
            side: 'buy' or 'sell'
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            hold_time: Duration of trade in hours
            mae: Maximum Adverse Excursion (% from entry)
            mfe: Maximum Favorable Excursion (% from entry)
        """
        pnl = (exit_price - entry_price) * size if side == 'buy' else (entry_price - exit_price) * size
        pnl_pct = (pnl / (entry_price * size)) if entry_price > 0 else 0.0
        
        trade_record = {
            'timestamp': timestamp,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hold_time': hold_time,
            'mae': mae,  # Maximum Adverse Excursion (%)
            'mfe': mfe   # Maximum Favorable Excursion (%)
        }
        
        # Append to history
        self.trade_history.loc[len(self.trade_history)] = trade_record
        
        # Update current capital
        self._update_capital(pnl)
        
        self.logger.debug(f"Trade recorded: {side} @ {entry_price:.2f}→{exit_price:.2f}, "
                         f"PnL: ${pnl:.2f} ({pnl_pct:.2%}), MAE: {mae:.2%}, MFE: {mfe:.2%}")
    
    def _update_capital(self, pnl: float):
        """
        Update current capital after trade completion.
        
        Args:
            pnl: Profit/loss from the trade
        """
        self.current_capital += pnl
        
        # Ensure capital doesn't go negative (risk of ruin protection)
        if self.current_capital <= 0:
            self.logger.error(f"⚠️ RISK OF RUIN: Capital depleted (${self.current_capital:.2f})")
            self.current_capital = max(0.01, self.current_capital)  # Keep minimum viable
    
    def _get_strategy_statistics(self, lookback: int = 50) -> Tuple[float, float]:
        """
        Calculate win rate and win/loss ratio from recent trade history.
        
        Args:
            lookback: Number of recent trades to analyze
            
        Returns:
            Tuple of (win_rate, win_loss_ratio)
        """
        if len(self.trade_history) < 20:
            # Not enough history, use conservative defaults
            self.logger.debug("Insufficient trade history, using conservative defaults")
            return 0.50, 1.2  # Breakeven with low expectancy
        
        # Analyze recent trades
        recent_trades = self.trade_history.tail(lookback)
        wins = recent_trades[recent_trades['pnl'] > 0]
        losses = recent_trades[recent_trades['pnl'] < 0]
        
        if len(losses) == 0:
            # All wins (unlikely but handle gracefully)
            return 1.0, 2.0  # Very conservative despite 100% win rate
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean())
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        self.logger.debug(f"Strategy stats from last {len(recent_trades)} trades: "
                         f"WR={win_rate:.2%}, W/L={win_loss_ratio:.2f}")
        
        return win_rate, win_loss_ratio
    
    def _record_trade(self, timestamp, side: str, entry_price: float, 
                     exit_price: float, size: float, hold_time: float = 0.0,
                     mae: float = 0.0, mfe: float = 0.0):
        """
        Record a completed trade in trade history.
        
        Args:
            timestamp: Trade timestamp
            side: 'buy' or 'sell'
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            hold_time: Duration of trade in hours
            mae: Maximum Adverse Excursion (% from entry)
            mfe: Maximum Favorable Excursion (% from entry)
        """
        pnl = (exit_price - entry_price) * size if side == 'buy' else (entry_price - exit_price) * size
        pnl_pct = (pnl / (entry_price * size)) if entry_price > 0 else 0.0
        
        trade_record = {
            'timestamp': timestamp,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hold_time': hold_time,
            'mae': mae,  # Maximum Adverse Excursion (%)
            'mfe': mfe   # Maximum Favorable Excursion (%)
        }
        
        # Append to history
        self.trade_history.loc[len(self.trade_history)] = trade_record
        
        # Update current capital
        self._update_capital(pnl)
        
        self.logger.debug(f"Trade recorded: {side} @ {entry_price:.2f}→{exit_price:.2f}, "
                         f"PnL: ${pnl:.2f} ({pnl_pct:.2%}), MAE: {mae:.2%}, MFE: {mfe:.2%}")
    
    def _update_capital(self, pnl: float):
        """
        Update current capital after trade completion.
        
        Args:
            pnl: Profit/loss from the trade
        """
        self.current_capital += pnl
        
        # Ensure capital doesn't go negative (risk of ruin protection)
        if self.current_capital <= 0:
            self.logger.error(f"⚠️ RISK OF RUIN: Capital depleted (${self.current_capital:.2f})")
            self.current_capital = max(0.01, self.current_capital)  # Keep minimum viable
    
    def _process_and_record_trades(self, portfolio, df_5m):
        """
        Process trades from VectorBT portfolio and record them in trade history.
        Updates current_capital based on realized PnL.
        
        Args:
            portfolio: VectorBT portfolio object
            df_5m: DataFrame with 5min data for timestamps
        """
        if not hasattr(portfolio, 'trades') or portfolio.trades.count() == 0:
            self.logger.debug("No trades to record")
            return
        
        try:
            # Get trades as records array (numpy structured array)
            trades_arr = portfolio.trades.records
            
            for i in range(len(trades_arr)):
                trade = trades_arr[i]
                
                # Extract trade information using numpy array indexing
                entry_idx = int(trade[2])  # entry_idx is 3rd field
                exit_idx = int(trade[3])   # exit_idx is 4th field
                
                if entry_idx >= len(df_5m) or exit_idx >= len(df_5m):
                    continue
                
                timestamp = df_5m.index[exit_idx]
                entry_price = float(trade[5])  # entry_price field
                exit_price = float(trade[6])   # exit_price field
                size = float(trade[4])         # size field
                
                # Calculate PnL to determine side
                pnl = float(trade[7]) if len(trade) > 7 else (exit_price - entry_price) * size
                side = 'buy' if pnl >= 0 else 'sell'
                
                # Calculate hold time
                entry_time = df_5m.index[entry_idx]
                exit_time = df_5m.index[exit_idx]
                hold_time = (exit_time - entry_time).total_seconds() / 3600.0
                
                # Calculate MAE and MFE during the trade
                price_series = df_5m['high'].iloc[entry_idx:exit_idx+1] if side == 'buy' else df_5m['low'].iloc[entry_idx:exit_idx+1]
                max_price = price_series.max()
                min_price = price_series.min()
                
                if side == 'buy':
                    # Long trade: MAE is how far price went below entry, MFE above entry
                    mae = (entry_price - min_price) / entry_price if min_price < entry_price else 0.0
                    mfe = (max_price - entry_price) / entry_price if max_price > entry_price else 0.0
                else:
                    # Short trade: MAE is how far price went above entry, MFE below entry
                    mae = (max_price - entry_price) / entry_price if max_price > entry_price else 0.0
                    mfe = (entry_price - min_price) / entry_price if min_price < entry_price else 0.0
                
                # Record trade with MAE/MFE
                self._record_trade(
                    timestamp=timestamp,
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=size,
                    hold_time=hold_time,
                    mae=mae,
                    mfe=mfe
                )
            
            self.logger.info(f"Recorded {len(trades_arr)} trades. Current capital: ${self.current_capital:.2f}")
            
        except Exception as e:
            self.logger.warning(f"Could not record trades: {e}. Continuing without trade history update.")
            # Not critical - continue without recording
    
    def _calculate_position_size(self, capital: float, win_rate: float = None,
                                win_loss_ratio: float = None, current_volatility: float = 0.0,
                                market_impact_pct: float = 0.0) -> float:
        """
        Calculate dynamic position size using Kelly Criterion or fallback to simple method.

        Args:
            capital: Current available capital
            win_rate: Strategy win rate (0.0-1.0) - if None, calculate from history
            win_loss_ratio: Average win/loss ratio - if None, calculate from history
            current_volatility: Current market volatility (0.0-1.0)
            market_impact_pct: Estimated market impact cost

        Returns:
            Position size in dollars
        """
        if self.enable_kelly_position_sizing and hasattr(self, 'kelly_sizer'):
            # Calculate real statistics from trade history if not provided
            if win_rate is None or win_loss_ratio is None:
                win_rate, win_loss_ratio = self._get_strategy_statistics()

            try:
                sizing_result = self.kelly_sizer.calculate_position_size(
                    capital=capital,
                    win_rate=win_rate,
                    win_loss_ratio=win_loss_ratio,
                    current_volatility=current_volatility,
                    market_impact_pct=market_impact_pct
                )
                position_size = sizing_result['position_size']

                # Log Kelly sizing details
                self.logger.debug(f"Kelly position sizing: ${position_size:.2f} "
                                f"({sizing_result['position_pct']:.1%} of capital)")

                return position_size

            except Exception as e:
                self.logger.error(f"Error in Kelly position sizing: {e}")
                return capital * 0.01  # Fallback

        else:
            # Simple position sizing: 1% of capital
            return capital * 0.01
    
    def _calculate_order_size_for_execution(self, base_price: float, 
                                           current_capital: float,
                                           volatility_val: float) -> float:
        """
        Helper method to calculate order size for realistic execution.
        Eliminates code duplication between entries and exits.
        
        Args:
            base_price: Current market price
            current_capital: Available capital (dynamic)
            volatility_val: Current market volatility
            
        Returns:
            Order size in base currency units
        """
        if self.enable_kelly_position_sizing:
            # Use Kelly sizing with real statistics or defaults
            position_size_dollars = self._calculate_position_size(
                capital=current_capital,  # Use dynamic capital
                win_rate=None,  # Will calculate from history
                win_loss_ratio=None,  # Will calculate from history
                current_volatility=volatility_val,
                market_impact_pct=0.001  # 0.1% market impact estimate
            )
            return position_size_dollars / base_price
        else:
            # Simple position sizing: 1% of current capital
            return (current_capital * 0.01) / base_price

    def cancel_backtest(self):
        """Cancel ongoing backtest operation"""
        self._cancel_flag.set()
        self.logger.info("Backtest cancellation requested")

    def _check_cancellation(self):
        """Check if cancellation has been requested"""
        if self._cancel_flag.is_set():
            raise InterruptedError("Backtest cancelled by user")

    def validate_data_sufficiency(self, df_multi_tf: Dict[str, pd.DataFrame], min_bars: int = 50):
        """Validate that datasets have sufficient data for backtesting"""
        for tf, df in df_multi_tf.items():
            if df is None or df.empty:
                raise ValueError(f"Empty dataset for timeframe {tf}")

            if len(df) < min_bars:
                raise ValueError(
                    f"Insufficient data for timeframe {tf}: {len(df)} bars < {min_bars} minimum")

        return True

    def cap_extreme_metrics(self, metrics: Dict) -> Dict:
        """Cap extreme metric values to prevent unrealistic results"""
        capped_metrics = metrics.copy()

        # Cap Sharpe ratio to [-10, 10]
        if 'sharpe' in capped_metrics:
            original_sharpe = capped_metrics['sharpe']
            capped_metrics['sharpe'] = max(-10, min(10, capped_metrics['sharpe']))
            if original_sharpe != capped_metrics['sharpe']:
                self.logger.warning(f"Sharpe ratio capped from {original_sharpe} to "
                                    f"{capped_metrics['sharpe']}")

        # Cap Sortino ratio to [-10, 10]
        if 'sortino' in capped_metrics:
            original_sortino = capped_metrics['sortino']
            capped_metrics['sortino'] = max(-10, min(10, capped_metrics['sortino']))
            if original_sortino != capped_metrics['sortino']:
                self.logger.warning(f"Sortino ratio capped from {original_sortino} to "
                                    f"{capped_metrics['sortino']}")

        # Cap drawdown to [0, 1] (0-100%)
        if 'max_dd' in capped_metrics:
            capped_metrics['max_dd'] = max(0, min(1, capped_metrics['max_dd']))

        # Cap profit factor to [0, 100]
        if 'profit_factor' in capped_metrics:
            if capped_metrics['profit_factor'] == float('inf'):
                capped_metrics['profit_factor'] = 100
            else:
                capped_metrics['profit_factor'] = min(100, capped_metrics['profit_factor'])

        return capped_metrics

    def run_simple_backtest(self,
                            df_multi_tf: Dict[str,
                                              pd.DataFrame],
                            strategy_class,
                            strategy_params: Dict) -> Dict:
        try:
            # Reset cancellation flag
            self._cancel_flag.clear()

            # Validate data sufficiency
            self.validate_data_sufficiency(df_multi_tf)

            # Check for cancellation
            self._check_cancellation()

            # Extract 5min data for backtesting
            df_5m = df_multi_tf['5min'].copy()
            
            # Normalize column names to lowercase
            df_5m.columns = df_5m.columns.str.lower()

            # Initialize strategy
            strategy = strategy_class(**strategy_params)

            # Check for cancellation
            self._check_cancellation()

            # Generate signals
            signals = strategy.generate_signals(df_multi_tf)

            # Check for cancellation
            self._check_cancellation()

            # Run backtest with realistic execution if enabled
            if self.enable_realistic_execution:
                # Track execution costs
                total_market_impact = 0.0
                total_latency_cost = 0.0
                
                # Calculate average volume for market impact
                avg_volume = self.volume_analyzer.calculate_average_volume(
                    df_5m, lookback_periods=20
                )
                
                # Calculate volatility (ATR as % of price)
                if 'atr' in df_5m.columns or 'ATR' in df_5m.columns:
                    atr_col = 'atr' if 'atr' in df_5m.columns else 'ATR'
                    volatility = df_5m[atr_col] / df_5m['close']
                else:
                    # Fallback: rolling std
                    volatility = df_5m['close'].pct_change().rolling(20).std()
                
                # Adjust entry/exit prices for market impact
                adjusted_entries = signals['entries'].copy()
                adjusted_exits = signals['exits'].copy()
                entry_prices = df_5m['close'].copy()
                exit_prices = df_5m['close'].copy()
                
                # Apply realistic execution to entry signals
                entry_indices = signals['entries'][signals['entries']].index
                for idx in entry_indices:
                    if idx not in df_5m.index:
                        continue
                    loc = df_5m.index.get_loc(idx)
                    if loc >= len(avg_volume):
                        continue
                        
                    base_price = df_5m.loc[idx, 'close']
                    vol = avg_volume.iloc[loc] if loc < len(avg_volume) else avg_volume.iloc[-1]
                    volatility_val = volatility.iloc[loc] if loc < len(volatility) else 0.02
                    
                    # Calculate order size using helper (dynamic capital + real statistics)
                    order_size = self._calculate_order_size_for_execution(
                        base_price=base_price,
                        current_capital=self.current_capital,
                        volatility_val=volatility_val
                    )
                    
                    exec_data = self._calculate_realistic_execution_price(
                        base_price=base_price,
                        order_size=order_size,
                        avg_volume=vol,
                        volatility=volatility_val,
                        side='buy',
                        timestamp=idx
                    )
                    
                    entry_prices.loc[idx] = exec_data['execution_price']
                    
                    # Track costs
                    impact_cost = abs(exec_data['execution_price'] - base_price) * order_size
                    total_market_impact += impact_cost
                    
                    if 'latency_cost' in exec_data:
                        total_latency_cost += exec_data['latency_cost']
                
                # Similar for exits
                exit_indices = signals['exits'][signals['exits']].index
                for idx in exit_indices:
                    if idx not in df_5m.index:
                        continue
                    loc = df_5m.index.get_loc(idx)
                    if loc >= len(avg_volume):
                        continue
                        
                    base_price = df_5m.loc[idx, 'close']
                    vol = avg_volume.iloc[loc] if loc < len(avg_volume) else avg_volume.iloc[-1]
                    volatility_val = volatility.iloc[loc] if loc < len(volatility) else 0.02
                    
                    # Calculate order size using helper (dynamic capital + real statistics)
                    order_size = self._calculate_order_size_for_execution(
                        base_price=base_price,
                        current_capital=self.current_capital,
                        volatility_val=volatility_val
                    )
                    
                    exec_data = self._calculate_realistic_execution_price(
                        base_price=base_price,
                        order_size=order_size,
                        avg_volume=vol,
                        volatility=volatility_val,
                        side='sell',
                        timestamp=idx
                    )
                    
                    exit_prices.loc[idx] = exec_data['execution_price']
                    
                    # Track costs
                    impact_cost = abs(exec_data['execution_price'] - base_price) * order_size
                    total_market_impact += impact_cost
                    
                    if 'latency_cost' in exec_data:
                        total_latency_cost += exec_data['latency_cost']
                
                # Run portfolio with adjusted prices
                portfolio = vbt.Portfolio.from_signals(
                    close=df_5m['close'],
                    entries=adjusted_entries,
                    exits=adjusted_exits,
                    price=entry_prices,  # Use realistic entry prices
                    init_cash=self.initial_capital,
                    fees=0.0001,  # Minimal fees (impact already included)
                    slippage=0.0  # Slippage already included in impact
                )
                
                self.logger.info("✓ Backtest with realistic execution complete")
            else:
                # Simple execution (legacy)
                portfolio = vbt.Portfolio.from_signals(
                    close=df_5m['close'],
                    entries=signals['entries'],
                    exits=signals['exits'],
                    price=df_5m['close'],
                    init_cash=self.initial_capital,
                    fees=self.commission,
                    slippage=self.slippage_pct
                )

            # Check if portfolio was created successfully
            if portfolio is None or not hasattr(portfolio, 'trades'):
                raise ValueError("Portfolio creation failed")
            
            # Calculate metrics
            metrics = self.calculate_metrics(portfolio.returns(), portfolio.trades.records)

            # Cap extreme metrics
            metrics = self.cap_extreme_metrics(metrics)
            
            # Process and record trades for Kelly statistics (CRITICAL)
            if self.enable_kelly_position_sizing:
                self._process_and_record_trades(portfolio, df_5m)
                self.logger.info(f"📊 Kelly Statistics updated: {len(self.trade_history)} trades recorded")

            # Format trades
            trades = self._format_trades(portfolio.trades.records, df_5m.index)

            # Calculate realistic costs
            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df = self.calculate_realistic_costs(trades_df)

            # Get strategy parameters
            strategy_parameters = strategy.get_parameters()
            
            # Build result dictionary
            result = {
                'metrics': metrics,
                'trades': trades,
                'equity_curve': portfolio.value().tolist(),
                'signals': signals[['signals']].to_dict('records') if hasattr(
                    signals,
                    'to_dict') and 'signals' in signals.columns else [],
                'strategy_parameters': strategy_parameters,
                'final_capital': self.current_capital,
                'capital_growth': ((self.current_capital / self.initial_capital) - 1.0) * 100
            }
            
            # Add Kelly position sizing info if enabled
            if self.enable_kelly_position_sizing and len(self.trade_history) > 0:
                win_rate, wl_ratio = self._get_strategy_statistics()
                result['kelly_info'] = {
                    'enabled': True,
                    'trades_recorded': len(self.trade_history),
                    'win_rate': win_rate,
                    'win_loss_ratio': wl_ratio,
                    'kelly_fraction': self.kelly_sizer.kelly_fraction,
                    'max_position_pct': self.kelly_sizer.max_position_pct
                }
            
            # Add Kelly position sizing info if enabled
            if self.enable_kelly_position_sizing and len(self.trade_history) > 0:
                win_rate, wl_ratio = self._get_strategy_statistics()
                result['kelly_info'] = {
                    'enabled': True,
                    'trades_recorded': len(self.trade_history),
                    'win_rate': win_rate,
                    'win_loss_ratio': wl_ratio,
                    'kelly_fraction': self.kelly_sizer.kelly_fraction,
                    'max_position_pct': self.kelly_sizer.max_position_pct
                }
            
            # Add execution costs if realistic execution was used
            if self.enable_realistic_execution:
                result['execution_costs'] = {
                    'total_market_impact': total_market_impact,
                    'total_latency_cost': total_latency_cost,
                    'total_execution_cost': total_market_impact + total_latency_cost,
                    'num_trades': len(entry_indices) + len(exit_indices),
                    'avg_cost_per_trade': (total_market_impact + total_latency_cost) / max(1, len(entry_indices) + len(exit_indices)),
                    'latency_profile': self.latency_profile_name
                }

            return result

        except InterruptedError:
            self.logger.info("Backtest cancelled")
            return {'error': 'Backtest cancelled by user'}
        except Exception as e:
            error_msg = f"Error in simple backtest: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def run_backtest(self,
                     strategy_class,
                     df_multi_tf: Union[Dict[str,
                                             pd.DataFrame],
                                        pd.DataFrame],
                     strategy_params: Optional[Dict] = None) -> Dict:
        """Alias for run_simple_backtest with different parameter order for compatibility"""
        if strategy_params is None:
            strategy_params = {}

        # Convert DataFrame to dict format if needed
        if isinstance(df_multi_tf, pd.DataFrame):
            df_multi_tf = {'5min': df_multi_tf}

        return self.run_simple_backtest(df_multi_tf, strategy_class, strategy_params)

    def run_walk_forward(self,
                         df_multi_tf: Dict[str,
                                           pd.DataFrame],
                         strategy_class,
                         strategy_params: Dict,
                         n_periods: int = 8,
                         opt_method: str = 'bayes') -> Dict:
        try:
            df_5m = df_multi_tf['5min'].copy()
            total_bars = len(df_5m)
            period_size = total_bars // n_periods

            periods_results = []
            all_train_metrics = []
            all_test_metrics = []
            best_params = strategy_params  # Initialize with default params

            for i in range(n_periods):
                # Check for cancellation
                self._check_cancellation()

                train_start = i * period_size
                train_end = (i + 1) * period_size
                test_start = train_end
                test_end = min((i + 2) * period_size, total_bars)

                if test_end - test_start < 100:  # Minimum test size
                    break

                # Split data
                train_data = {tf: df.iloc[train_start:train_end] for tf, df in df_multi_tf.items()}
                test_data = {tf: df.iloc[test_start:test_end] for tf, df in df_multi_tf.items()}

                # Optimize on training data
                if opt_method == 'bayes':
                    # Skip optimization for now - use provided params
                    best_params = strategy_params
                else:
                    best_params = strategy_params  # Use provided params

                # Test on out-of-sample data
                train_result = self.run_backtest(strategy_class, train_data, best_params)
                test_result = self.run_backtest(strategy_class, test_data, best_params)

                if 'error' not in train_result and 'error' not in test_result:
                    train_sharpe = train_result['metrics']['sharpe']
                    test_sharpe = test_result['metrics']['sharpe']
                    if train_sharpe != 0:
                        degradation_pct = ((test_sharpe - train_sharpe) / abs(train_sharpe)) * 100
                    else:
                        degradation_pct = 0

                    period_result = {
                        'period': i + 1,
                        'train_metrics': train_result['metrics'],
                        'test_metrics': test_result['metrics'],
                        'degradation_pct': degradation_pct
                    }
                    periods_results.append(period_result)

                    all_train_metrics.append(train_result['metrics']['sharpe'])
                    all_test_metrics.append(test_result['metrics']['sharpe'])

            avg_degradation = np.mean([p['degradation_pct']
                                      for p in periods_results]) if periods_results else 0

            return {
                'period_results': periods_results,
                'avg_degradation': avg_degradation,
                'best_params': best_params
            }

        except InterruptedError:
            self.logger.info("Walk-forward analysis cancelled")
            return {'error': 'Walk-forward cancelled by user'}
        except Exception as e:
            error_msg = f"Error in walk-forward analysis: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def run_monte_carlo(self,
                        df_multi_tf: Dict[str,
                                          pd.DataFrame],
                        strategy_class,
                        strategy_params: Dict,
                        n_simulations: int = 500,
                        noise_pct: float = 10,
                        seed: int | None = None) -> Dict:
        try:
            # Set seed for reproducibility
            if seed is not None:
                np.random.seed(seed)
                self.logger.info(f"Monte Carlo using seed: {seed}")

            sharpe_results = []
            win_rate_results = []

            for i in range(n_simulations):
                # Check for cancellation
                self._check_cancellation()

                # Add noise to data
                noisy_data = {}
                for tf, df in df_multi_tf.items():
                    noise = np.random.normal(0, noise_pct / 100, len(df))
                    noisy_df = df.copy()
                    noisy_df['Close'] = df['Close'] * (1 + noise)
                    noisy_df['High'] = df['High'] * (1 + noise * 0.5)
                    noisy_df['Low'] = df['Low'] * (1 + noise * 0.5)
                    noisy_data[tf] = noisy_df

                # Run backtest
                result = self.run_simple_backtest(noisy_data, strategy_class, strategy_params)

                if 'error' not in result:
                    sharpe_results.append(result['metrics']['sharpe'])
                    win_rate_results.append(result['metrics']['win_rate'])

            if sharpe_results:
                sharpe_mean = np.mean(sharpe_results)
                sharpe_std = np.std(sharpe_results)
                robust = sharpe_std < 0.2  # Robust if std < 0.2

                # Create simulations list with individual results
                simulations = []
                for i, (sharpe, win_rate) in enumerate(zip(sharpe_results, win_rate_results)):
                    simulations.append({
                        'simulation_id': i,
                        'sharpe_ratio': sharpe,
                        'win_rate': win_rate
                    })

                return {
                    'simulations': simulations,
                    'summary_stats': {
                        'sharpe_mean': sharpe_mean,
                        'sharpe_std': sharpe_std,
                        'win_rate_mean': np.mean(win_rate_results),
                        'win_rate_std': np.std(win_rate_results),
                        'robust': robust
                    },
                    'sharpe_distribution': sharpe_results
                }
            else:
                return {'error': 'No valid Monte Carlo results'}

        except InterruptedError:
            self.logger.info("Monte Carlo cancelled")
            return {'error': 'Monte Carlo cancelled by user'}
            error_msg = f"Error in Monte Carlo simulation: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'traceback': traceback.format_exc()}

    def _calculate_realistic_execution_price(
        self,
        base_price: float,
        order_size: float,
        avg_volume: float,
        volatility: float,
        side: str,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Dict:
        """
        Calculate realistic execution price including market impact and latency.

        Args:
            base_price: Market price at signal time
            order_size: Order quantity
            avg_volume: Average trading volume
            volatility: Current volatility (ATR/price)
            side: 'buy' or 'sell'
            timestamp: Signal timestamp (for time-of-day effects)

        Returns:
            Dictionary with execution_price, impact_cost, latency_ms
        """
        if not self.enable_realistic_execution:
            # Simple model: just apply fixed slippage
            slippage_mult = 1 + self.slippage_pct if side == 'buy' else 1 - self.slippage_pct
            return {
                'execution_price': base_price * slippage_mult,
                'impact_cost': 0.0,
                'latency_ms': 0.0,
                'realistic': False
            }

        # Calculate market impact
        hour = timestamp.hour if timestamp is not None else None
        
        impact = self.market_impact_model.calculate_impact(
            order_size=order_size,
            price=base_price,
            avg_volume=avg_volume,
            volatility=volatility,
            bid_ask_spread=base_price * 0.001,  # Assume 0.1% spread
            time_of_day=hour,
            urgency=1.0
        )

        # Calculate execution price with impact
        execution_price = self.market_impact_model.calculate_execution_price(
            side=side,
            price=base_price,
            impact_pct=impact['total_impact_pct']
        )

        # Calculate latency
        vol_multiplier = 1.0 + (volatility - 0.02) / 0.02  # Scale around 2% baseline
        latency_ms = self.latency_model.calculate_total_latency(
            order_type='market',
            market_volatility=max(0.5, vol_multiplier),
            time_of_day=hour
        )

        return {
            'execution_price': execution_price,
            'impact_cost': impact['total_impact_dollars'],
            'latency_ms': latency_ms,
            'impact_pct': impact['total_impact_pct'],
            'realistic': True
        }

    def calculate_realistic_costs(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Commission: 0.1% round-trip
            trades_df['commission_cost'] = trades_df['pnl_pct'].abs() * 0.001

            # Slippage: base + vol_spike adjustment
            base_slippage = self.slippage_pct
            vol_spike_mult = 1.5  # Could be calculated from volatility
            trades_df['slippage_cost'] = trades_df['pnl_pct'].abs() * (base_slippage *
                                                                       vol_spike_mult)

            # Funding rate (if perpetual futures) - simplified
            funding_rate = 0.0001  # 0.01% per 8h, simplified to per trade
            trades_df['funding_cost'] = trades_df['pnl_pct'].abs() * funding_rate

            # Total cost
            trades_df['total_cost'] = (trades_df['commission_cost'] + trades_df['slippage_cost'] +
                                       trades_df['funding_cost'])

            return trades_df

        except Exception as e:
            self.logger.error(f"Error calculating realistic costs: {e}")
            return trades_df

    def calculate_metrics(self, returns: pd.Series, trades_records: pd.DataFrame) -> Dict:
        try:
            # Basic returns metrics
            cumulative_returns = (1 + returns).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1

            # Sharpe Ratio (annualized, assuming daily returns)
            risk_free_rate = 0.04 / 252  # 4% annual risk-free rate
            excess_returns = returns - risk_free_rate
            sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                     if excess_returns.std() > 0 else 0)

            # Calmar Ratio
            max_dd = self._calculate_max_drawdown(cumulative_returns)
            calmar = total_return / max_dd if max_dd > 0 else 0

            # Win Rate
            if not trades_records.empty:
                win_rate = (trades_records['pnl'] > 0).mean()
                num_trades = len(trades_records)
            else:
                win_rate = 0
                num_trades = 0

            # Information Ratio (vs buy-and-hold)
            bh_returns = returns  # Simplified, should be market returns
            ir = excess_returns.mean() / (returns - bh_returns).std() * \
                np.sqrt(252) if (returns - bh_returns).std() > 0 else 0

            # Ulcer Index
            ulcer = self._calculate_ulcer_index(cumulative_returns)

            # Sortino Ratio
            downside_returns = returns[returns < 0]
            sortino = excess_returns.mean() / downside_returns.std() * \
                np.sqrt(252) if len(downside_returns) > 0 else 0

            # Profit Factor
            gross_profit = trades_records[trades_records['pnl'] >
                                          0]['pnl'].sum() if not trades_records.empty else 0
            gross_loss = abs(trades_records[trades_records['pnl'] < 0]
                             ['pnl'].sum()) if not trades_records.empty else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # MAE/MFE Metrics from trade history (if available)
            avg_mae = self.trade_history['mae'].mean() if not self.trade_history.empty and 'mae' in self.trade_history.columns else 0.0
            avg_mfe = self.trade_history['mfe'].mean() if not self.trade_history.empty and 'mfe' in self.trade_history.columns else 0.0
            max_mae = self.trade_history['mae'].max() if not self.trade_history.empty and 'mae' in self.trade_history.columns else 0.0
            max_mfe = self.trade_history['mfe'].max() if not self.trade_history.empty and 'mfe' in self.trade_history.columns else 0.0

            return {
                'sharpe': round(sharpe, 3),
                'calmar': round(calmar, 3),
                'win_rate': round(win_rate, 3),
                'max_dd': round(max_dd, 3),
                'num_trades': num_trades,
                'ir': round(ir, 3),
                'ulcer': round(ulcer, 3),
                'sortino': round(sortino, 3),
                'profit_factor': round(profit_factor, 3),
                'total_return': round(total_return, 3),
                'avg_mae': round(avg_mae, 4),  # Average Maximum Adverse Excursion
                'avg_mfe': round(avg_mfe, 4),  # Average Maximum Favorable Excursion
                'max_mae': round(max_mae, 4),  # Maximum MAE across all trades
                'max_mfe': round(max_mfe, 4)   # Maximum MFE across all trades
            }

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {'error': str(e)}

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return abs(drawdown.min())

    def _calculate_ulcer_index(self, cumulative_returns: pd.Series) -> float:
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return np.sqrt((drawdown ** 2).mean())

    def _bayesian_optimize(self, strategy_class, train_data: Dict, param_space: Dict) -> Dict:
        try:
            # Define parameter space for optimization
            space = []
            param_names = []

            for param_name, param_config in param_space.items():
                if param_config.get('type') == 'int':
                    space.append(Integer(param_config['min'], param_config['max'], name=param_name))
                else:
                    space.append(Real(param_config['min'], param_config['max'], name=param_name))
                param_names.append(param_name)

            def objective(params):
                param_dict = dict(zip(param_names, params))
                result = self.run_simple_backtest(train_data, strategy_class, param_dict)
                if 'error' in result:
                    return 0  # Return neutral score for errors
                return -result['metrics']['sharpe']  # Minimize negative Sharpe

            # Run optimization
            res = gp_minimize(objective, space, n_calls=50, random_state=42)

            # Return best parameters
            best_params = dict(zip(param_names, res.x))
            return best_params

        except Exception as e:
            self.logger.error(f"Error in Bayesian optimization: {e}")
            return param_space  # Return original params on error

    def _format_trades(self, trades_records: pd.DataFrame,
                       df_index: pd.DatetimeIndex) -> List[Dict]:
        if trades_records.empty:
            return []

        trades = []
        for _, trade in trades_records.iterrows():
            # Map entry_idx to timestamp
            entry_idx = trade['entry_idx']
            entry_idx_int = int(entry_idx)  # Convert float to int
            entry_timestamp = df_index[entry_idx_int] if entry_idx_int < len(df_index) else None

            trades.append({
                'timestamp': entry_timestamp,
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'pnl_pct': trade['return'],  # VectorBT return is already in decimal format
                'score': 4,  # Placeholder, should be calculated by strategy
                # 0=long, 1=short in VectorBT
                'entry_type': 'long' if trade['direction'] == 0 else 'short',
                'reason_exit': 'target'  # Placeholder
            })

        return trades
