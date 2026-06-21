"""
Advanced Backtester Core for Multi-Timeframe Trading Strategies.

This module provides a comprehensive backtesting framework with:
- Walk-forward optimization
- Monte Carlo simulation
- Risk management and metrics calculation
- VectorBT integration for portfolio simulation
- FASE 1: Realistic execution modeling (market impact, order types, latency)

REFACTORED: Metrics, WFA, and Monte Carlo now use extracted modules.
"""

import pandas as pd
import numpy as np
import logging
import traceback
import threading
import sys
import os

# Conditional import for sklearn/skopt (not compatible with Python 3.14+)
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    gp_minimize = None
    Real = None
    Integer = None
    SKOPT_AVAILABLE = False
    logging.warning("skopt not available - Bayesian optimization disabled")

import vectorbt as vbt
from typing import Dict, List, Optional, Union, Tuple

# Add src to path for realistic execution components
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# AUDITORÍA FIX: Importar constantes globales
try:
    from core.constants import (
        RISK_FREE_RATE_DAILY,
        TRADING_DAYS_PER_YEAR,
        DEFAULT_INITIAL_CAPITAL,
        DEFAULT_WFA_STABILITY_THRESHOLD,
    )
except ImportError:
    # Fallbacks si constantes no disponibles
    RISK_FREE_RATE_DAILY = 0.04 / 252
    TRADING_DAYS_PER_YEAR = 252
    DEFAULT_INITIAL_CAPITAL = 100_000.0
    DEFAULT_WFA_STABILITY_THRESHOLD = 0.6

# Import extracted modules
from core.execution.metrics_calculator import MetricsCalculator
from core.execution.monte_carlo_simulator import MonteCarloSimulator
from core.execution.walk_forward_optimizer import WalkForwardOptimizer, WFAMethod

try:
    from src.execution.market_impact import MarketImpactModel, VolumeProfileAnalyzer
    from src.execution.order_manager import OrderManager, OrderType, OrderSide
    from src.execution.latency_model import LatencyProfile
    from core.risk.kelly_sizer import KellyPositionSizer
    from core.risk.risk_manager import RiskManager

    REALISTIC_EXECUTION_AVAILABLE = True
except ImportError as e:
    REALISTIC_EXECUTION_AVAILABLE = False
    logging.warning(f"Realistic execution components not available: {e}")

# ÁREA 4: Council Integration
try:
    from core.council import Council
    COUNCIL_AVAILABLE = True
except ImportError as e:
    COUNCIL_AVAILABLE = False
    logging.warning(f"Council not available: {e}")


class BacktesterCore:
    """
    Advanced backtesting engine for trading strategies.

    Provides comprehensive backtesting capabilities including:
    - Simple backtesting with metrics calculation
    - Walk-forward optimization
    - Monte Carlo simulation for robustness testing
    - Risk management and realistic cost modeling
    - VectorBT integration for portfolio simulation
    
    REFACTORED: Now delegates to extracted modules for metrics, WFA, and Monte Carlo.
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,
        slippage_pct: float = 0.001,
        enable_realistic_execution: bool = False,
        latency_profile: str = "retail_average",
        enable_kelly_position_sizing: bool = False,
        kelly_fraction: float = 0.5,
        max_position_pct: float = 0.10,
    ) -> None:
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
        
        # Initialize extracted modules
        self.metrics_calculator = MetricsCalculator(
            risk_free_rate=RISK_FREE_RATE_DAILY,
            trading_days=TRADING_DAYS_PER_YEAR
        )
        self.monte_carlo = MonteCarloSimulator(
            num_simulations=500,
            noise_percent=0.005,
            robustness_threshold=0.2
        )

        # Trade history tracking for Kelly statistics
        self.trade_history = pd.DataFrame(
            columns=[
                "timestamp",
                "side",
                "entry_price",
                "exit_price",
                "size",
                "pnl",
                "pnl_pct",
                "hold_time",
                "mae",
                "mfe",
            ]
        )
        self.current_capital = initial_capital  # Track capital dynamically

        # FASE 2: Initialize Kelly position sizer
        if enable_kelly_position_sizing and REALISTIC_EXECUTION_AVAILABLE:
            self.logger.info("🎯 Kelly position sizing enabled (FASE 2)")
            self.kelly_sizer = KellyPositionSizer(kelly_fraction=kelly_fraction, max_position_pct=max_position_pct)
            self.logger.info(f"   Kelly fraction: {kelly_fraction}")
            self.logger.info(f"   Max position: {max_position_pct*100}%")
        elif enable_kelly_position_sizing and not REALISTIC_EXECUTION_AVAILABLE:
            self.logger.warning("⚠️ Kelly position sizing requested but components not available")
            self.enable_kelly_position_sizing = False

        # FASE 2.5: Initialize Risk Manager (Kill Switch)
        if REALISTIC_EXECUTION_AVAILABLE:
            self.risk_manager = RiskManager({'max_daily_drawdown': 0.05}) # Default 5%
            self.logger.info("🛡️ Risk Manager initialized (Kill Switch active)")
        else:
            self.risk_manager = None

        # ÁREA 4: Initialize Council for trade approval
        self.enable_council = True  # Can be disabled for comparison
        if COUNCIL_AVAILABLE and self.enable_council:
            rules_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config", "rules")
            rules_dir = rules_dir if os.path.exists(rules_dir) else None
            self.council = Council(rules_dir=rules_dir)
            self.council.register_standard_experts()
            self.logger.info("🏛️ Council initialized (ÁREA 4)")
            self.logger.info(f"   Experts: {list(self.council.experts.keys())}")
        else:
            self.council = None
            self.logger.info("📊 Council disabled - direct execution mode")
        
        # Council decision tracking
        self.council_decisions = {
            "approved": [],
            "vetoed": [],
            "warnings": []
        }
        self.strategy_id = "unknown"  # Will be set when running backtest

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

    def _record_trade(
        self,
        timestamp,
        side: str,
        entry_price: float,
        exit_price: float,
        size: float,
        hold_time: float = 0.0,
        mae: float = 0.0,
        mfe: float = 0.0,
    ):
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
        pnl = (exit_price - entry_price) * size if side == "buy" else (entry_price - exit_price) * size
        pnl_pct = (pnl / (entry_price * size)) if entry_price > 0 else 0.0

        trade_record = {
            "timestamp": timestamp,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "hold_time": hold_time,
            "mae": mae,  # Maximum Adverse Excursion (%)
            "mfe": mfe,  # Maximum Favorable Excursion (%)
        }

        # Append to history
        self.trade_history.loc[len(self.trade_history)] = trade_record

        # Update current capital
        self._update_capital(pnl)

        self.logger.debug(
            f"Trade recorded: {side} @ {entry_price:.2f}→{exit_price:.2f}, "
            f"PnL: ${pnl:.2f} ({pnl_pct:.2%}), MAE: {mae:.2%}, MFE: {mfe:.2%}"
        )

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
        wins = recent_trades[recent_trades["pnl"] > 0]
        losses = recent_trades[recent_trades["pnl"] < 0]

        if len(losses) == 0:
            # All wins (unlikely but handle gracefully)
            return 1.0, 2.0  # Very conservative despite 100% win rate

        win_rate = len(wins) / len(recent_trades)
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["pnl"].mean())
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        self.logger.debug(
            f"Strategy stats from last {len(recent_trades)} trades: " f"WR={win_rate:.2%}, W/L={win_loss_ratio:.2f}"
        )

        return win_rate, win_loss_ratio

    # ========================================================================
    # ÁREA 4: Council Integration Methods
    # ========================================================================
    
    def _consult_council_for_trade(
        self,
        signal_type: str,
        timestamp: pd.Timestamp,
        df: pd.DataFrame,
        signal_value: int = 1,
        equity_curve: list = None
    ) -> Dict[str, Any]:
        """
        Consulta al Council sobre una señal de trading.
        
        Args:
            signal_type: "entry" or "exit"
            timestamp: Timestamp de la señal
            df: DataFrame con datos OHLCV
            signal_value: 1=long, -1=short
            equity_curve: Lista de equity history para calcular drawdown
        
        Returns:
            dict: Council decision con campos:
                  - decision: 1 (APPROVE), 0 (WARNING), -1 (VETO)
                  - aggregate_score: float (-1.0 a 1.0)
                  - expert_votes: dict con votos de cada experto
        """
        if self.council is None:
            # Council disabled - approve all trades
            return {"decision": 1, "aggregate_score": 1.0, "reason": "Council disabled"}
        
        try:
            loc = df.index.get_loc(timestamp)
        except KeyError:
            return {"decision": 1, "aggregate_score": 0.5, "reason": "Timestamp not in index"}
        
        # Calculate current drawdown
        if equity_curve and len(equity_curve) > 0:
            peak = max(equity_curve)
            current_dd = (peak - self.current_capital) / peak if peak > 0 else 0
        else:
            current_dd = 0
        
        # Build context for Council
        context = {
            "signal": signal_value,
            "signal_type": signal_type,
            "timestamp": str(timestamp),
            "current_equity": self.current_capital,
            "initial_capital": self.initial_capital,
            "current_dd": current_dd,
            "strategy_id": self.strategy_id,
            "num_trades": len(self.trade_history),
            "win_rate": self._calculate_current_win_rate(),
            # Data quality context
            "data_quality": {
                "has_gaps": self._check_data_gaps(df, loc),
                "volume_ok": df["volume"].iloc[loc] > 0 if "volume" in df.columns and loc < len(df) else True
            }
        }
        
        # Consult Council
        decision = self.council.decide(context)
        
        # Record decision
        decision_record = {
            "timestamp": str(timestamp),
            "signal_type": signal_type,
            "context_summary": {
                "equity": self.current_capital,
                "dd": current_dd,
                "win_rate": context["win_rate"]
            },
            "decision": decision
        }
        
        if decision.get("decision", 0) > 0:
            self.council_decisions["approved"].append(decision_record)
        elif decision.get("decision", 0) < 0 or decision.get("phase") == "VETO":
            self.council_decisions["vetoed"].append(decision_record)
        else:
            self.council_decisions["warnings"].append(decision_record)
        
        return decision
    
    def _calculate_current_win_rate(self) -> float:
        """Calculate win rate from trade history."""
        if len(self.trade_history) == 0:
            return 0.5  # Default neutral
        
        wins = len(self.trade_history[self.trade_history["pnl"] > 0])
        return wins / len(self.trade_history)
    
    def _check_data_gaps(self, df: pd.DataFrame, loc: int, lookback: int = 10) -> bool:
        """Check if there are data gaps in recent history."""
        if loc < lookback:
            return False
        
        try:
            recent_df = df.iloc[loc-lookback:loc]
            if len(recent_df) < 2:
                return False
            
            # Check for time gaps (more than 2x expected frequency)
            time_diffs = recent_df.index.to_series().diff()
            median_diff = time_diffs.median()
            
            if median_diff is None or pd.isna(median_diff):
                return False
                
            # Gap if any diff is more than 3x median
            has_gap = any(time_diffs > median_diff * 3)
            return has_gap
        except Exception:
            return False
    
    def _get_council_stats(self) -> Dict[str, Any]:
        """Get summary statistics of Council decisions."""
        total_signals = (
            len(self.council_decisions["approved"]) +
            len(self.council_decisions["vetoed"]) +
            len(self.council_decisions["warnings"])
        )
        
        return {
            "total_signals": total_signals,
            "approved": len(self.council_decisions["approved"]),
            "vetoed": len(self.council_decisions["vetoed"]),
            "warnings": len(self.council_decisions["warnings"]),
            "veto_rate": (
                len(self.council_decisions["vetoed"]) / total_signals
                if total_signals > 0 else 0
            ),
            "approval_rate": (
                len(self.council_decisions["approved"]) / total_signals
                if total_signals > 0 else 0
            )
        }
    
    def _reset_council_decisions(self):
        """Reset Council decision tracking for new backtest."""
        self.council_decisions = {
            "approved": [],
            "vetoed": [],
            "warnings": []
        }
    
    # ========================================================================
    # AUDITORÍA FIX: Eliminados métodos duplicados (_record_trade y _update_capital)
    # ========================================================================
    
    # ========================================================================
    # AUDITORÍA ROUND 12: Helpers para _process_and_record_trades
    # ========================================================================

    def _extract_trade_info(self, trade) -> dict:
        """
        Extrae información de un trade del array de VectorBT.
        
        Returns:
            Dict con entry_idx, exit_idx, entry_price, exit_price, size, side
            o None si hay error.
        """
        try:
            entry_idx = int(trade['entry_idx'])
            exit_idx = int(trade['exit_idx'])
            entry_price = float(trade['entry_price'])
            exit_price = float(trade['exit_price'])
            size = float(trade['size'])
            
            # VectorBT direction: 0=Long, 1=Short
            if 'direction' in trade.dtype.names:
                direction = int(trade['direction'])
                side = "buy" if direction == 0 else "sell"
            else:
                side = "buy"
                self.logger.debug("Trade direction not available, assuming long trade")
            
            return {
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'side': side
            }
        except (ValueError, IndexError, KeyError):
            # Fallback to legacy index-based access
            self.logger.warning("Using legacy index access for trade records")
            return {
                'entry_idx': int(trade[2]),
                'exit_idx': int(trade[3]),
                'entry_price': float(trade[5]),
                'exit_price': float(trade[6]),
                'size': float(trade[4]),
                'side': "buy"
            }

    def _calculate_mae_mfe(self, df_5m, entry_idx: int, exit_idx: int, 
                           entry_price: float, side: str) -> tuple:
        """
        Calcula MAE (Maximum Adverse Excursion) y MFE (Maximum Favorable Excursion).
        """
        high_series = df_5m["high"].iloc[entry_idx:exit_idx + 1]
        low_series = df_5m["low"].iloc[entry_idx:exit_idx + 1]
        max_price = high_series.max()
        min_price = low_series.min()

        if side == "buy":
            mae = (entry_price - min_price) / entry_price if min_price < entry_price else 0.0
            mfe = (max_price - entry_price) / entry_price if max_price > entry_price else 0.0
        else:
            mae = (max_price - entry_price) / entry_price if max_price > entry_price else 0.0
            mfe = (entry_price - min_price) / entry_price if min_price < entry_price else 0.0

        return mae, mfe

    def _process_and_record_trades(self, portfolio, df_5m):
        """
        Process trades from VectorBT portfolio and record them in trade history.
        Updates current_capital based on realized PnL.

        Args:
            portfolio: VectorBT portfolio object
            df_5m: DataFrame with 5min data for timestamps
            
        AUDITORÍA ROUND 12: Refactorizado para reducir complejidad de 32 a ~12.
        """
        if not hasattr(portfolio, "trades") or portfolio.trades.count() == 0:
            self.logger.debug("No trades to record")
            return

        try:
            trades_arr = portfolio.trades.records
            recorded_count = 0

            for i in range(len(trades_arr)):
                trade_info = self._extract_trade_info(trades_arr[i])
                
                # Validar índices
                if trade_info['entry_idx'] >= len(df_5m) or trade_info['exit_idx'] >= len(df_5m):
                    continue

                # Calcular timestamps y hold time
                entry_time = df_5m.index[trade_info['entry_idx']]
                exit_time = df_5m.index[trade_info['exit_idx']]
                hold_time = (exit_time - entry_time).total_seconds() / 3600.0

                # Calcular MAE/MFE
                mae, mfe = self._calculate_mae_mfe(
                    df_5m, 
                    trade_info['entry_idx'], 
                    trade_info['exit_idx'],
                    trade_info['entry_price'], 
                    trade_info['side']
                )

                # Registrar trade
                self._record_trade(
                    timestamp=exit_time,
                    side=trade_info['side'],
                    entry_price=trade_info['entry_price'],
                    exit_price=trade_info['exit_price'],
                    size=trade_info['size'],
                    hold_time=hold_time,
                    mae=mae,
                    mfe=mfe,
                )
                recorded_count += 1

            self.logger.info(f"Recorded {recorded_count} trades. Current capital: ${self.current_capital:.2f}")

        except (KeyError, IndexError, TypeError) as e:
            self.logger.warning(f"Could not record trades: {e}. Continuing without trade history update.")

    def _calculate_position_size(
        self,
        capital: float,
        win_rate: float = None,
        win_loss_ratio: float = None,
        current_volatility: float = 0.0,
        market_impact_pct: float = 0.0,
    ) -> float:
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
        if self.enable_kelly_position_sizing and hasattr(self, "kelly_sizer"):
            # Calculate real statistics from trade history if not provided
            if win_rate is None or win_loss_ratio is None:
                win_rate, win_loss_ratio = self._get_strategy_statistics()

            try:
                sizing_result = self.kelly_sizer.calculate_position_size(
                    capital=capital,
                    win_rate=win_rate,
                    win_loss_ratio=win_loss_ratio,
                    current_volatility=current_volatility,
                    market_impact_pct=market_impact_pct,
                )
                position_size = sizing_result["position_size"]

                # Log Kelly sizing details
                self.logger.debug(
                    f"Kelly position sizing: ${position_size:.2f} " f"({sizing_result['position_pct']:.1%} of capital)"
                )

                return position_size

            except Exception as e:
                self.logger.error(f"Error in Kelly position sizing: {e}")
                return capital * 0.01  # Fallback

        else:
            # Simple position sizing: 1% of capital
            return capital * 0.01

    def _calculate_order_size_for_execution(
        self, base_price: float, current_capital: float, volatility_val: float
    ) -> float:
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
                market_impact_pct=0.001,  # 0.1% market impact estimate
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
                raise ValueError(f"Insufficient data for timeframe {tf}: {len(df)} bars < {min_bars} minimum")

        return True

    def cap_extreme_metrics(self, metrics: Dict) -> Dict:
        """Cap extreme metric values to prevent unrealistic results"""
        capped_metrics = metrics.copy()

        # Cap Sharpe ratio to [-10, 10]
        if "sharpe" in capped_metrics:
            original_sharpe = capped_metrics["sharpe"]
            capped_metrics["sharpe"] = max(-10, min(10, capped_metrics["sharpe"]))
            if original_sharpe != capped_metrics["sharpe"]:
                self.logger.warning(f"Sharpe ratio capped from {original_sharpe} to " f"{capped_metrics['sharpe']}")

        # Cap Sortino ratio to [-10, 10]
        if "sortino" in capped_metrics:
            original_sortino = capped_metrics["sortino"]
            capped_metrics["sortino"] = max(-10, min(10, capped_metrics["sortino"]))
            if original_sortino != capped_metrics["sortino"]:
                self.logger.warning(f"Sortino ratio capped from {original_sortino} to " f"{capped_metrics['sortino']}")

        # Cap drawdown to [0, 1] (0-100%)
        if "max_dd" in capped_metrics:
            capped_metrics["max_dd"] = max(0, min(1, capped_metrics["max_dd"]))

        # Cap profit factor to [0, 100]
        if "profit_factor" in capped_metrics:
            if capped_metrics["profit_factor"] == float("inf"):
                capped_metrics["profit_factor"] = 100
            else:
                capped_metrics["profit_factor"] = min(100, capped_metrics["profit_factor"])

        return capped_metrics

    def _check_kill_switch(self, portfolio) -> Optional[pd.Timestamp]:
        """
        Check if the Kill Switch would have been triggered during the backtest.
        Returns the timestamp where the halt occurred, or None.
        """
        if not self.risk_manager:
            return None

        equity_curve = portfolio.value()
        
        # Reset risk manager for this run
        self.risk_manager.reset_kill_switch()
        # Initialize with first equity point
        if not equity_curve.empty:
            self.risk_manager.daily_start_equity = equity_curve.iloc[0]
            self.risk_manager.current_date = equity_curve.index[0].date()
            self.risk_manager.current_equity = equity_curve.iloc[0]

        for timestamp, equity in equity_curve.items():
            self.risk_manager.update_state(equity, timestamp.date())

    # ==========================================================================
    # REFACTORED: Extracted methods for run_simple_backtest complexity reduction
    # ==========================================================================
    
    def _prepare_backtest_data(self, df_multi_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare and validate data for backtesting."""
        self.validate_data_sufficiency(df_multi_tf)
        self._check_cancellation()
        
        df_5m = df_multi_tf["5min"].copy()
        df_5m.columns = df_5m.columns.str.lower()
        return df_5m
    
    def _calculate_volatility(self, df_5m: pd.DataFrame) -> pd.Series:
        """Calculate volatility from ATR or rolling std."""
        if "atr" in df_5m.columns:
            return df_5m["atr"] / df_5m["close"]
        elif "ATR" in df_5m.columns:
            return df_5m["ATR"] / df_5m["close"]
        else:
            return df_5m["close"].pct_change().rolling(20).std()
    
    def _process_entry_signals(
        self, 
        signals: pd.DataFrame, 
        df_5m: pd.DataFrame, 
        avg_volume: pd.Series, 
        volatility: pd.Series,
        equity_history: list
    ) -> tuple:
        """
        Process entry signals with realistic execution and Council veto.
        
        Returns:
            tuple: (adjusted_entries, entry_prices, total_market_impact, total_latency_cost, vetoed_entries)
        """
        adjusted_entries = signals["entries"].copy()
        entry_prices = df_5m["close"].copy()
        total_market_impact = 0.0
        total_latency_cost = 0.0
        vetoed_entries = []
        
        entry_indices = signals["entries"][signals["entries"]].index
        
        for idx in entry_indices:
            if idx not in df_5m.index:
                continue
            loc = df_5m.index.get_loc(idx)
            if loc >= len(avg_volume):
                continue
            
            # Consult Council before executing
            if self.council is not None:
                council_decision = self._consult_council_for_trade(
                    signal_type="entry",
                    timestamp=idx,
                    df=df_5m,
                    signal_value=1,
                    equity_curve=equity_history
                )
                
                if council_decision.get("decision", 0) < 0 or council_decision.get("phase") == "VETO":
                    self.logger.debug(f"⛔ Council VETOED entry at {idx}")
                    adjusted_entries.loc[idx] = False
                    vetoed_entries.append(idx)
                    continue
            
            # Calculate execution with market impact
            base_price = df_5m.loc[idx, "close"]
            vol = avg_volume.iloc[loc] if loc < len(avg_volume) else avg_volume.iloc[-1]
            volatility_val = volatility.iloc[loc] if loc < len(volatility) else 0.02
            
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
                side="buy",
                timestamp=idx,
            )
            
            entry_prices.loc[idx] = exec_data["execution_price"]
            
            # Track costs
            impact_cost = abs(exec_data["execution_price"] - base_price) * order_size
            total_market_impact += impact_cost
            
            if "latency_cost" in exec_data:
                total_latency_cost += exec_data["latency_cost"]
        
        return adjusted_entries, entry_prices, total_market_impact, total_latency_cost, vetoed_entries
    
    def _process_exit_signals(
        self,
        signals: pd.DataFrame,
        df_5m: pd.DataFrame,
        avg_volume: pd.Series,
        volatility: pd.Series
    ) -> tuple:
        """
        Process exit signals with realistic execution.
        
        Returns:
            tuple: (exit_prices, total_market_impact, total_latency_cost)
        """
        exit_prices = df_5m["close"].copy()
        total_market_impact = 0.0
        total_latency_cost = 0.0
        
        exit_indices = signals["exits"][signals["exits"]].index
        
        for idx in exit_indices:
            if idx not in df_5m.index:
                continue
            loc = df_5m.index.get_loc(idx)
            if loc >= len(avg_volume):
                continue
            
            base_price = df_5m.loc[idx, "close"]
            vol = avg_volume.iloc[loc] if loc < len(avg_volume) else avg_volume.iloc[-1]
            volatility_val = volatility.iloc[loc] if loc < len(volatility) else 0.02
            
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
                side="sell",
                timestamp=idx,
            )
            
            exit_prices.loc[idx] = exec_data["execution_price"]
            
            # Track costs
            impact_cost = abs(exec_data["execution_price"] - base_price) * order_size
            total_market_impact += impact_cost
            
            if "latency_cost" in exec_data:
                total_latency_cost += exec_data["latency_cost"]
        
        return exit_prices, total_market_impact, total_latency_cost
    
    def _build_backtest_result(
        self,
        metrics: Dict,
        trades: list,
        equity_curve: pd.Series,
        signals: pd.DataFrame,
        strategy_parameters: Dict,
        execution_costs: Optional[Dict] = None,
        entry_indices: Optional[list] = None,
        exit_indices: Optional[list] = None
    ) -> Dict:
        """Build the final backtest result dictionary."""
        result = {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve.tolist(),
            "signals": (
                signals[["signals"]].to_dict("records")
                if hasattr(signals, "to_dict") and "signals" in signals.columns
                else []
            ),
            "strategy_parameters": strategy_parameters,
            "final_capital": self.current_capital,
            "capital_growth": ((self.current_capital / self.initial_capital) - 1.0) * 100,
        }
        
        # Add Kelly position sizing info if enabled
        if self.enable_kelly_position_sizing and len(self.trade_history) > 0:
            win_rate, wl_ratio = self._get_strategy_statistics()
            result["kelly_info"] = {
                "enabled": True,
                "trades_recorded": len(self.trade_history),
                "win_rate": win_rate,
                "win_loss_ratio": wl_ratio,
                "kelly_fraction": self.kelly_sizer.kelly_fraction,
                "max_position_pct": self.kelly_sizer.max_position_pct,
            }
        
        # Add execution costs if realistic execution was used
        if execution_costs:
            total_trades = len(entry_indices or []) + len(exit_indices or [])
            total_cost = execution_costs["market_impact"] + execution_costs["latency_cost"]
            result["execution_costs"] = {
                "total_market_impact": execution_costs["market_impact"],
                "total_latency_cost": execution_costs["latency_cost"],
                "total_execution_cost": total_cost,
                "num_trades": total_trades,
                "avg_cost_per_trade": total_cost / max(1, total_trades),
                "latency_profile": self.latency_profile_name,
            }
        
        # Add Council statistics
        if self.council is not None:
            council_stats = self._get_council_stats()
            result["council_stats"] = council_stats
            
            if council_stats["vetoed"] > 0:
                self.logger.info(
                    f"🏛️ Council Summary: {council_stats['approved']} approved, "
                    f"{council_stats['vetoed']} vetoed ({council_stats['veto_rate']:.1%} veto rate)"
                )
        
        return result

    def run_simple_backtest(self, df_multi_tf: Dict[str, pd.DataFrame], strategy_class, strategy_params: Dict) -> Dict:
        """
        Run a simple backtest with optional realistic execution and Council integration.
        
        Refactored to reduce cognitive complexity by extracting phases into helper methods.
        """
        try:
            # Reset cancellation flag and Council decisions
            self._cancel_flag.clear()
            self._reset_council_decisions()
            
            # Set strategy ID for Council context
            self.strategy_id = strategy_params.get('name', strategy_params.get('strategy_name', strategy_class.__name__))

            # Phase 1: Prepare data
            df_5m = self._prepare_backtest_data(df_multi_tf)

            # Phase 2: Initialize and generate signals
            strategy = strategy_class(**strategy_params)
            self._check_cancellation()
            signals = strategy.generate_signals(df_multi_tf)
            self._check_cancellation()

            # Phase 3: Execute backtest (realistic or simple)
            portfolio, execution_costs, entry_indices, exit_indices = self._execute_backtest(
                df_5m, signals
            )

            # Phase 4: Process results
            return self._process_backtest_results(
                portfolio, df_5m, signals, strategy, execution_costs, entry_indices, exit_indices
            )

        except InterruptedError:
            self.logger.info("Backtest cancelled")
            return {"error": "Backtest cancelled by user"}
        except Exception as e:
            error_msg = f"Error in simple backtest: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "traceback": traceback.format_exc()}
    
    def _execute_backtest(
        self, 
        df_5m: pd.DataFrame, 
        signals: pd.DataFrame
    ) -> tuple:
        """
        Execute backtest with either realistic or simple execution.
        
        Returns:
            tuple: (portfolio, execution_costs, entry_indices, exit_indices)
        """
        entry_indices = None
        exit_indices = None
        execution_costs = None
        
        if self.enable_realistic_execution:
            portfolio, execution_costs, entry_indices, exit_indices = self._run_realistic_execution(
                df_5m, signals
            )
            self.logger.info("✓ Backtest with realistic execution complete")
        else:
            portfolio = self._run_simple_execution(df_5m, signals)
        
        # Validate portfolio
        if portfolio is None or not hasattr(portfolio, "trades"):
            raise ValueError("Portfolio creation failed")
        
        return portfolio, execution_costs, entry_indices, exit_indices
    
    def _run_realistic_execution(
        self, 
        df_5m: pd.DataFrame, 
        signals: pd.DataFrame
    ) -> tuple:
        """
        Run backtest with realistic execution including market impact and Council veto.
        
        Returns:
            tuple: (portfolio, execution_costs, entry_indices, exit_indices)
        """
        # Calculate volume and volatility
        avg_volume = self.volume_analyzer.calculate_average_volume(df_5m, lookback_periods=20)
        volatility = self._calculate_volatility(df_5m)
        
        # Track equity for Council context
        equity_history = [self.initial_capital]
        
        # Process entry signals
        adjusted_entries, entry_prices, entry_impact, entry_latency, vetoed = self._process_entry_signals(
            signals, df_5m, avg_volume, volatility, equity_history
        )
        
        # Process exit signals
        exit_prices, exit_impact, exit_latency = self._process_exit_signals(
            signals, df_5m, avg_volume, volatility
        )
        
        # Create portfolio with adjusted prices
        portfolio = vbt.Portfolio.from_signals(
            close=df_5m["close"],
            entries=adjusted_entries,
            exits=signals["exits"],
            price=entry_prices,
            init_cash=self.initial_capital,
            fees=0.0001,
            slippage=0.0,
        )
        
        # Build execution costs
        execution_costs = {
            "market_impact": entry_impact + exit_impact,
            "latency_cost": entry_latency + exit_latency
        }
        
        entry_indices = list(signals["entries"][signals["entries"]].index)
        exit_indices = list(signals["exits"][signals["exits"]].index)
        
        return portfolio, execution_costs, entry_indices, exit_indices
    
    def _run_simple_execution(
        self, 
        df_5m: pd.DataFrame, 
        signals: pd.DataFrame
    ) -> Any:
        """Run simple backtest without realistic execution."""
        return vbt.Portfolio.from_signals(
            close=df_5m["close"],
            entries=signals["entries"],
            exits=signals["exits"],
            price=df_5m["close"],
            init_cash=self.initial_capital,
            fees=self.commission,
            slippage=self.slippage_pct,
        )
    
    def _process_backtest_results(
        self,
        portfolio,
        df_5m: pd.DataFrame,
        signals: pd.DataFrame,
        strategy,
        execution_costs: Optional[Dict],
        entry_indices: Optional[list],
        exit_indices: Optional[list]
    ) -> Dict:
        """Process portfolio results and build result dictionary."""
        # Check Kill Switch
        halt_timestamp = self._check_kill_switch(portfolio)
        
        returns = portfolio.returns()
        trades_records = portfolio.trades.records
        equity_curve = portfolio.value()
        
        # Truncate if halted
        if halt_timestamp:
            self.logger.info(f"⚠️ Backtest truncated due to Kill Switch at {halt_timestamp}")
            returns = returns[returns.index <= halt_timestamp]
            equity_curve = equity_curve[equity_curve.index <= halt_timestamp]
            
            if halt_timestamp in df_5m.index:
                halt_idx = df_5m.index.get_loc(halt_timestamp)
                try:
                    trades_records = trades_records[trades_records['exit_idx'] <= halt_idx]
                except (ValueError, IndexError, KeyError):
                    pass
        
        # Calculate metrics
        metrics = self.calculate_metrics(returns, trades_records, df_5m["close"])
        metrics = self.cap_extreme_metrics(metrics)
        
        # Process trades for Kelly statistics
        if self.enable_kelly_position_sizing:
            self._process_and_record_trades(portfolio, df_5m)
            self.logger.info(f"📊 Kelly Statistics updated: {len(self.trade_history)} trades recorded")
        
        # Format trades
        trades = self._format_trades(trades_records, df_5m.index)
        
        # Calculate realistic costs for trades
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df = self.calculate_realistic_costs(trades_df)
        
        # Build and return result
        return self._build_backtest_result(
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            signals=signals,
            strategy_parameters=strategy.get_parameters(),
            execution_costs=execution_costs,
            entry_indices=entry_indices,
            exit_indices=exit_indices
        )

    def run_backtest(
        self,
        strategy_class,
        df_multi_tf: Union[Dict[str, pd.DataFrame], pd.DataFrame],
        strategy_params: Optional[Dict] = None,
    ) -> Dict:
        """Alias for run_simple_backtest with different parameter order for compatibility"""
        if strategy_params is None:
            strategy_params = {}

        # Convert DataFrame to dict format if needed
        if isinstance(df_multi_tf, pd.DataFrame):
            df_multi_tf = {"5min": df_multi_tf}

        return self.run_simple_backtest(df_multi_tf, strategy_class, strategy_params)

    def run_walk_forward(
        self,
        df_multi_tf: Dict[str, pd.DataFrame],
        strategy_class,
        strategy_params: Dict = None,
        param_ranges: Dict = None,
        n_periods: int = 8,
        opt_method: str = "bayes",
        min_test_bars: int = 100,
    ) -> Dict:
        """
        Walk-Forward Analysis con optimización real.
        
        ÁREA 2 FIX: Ahora optimiza parámetros en cada período IS y valida en OOS.
        
        Args:
            df_multi_tf: Dict de DataFrames por timeframe
            strategy_class: Clase de estrategia a optimizar
            strategy_params: Parámetros iniciales (usados si no hay param_ranges)
            param_ranges: Rangos para optimización. Formato:
                {
                    'param_name': {'type': 'int'|'float', 'min': X, 'max': Y},
                    ...
                }
            n_periods: Número de períodos WFA
            opt_method: 'bayes' para optimización bayesiana, 'none' para usar params fijos
            min_test_bars: Mínimo de barras para período de test
            
        Returns:
            Dict con period_results, stability_score, certified, best_params
        """
        try:
            df_5m = df_multi_tf["5min"].copy()
            total_bars = len(df_5m)
            period_size = total_bars // n_periods
            
            # ÁREA 2: Validar que hay param_ranges para optimización real
            use_optimization = opt_method == "bayes" and param_ranges is not None
            if use_optimization:
                self.logger.info(f"🧬 WFA con optimización bayesiana ({n_periods} períodos)")
            else:
                self.logger.info(f"📊 WFA sin optimización ({n_periods} períodos)")
                if param_ranges is None and strategy_params is None:
                    return {"error": "Debe proporcionar strategy_params o param_ranges"}

            periods_results = []
            all_train_sharpes = []
            all_test_sharpes = []
            all_degradations = []
            all_optimized_params = []
            best_params = strategy_params or {}  # Initialize with default params

            for i in range(n_periods):
                # Check for cancellation
                self._check_cancellation()

                # ÁREA 2: Usar ventana expandida (Anchored WFA)
                # IS: desde el inicio hasta el período actual
                # OOS: período siguiente
                train_start = 0  # Anchored: siempre desde el inicio
                train_end = (i + 1) * period_size
                test_start = train_end
                test_end = min((i + 2) * period_size, total_bars)

                if test_end - test_start < min_test_bars:
                    self.logger.warning(f"Período {i+1}: OOS muy pequeño ({test_end - test_start} < {min_test_bars}), saltando")
                    break

                # Split data
                train_data = {tf: df.iloc[train_start:train_end] for tf, df in df_multi_tf.items()}
                test_data = {tf: df.iloc[test_start:test_end] for tf, df in df_multi_tf.items()}
                
                self.logger.info(f"📈 Período {i+1}/{n_periods}: IS[0:{train_end}] -> OOS[{test_start}:{test_end}]")

                # ÁREA 2 FIX: Optimizar en cada período si hay param_ranges
                if use_optimization:
                    self.logger.info(f"   🔍 Optimizando parámetros en IS...")
                    best_params = self._bayesian_optimize(strategy_class, train_data, param_ranges)
                    all_optimized_params.append(best_params.copy())
                    self.logger.info(f"   ✅ Params período {i+1}: {best_params}")
                elif strategy_params:
                    best_params = strategy_params

                # Ejecutar backtest en IS y OOS con los parámetros optimizados
                train_result = self.run_backtest(strategy_class, train_data, best_params)
                test_result = self.run_backtest(strategy_class, test_data, best_params)

                if "error" not in train_result and "error" not in test_result:
                    train_sharpe = train_result["metrics"]["sharpe"]
                    test_sharpe = test_result["metrics"]["sharpe"]
                    
                    # ÁREA 2: Calcular degradación correctamente
                    if abs(train_sharpe) > 0.01:
                        # Degradación = (IS - OOS) / |IS| * 100
                        # Positivo = OOS peor que IS (esperado)
                        # Negativo = OOS mejor que IS (raro pero posible)
                        degradation_pct = ((train_sharpe - test_sharpe) / abs(train_sharpe)) * 100
                    else:
                        degradation_pct = 0 if abs(test_sharpe) < 0.01 else -100

                    period_result = {
                        "period": i + 1,
                        "train_bars": train_end - train_start,
                        "test_bars": test_end - test_start,
                        "train_metrics": train_result["metrics"],
                        "test_metrics": test_result["metrics"],
                        "best_params": best_params.copy() if use_optimization else None,
                        "degradation_pct": degradation_pct,
                    }
                    periods_results.append(period_result)

                    all_train_sharpes.append(train_sharpe)
                    all_test_sharpes.append(test_sharpe)
                    all_degradations.append(degradation_pct)
                    
                    self.logger.info(f"   📊 IS Sharpe: {train_sharpe:.2f} -> OOS Sharpe: {test_sharpe:.2f} (Degradación: {degradation_pct:.1f}%)")
                else:
                    error_msg = train_result.get("error", "") or test_result.get("error", "")
                    self.logger.warning(f"   ⚠️ Período {i+1} falló: {error_msg}")

            # ÁREA 2: Calcular métricas de robustez
            if periods_results:
                avg_degradation = np.mean(all_degradations)
                std_degradation = np.std(all_degradations)
                avg_oos_sharpe = np.mean(all_test_sharpes)
                
                # Stability Score (0-1):
                # - Penaliza degradación alta
                # - Penaliza variabilidad entre períodos
                # - 1.0 = perfecto (sin degradación, OOS igual a IS)
                degradation_penalty = min(abs(avg_degradation) / 100, 1.0)  # 0-1
                variability_penalty = min(std_degradation / 50, 0.5)  # Max 0.5
                stability_score = max(0, 1.0 - degradation_penalty - variability_penalty)
                
                # ÁREA 2: Certificación basada en criterios
                # Estrategia certificada si:
                # 1. Degradación promedio < 30%
                # 2. OOS Sharpe promedio > 0.5
                # 3. Stability Score > 0.5
                certified = (
                    abs(avg_degradation) < 30 and
                    avg_oos_sharpe > 0.5 and
                    stability_score > 0.5
                )
                
                self.logger.info(f"\n🏁 WFA Completado:")
                self.logger.info(f"   📉 Degradación Promedio: {avg_degradation:.1f}%")
                self.logger.info(f"   📊 OOS Sharpe Promedio: {avg_oos_sharpe:.2f}")
                self.logger.info(f"   🎯 Stability Score: {stability_score:.2f}")
                self.logger.info(f"   {'✅ CERTIFICADA' if certified else '❌ NO CERTIFICADA'}")
            else:
                avg_degradation = 0
                stability_score = 0
                certified = False
                avg_oos_sharpe = 0

            return {
                "period_results": periods_results,
                "avg_degradation": avg_degradation,
                "avg_oos_sharpe": avg_oos_sharpe,
                "stability_score": stability_score,
                "certified": certified,
                "best_params": best_params,
                "all_optimized_params": all_optimized_params if use_optimization else None,
                "optimization_used": use_optimization,
            }

        except InterruptedError:
            self.logger.info("Walk-forward analysis cancelled")
            return {"error": "Walk-forward cancelled by user"}
        except Exception as e:
            error_msg = f"Error in walk-forward analysis: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "traceback": traceback.format_exc()}

    def run_monte_carlo(
        self,
        df_multi_tf: Dict[str, pd.DataFrame],
        strategy_class,
        strategy_params: Dict,
        n_simulations: int = 500,
        noise_pct: float = 10,
        seed: int | None = None,
    ) -> Dict:
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
                    noisy_df["Close"] = df["Close"] * (1 + noise)
                    noisy_df["High"] = df["High"] * (1 + noise * 0.5)
                    noisy_df["Low"] = df["Low"] * (1 + noise * 0.5)
                    noisy_data[tf] = noisy_df

                # Run backtest
                result = self.run_simple_backtest(noisy_data, strategy_class, strategy_params)

                if "error" not in result:
                    sharpe_results.append(result["metrics"]["sharpe"])
                    win_rate_results.append(result["metrics"]["win_rate"])

            if sharpe_results:
                sharpe_mean = np.mean(sharpe_results)
                sharpe_std = np.std(sharpe_results)
                robust = sharpe_std < 0.2  # Robust if std < 0.2

                # Create simulations list with individual results
                simulations = []
                for i, (sharpe, win_rate) in enumerate(zip(sharpe_results, win_rate_results)):
                    simulations.append({"simulation_id": i, "sharpe_ratio": sharpe, "win_rate": win_rate})

                return {
                    "simulations": simulations,
                    "summary_stats": {
                        "sharpe_mean": sharpe_mean,
                        "sharpe_std": sharpe_std,
                        "win_rate_mean": np.mean(win_rate_results),
                        "win_rate_std": np.std(win_rate_results),
                        "robust": robust,
                    },
                    "sharpe_distribution": sharpe_results,
                }
            else:
                return {"error": "No valid Monte Carlo results"}

        except InterruptedError:
            self.logger.info("Monte Carlo cancelled")
            return {"error": "Monte Carlo cancelled by user"}
        except Exception as e:
            error_msg = f"Error in Monte Carlo simulation: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "traceback": traceback.format_exc()}

    def _calculate_realistic_execution_price(
        self,
        base_price: float,
        order_size: float,
        avg_volume: float,
        volatility: float,
        side: str,
        timestamp: Optional[pd.Timestamp] = None,
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
            slippage_mult = 1 + self.slippage_pct if side == "buy" else 1 - self.slippage_pct
            return {
                "execution_price": base_price * slippage_mult,
                "impact_cost": 0.0,
                "latency_ms": 0.0,
                "realistic": False,
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
            urgency=1.0,
        )

        # Calculate execution price with impact
        execution_price = self.market_impact_model.calculate_execution_price(
            side=side, price=base_price, impact_pct=impact["total_impact_pct"]
        )

        # Calculate latency
        vol_multiplier = 1.0 + (volatility - 0.02) / 0.02  # Scale around 2% baseline
        latency_ms = self.latency_model.calculate_total_latency(
            order_type="market", market_volatility=max(0.5, vol_multiplier), time_of_day=hour
        )

        return {
            "execution_price": execution_price,
            "impact_cost": impact["total_impact_dollars"],
            "latency_ms": latency_ms,
            "impact_pct": impact["total_impact_pct"],
            "realistic": True,
        }

    def calculate_realistic_costs(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Commission: 0.1% round-trip
            trades_df["commission_cost"] = trades_df["pnl_pct"].abs() * 0.001

            # Slippage: base + vol_spike adjustment
            base_slippage = self.slippage_pct
            vol_spike_mult = 1.5  # Could be calculated from volatility
            trades_df["slippage_cost"] = trades_df["pnl_pct"].abs() * (base_slippage * vol_spike_mult)

            # Funding rate (if perpetual futures) - simplified
            funding_rate = 0.0001  # 0.01% per 8h, simplified to per trade
            trades_df["funding_cost"] = trades_df["pnl_pct"].abs() * funding_rate

            # Total cost
            trades_df["total_cost"] = (
                trades_df["commission_cost"] + trades_df["slippage_cost"] + trades_df["funding_cost"]
            )

            return trades_df

        except Exception as e:
            self.logger.error(f"Error calculating realistic costs: {e}")
            return trades_df

    def calculate_metrics(self, returns: pd.Series, trades_records: pd.DataFrame, close: pd.Series = None) -> Dict:
        try:
            # Basic returns metrics
            cumulative_returns = (1 + returns).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1

            # Sharpe Ratio (annualized, assuming daily returns)
            # AUDITORÍA FIX: Usar constante global en lugar de magic number
            excess_returns = returns - RISK_FREE_RATE_DAILY
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if excess_returns.std() > 0 else 0

            # Calmar Ratio
            max_dd = self._calculate_max_drawdown(cumulative_returns)
            calmar = total_return / max_dd if max_dd > 0 else 0

            # Win Rate
            if not trades_records.empty:
                win_rate = (trades_records["pnl"] > 0).mean()
                num_trades = len(trades_records)
            else:
                win_rate = 0
                num_trades = 0

            # Information Ratio (vs buy-and-hold)
            # AUDITORÍA FIX: Calcular buy-and-hold correctamente usando asset returns
            # El IR mide exceso de retorno vs benchmark / tracking error
            # Usamos buy-and-hold del mismo activo como benchmark
            if close is not None and len(close) > 1:
                bh_returns = close.pct_change().dropna()
                # Alinear longitudes
                min_len = min(len(returns), len(bh_returns))
                aligned_returns = returns.iloc[-min_len:]
                aligned_bh = bh_returns.iloc[-min_len:]
                active_return = aligned_returns - aligned_bh
                tracking_error = active_return.std()
                ir = (
                    active_return.mean() / tracking_error * np.sqrt(TRADING_DAYS_PER_YEAR)
                    if tracking_error > 0
                    else 0.0
                )
            else:
                ir = 0.0

            # Ulcer Index
            ulcer = self._calculate_ulcer_index(cumulative_returns)

            # Sortino Ratio
            # AUDITORÍA FIX: Validar que std > 0 para evitar división por cero
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
            sortino = (
                excess_returns.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)
                if len(downside_returns) > 0 and downside_std > 0
                else 0.0
            )

            # Profit Factor
            gross_profit = trades_records[trades_records["pnl"] > 0]["pnl"].sum() if not trades_records.empty else 0
            gross_loss = abs(trades_records[trades_records["pnl"] < 0]["pnl"].sum()) if not trades_records.empty else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # MAE/MFE Metrics from trade history (if available)
            avg_mae = (
                self.trade_history["mae"].mean()
                if not self.trade_history.empty and "mae" in self.trade_history.columns
                else 0.0
            )
            avg_mfe = (
                self.trade_history["mfe"].mean()
                if not self.trade_history.empty and "mfe" in self.trade_history.columns
                else 0.0
            )
            max_mae = (
                self.trade_history["mae"].max()
                if not self.trade_history.empty and "mae" in self.trade_history.columns
                else 0.0
            )
            max_mfe = (
                self.trade_history["mfe"].max()
                if not self.trade_history.empty and "mfe" in self.trade_history.columns
                else 0.0
            )

            return {
                "sharpe": round(sharpe, 3),
                "calmar": round(calmar, 3),
                "win_rate": round(win_rate, 3),
                "max_dd": round(max_dd, 3),
                "num_trades": num_trades,
                "ir": round(ir, 3),
                "ulcer": round(ulcer, 3),
                "sortino": round(sortino, 3),
                "profit_factor": round(profit_factor, 3),
                "total_return": round(total_return, 3),
                "avg_mae": round(avg_mae, 4),  # Average Maximum Adverse Excursion
                "avg_mfe": round(avg_mfe, 4),  # Average Maximum Favorable Excursion
                "max_mae": round(max_mae, 4),  # Maximum MAE across all trades
                "max_mfe": round(max_mfe, 4),  # Maximum MFE across all trades
            }

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {"error": str(e)}

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return abs(drawdown.min())

    def _calculate_ulcer_index(self, cumulative_returns: pd.Series) -> float:
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return np.sqrt((drawdown**2).mean())

    def _bayesian_optimize(self, strategy_class, train_data: Dict, param_space: Dict) -> Dict:
        try:
            # Define parameter space for optimization
            if not SKOPT_AVAILABLE:
                self.logger.warning("skopt not available - returning default parameters")
                return param_space
                
            space = []
            param_names = []

            for param_name, param_config in param_space.items():
                if param_config.get("type") == "int":
                    space.append(Integer(param_config["min"], param_config["max"], name=param_name))
                else:
                    space.append(Real(param_config["min"], param_config["max"], name=param_name))
                param_names.append(param_name)

            def objective(params):
                param_dict = dict(zip(param_names, params))
                result = self.run_simple_backtest(train_data, strategy_class, param_dict)
                if "error" in result:
                    return 0  # Return neutral score for errors
                return -result["metrics"]["sharpe"]  # Minimize negative Sharpe

            # Run optimization
            res = gp_minimize(objective, space, n_calls=50, random_state=42)

            # Return best parameters
            best_params = dict(zip(param_names, res.x))
            return best_params

        except Exception as e:
            self.logger.error(f"Error in Bayesian optimization: {e}")
            return param_space  # Return original params on error

    def _format_trades(self, trades_records: pd.DataFrame, df_index: pd.DatetimeIndex) -> List[Dict]:
        if trades_records.empty:
            return []

        trades = []
        for _, trade in trades_records.iterrows():
            # Map entry_idx to timestamp
            entry_idx = trade["entry_idx"]
            entry_idx_int = int(entry_idx)  # Convert float to int
            entry_timestamp = df_index[entry_idx_int] if entry_idx_int < len(df_index) else None

            trades.append(
                {
                    "timestamp": entry_timestamp,
                    "entry_price": trade["entry_price"],
                    "exit_price": trade["exit_price"],
                    "pnl_pct": trade["return"],  # VectorBT return is already in decimal format
                    "score": 4,  # Placeholder, should be calculated by strategy
                    # 0=long, 1=short in VectorBT
                    "entry_type": "long" if trade["direction"] == 0 else "short",
                    "reason_exit": "target",  # Placeholder
                }
            )

        return trades

    def list_available_strategies(self) -> List[str]:
        """List all available strategies."""
        try:
            from strategies import list_available_strategies
            return list_available_strategies()
        except ImportError:
            self.logger.error("Could not import strategies module")
            return []
