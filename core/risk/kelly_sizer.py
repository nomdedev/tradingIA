"""
Kelly Criterion Position Sizing for Optimal Capital Allocation

This module implements the Kelly Criterion formula for optimal position sizing
in trading strategies. The Kelly Criterion maximizes long-term growth by
balancing expected returns against risk.

Formula: f = (bp - q) / b
Where:
- f = fraction of capital to risk
- b = odds (win_loss_ratio - 1)
- p = probability of winning
- q = probability of losing (1 - p)

ÁREA 3 FIX: Agregado ajuste por régimen de mercado y correlación serial.

Author: AI Assistant
Date: 16 de Noviembre, 2025 (Updated: 13 de Enero, 2026)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ÁREA 3: Import análisis de régimen
try:
    from src.analysis_engines import AnalysisEngines
    ANALYSIS_ENGINES_AVAILABLE = True
except ImportError:
    ANALYSIS_ENGINES_AVAILABLE = False
    logger.warning("AnalysisEngines not available - regime adjustment disabled")


# ÁREA 3: Multiplicadores por régimen de mercado
REGIME_MULTIPLIERS = {
    'bull': 1.0,      # Kelly completo en tendencia alcista
    'bear': 0.5,      # Mitad de Kelly en tendencia bajista
    'chop': 0.3,      # 30% en mercado lateral (sideways)
    'sideways': 0.3,  # Alias para chop
    'high_vol': 0.25, # 25% en alta volatilidad
    'unknown': 0.5,   # Conservador por defecto
}

# ÁREA 3: Penalización por rachas (correlación serial)
STREAK_PENALTIES = {
    5: 0.50,  # 5+ wins seguidos: reducir 50%
    4: 0.35,  # 4 wins seguidos: reducir 35%
    3: 0.20,  # 3 wins seguidos: reducir 20%
    2: 0.10,  # 2 wins seguidos: reducir 10%
    1: 0.0,   # 1 win: sin penalización
    0: 0.0,   # Sin racha: sin penalización
}


@dataclass
class KellyResult:
    """Result of Kelly calculation"""

    kelly_fraction: float
    kelly_full: float
    kelly_half: float  # Conservative (50% of full Kelly)
    kelly_quarter: float  # Very conservative (25% of full Kelly)
    optimal_position_size: float
    expected_growth_rate: float
    confidence_interval: Tuple[float, float]


class KellyPositionSizer:
    """
    Kelly Criterion position sizing calculator.

    Provides optimal position sizing based on win rate and win/loss ratio.
    Includes risk adjustments and market impact considerations.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_position_pct: float = 0.10,  # Max 10% of capital
        min_position_pct: float = 0.001,  # Min 0.1% of capital
        volatility_adjustment: bool = True,
    ):
        """
        Initialize Kelly position sizer.

        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25-1.0, default 0.5)
            max_position_pct: Maximum position size as % of capital
            min_position_pct: Minimum position size as % of capital
            volatility_adjustment: Adjust for market volatility
        """
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.volatility_adjustment = volatility_adjustment

        # Validation
        if not 0.1 <= kelly_fraction <= 1.0:
            raise ValueError("Kelly fraction must be between 0.1 and 1.0")

        logger.info(f"Kelly Position Sizer initialized: fraction={kelly_fraction}, " f"max_pos={max_position_pct:.1%}")

    def calculate_kelly_fraction(
        self, win_rate: float, win_loss_ratio: float, market_impact_pct: float = 0.0
    ) -> KellyResult:
        """
        Calculate optimal Kelly fraction.

        Args:
            win_rate: Probability of winning (0.0-1.0)
            win_loss_ratio: Average win / Average loss ratio
            market_impact_pct: Estimated market impact cost (0.0-1.0)

        Returns:
            KellyResult with all sizing calculations
        """
        # Input validation
        if not 0.0 <= win_rate <= 1.0:
            raise ValueError("Win rate must be between 0.0 and 1.0")

        if win_loss_ratio <= 0:
            # No edge - return zero position
            return KellyResult(
                kelly_fraction=0.0,
                kelly_full=0.0,
                kelly_half=0.0,
                kelly_quarter=0.0,
                optimal_position_size=0.0,
                expected_growth_rate=0.0,
                confidence_interval=(0.0, 0.0),
            )

        # Adjust win/loss ratio for market impact costs
        adjusted_win_loss_ratio = win_loss_ratio * (1.0 - market_impact_pct)

        if adjusted_win_loss_ratio <= 1.0:
            # Expected value is negative after costs
            return KellyResult(
                kelly_fraction=0.0,
                kelly_full=0.0,
                kelly_half=0.0,
                kelly_quarter=0.0,
                optimal_position_size=0.0,
                expected_growth_rate=0.0,
                confidence_interval=(0.0, 0.0),
            )

        # Kelly formula: f = (bp - q) / b
        # Where: b = win_loss_ratio (odds), p = win_rate, q = 1 - p
        # Simplifica a: f = p - q/b = p - (1-p)/b
        p = win_rate
        q = 1.0 - win_rate
        b = adjusted_win_loss_ratio

        # Fórmula correcta de Kelly
        kelly_full = (p * b - q) / b
        # Equivalente: kelly_full = p - q/b

        # Handle edge cases
        if not np.isfinite(kelly_full) or kelly_full < 0:
            kelly_full = 0.0

        # Conservative fractions
        kelly_half = kelly_full * 0.5
        kelly_quarter = kelly_full * 0.25

        # Apply user-specified fraction
        kelly_fraction = kelly_full * self.kelly_fraction

        # Calculate expected growth rate
        expected_growth_rate = self._calculate_expected_growth(kelly_fraction, win_rate, adjusted_win_loss_ratio)

        # Calculate confidence interval (simplified approximation)
        confidence_interval = self._calculate_confidence_interval(kelly_fraction, win_rate, adjusted_win_loss_ratio)

        return KellyResult(
            kelly_fraction=kelly_fraction,
            kelly_full=kelly_full,
            kelly_half=kelly_half,
            kelly_quarter=kelly_quarter,
            optimal_position_size=kelly_fraction,  # Will be scaled by capital later
            expected_growth_rate=expected_growth_rate,
            confidence_interval=confidence_interval,
        )

    def calculate_position_size(
        self,
        capital: float,
        win_rate: float,
        win_loss_ratio: float,
        current_volatility: float = 0.0,
        market_impact_pct: float = 0.0,
    ) -> Dict:
        """
        Calculate actual position size for given capital.

        Args:
            capital: Available capital
            win_rate: Probability of winning
            win_loss_ratio: Average win / Average loss ratio
            current_volatility: Current market volatility (0.0-1.0)
            market_impact_pct: Estimated market impact cost

        Returns:
            Dictionary with position sizing details (includes non-float values)
        """
        # Get Kelly calculation
        kelly_result = self.calculate_kelly_fraction(win_rate, win_loss_ratio, market_impact_pct)

        # Adjust for volatility (reduce position in high volatility)
        volatility_multiplier = 1.0
        if self.volatility_adjustment and current_volatility > 0:
            # Use exponential decay for smoother adjustment
            # High volatility (0.5+) reduces position significantly
            # Low volatility (<0.1) has minimal impact
            volatility_multiplier = np.exp(-2.0 * current_volatility)
            # Ensure minimum of 0.3 (never reduce more than 70%)
            volatility_multiplier = max(0.3, min(1.0, volatility_multiplier))

        # Calculate base position size
        base_position_size = kelly_result.kelly_fraction * capital * volatility_multiplier

        # Apply bounds
        max_position = capital * self.max_position_pct
        min_position = capital * self.min_position_pct

        optimal_position_size = np.clip(base_position_size, min_position, max_position)

        # Calculate risk metrics
        position_pct = optimal_position_size / capital
        risk_per_trade = position_pct * (1.0 / win_loss_ratio)  # Risk = position / win_loss_ratio

        return {
            "position_size": optimal_position_size,
            "position_pct": position_pct,
            "risk_per_trade_pct": risk_per_trade,
            "kelly_fraction": kelly_result.kelly_fraction,
            "kelly_full": kelly_result.kelly_full,
            "expected_growth_rate": kelly_result.expected_growth_rate,
            "confidence_interval": kelly_result.confidence_interval,
            "volatility_adjustment": volatility_multiplier,
            "market_impact_adjusted": market_impact_pct > 0,
        }

    def optimize_kelly_fraction(self, historical_trades: pd.DataFrame, optimization_metric: str = "sharpe") -> float:
        """
        Optimize Kelly fraction based on historical performance.

        Args:
            historical_trades: DataFrame with trade results
            optimization_metric: 'sharpe', 'sortino', 'max_dd', 'total_return'

        Returns:
            Optimal Kelly fraction
        """
        if historical_trades.empty:
            return self.kelly_fraction  # Return default

        # Calculate win rate and win/loss ratio from historical data
        wins = historical_trades[historical_trades["pnl"] > 0]
        losses = historical_trades[historical_trades["pnl"] < 0]

        if len(losses) == 0:
            win_rate = 1.0
            win_loss_ratio = float("inf")
        else:
            win_rate = len(wins) / len(historical_trades)
            avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
            avg_loss = abs(losses["pnl"].mean())
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

        # Test different Kelly fractions
        kelly_fractions = np.linspace(0.1, 1.0, 10)
        best_metric = -float("inf")
        best_fraction = 0.5

        for fraction in kelly_fractions:
            temp_sizer = KellyPositionSizer(kelly_fraction=fraction)

            # Simulate portfolio with this Kelly fraction
            portfolio_values = self._simulate_portfolio(historical_trades, temp_sizer, initial_capital=10000)

            # Calculate optimization metric (con risk-free rate)
            if optimization_metric == "sharpe":
                returns = pd.Series(portfolio_values).pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    rf_daily = 0.04 / 252
                    excess_returns = returns - rf_daily
                    metric = (
                        (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
                    )
                else:
                    metric = 0
            elif optimization_metric == "sortino":
                returns = pd.Series(portfolio_values).pct_change().dropna()
                rf_daily = 0.04 / 252
                excess_returns = returns - rf_daily
                downside_returns = excess_returns[excess_returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    metric = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
                else:
                    metric = excess_returns.mean() * np.sqrt(252) if excess_returns.mean() > 0 else 0
            elif optimization_metric == "max_dd":
                peak = pd.Series(portfolio_values).expanding().max()
                drawdown = (pd.Series(portfolio_values) - peak) / peak
                metric = -drawdown.min()  # Negative because we want to minimize drawdown
            elif optimization_metric == "total_return":
                metric = (portfolio_values[-1] / portfolio_values[0]) - 1
            else:
                metric = 0

            if metric > best_metric:
                best_metric = metric
                best_fraction = fraction

        logger.info(
            f"Optimized Kelly fraction: {best_fraction:.2f} " f"(metric: {optimization_metric} = {best_metric:.3f})"
        )

        return best_fraction

    def _calculate_expected_growth(self, kelly_fraction: float, win_rate: float, win_loss_ratio: float) -> float:
        """Calculate expected growth rate with Kelly sizing"""
        if kelly_fraction <= 0:
            return 0.0

        # Expected growth rate formula
        growth_rate = win_rate * np.log(1 + kelly_fraction * (win_loss_ratio - 1)) + (1 - win_rate) * np.log(
            1 - kelly_fraction
        )

        return growth_rate

    def _calculate_confidence_interval(
        self, kelly_fraction: float, win_rate: float, win_loss_ratio: float, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for Kelly fraction (simplified)"""
        if kelly_fraction <= 0:
            return (0.0, 0.0)

        # Simplified confidence interval based on binomial distribution
        n = 100  # Assume 100 trades for confidence calculation
        variance = (win_rate * (1 - win_rate)) / n

        # Standard error of Kelly fraction
        se = np.sqrt(variance) * 2  # Approximation

        margin = se * 1.96  # 95% confidence
        lower = max(0, kelly_fraction - margin)
        upper = kelly_fraction + margin

        return (lower, upper)

    def _simulate_portfolio(
        self, trades: pd.DataFrame, sizer: "KellyPositionSizer", initial_capital: float = 10000
    ) -> List[float]:
        """Simulate portfolio growth with Kelly sizing"""
        capital = initial_capital
        portfolio_values = [capital]

        for _, trade in trades.iterrows():
            # Calculate position size using Kelly
            sizing_result = sizer.calculate_position_size(
                capital=capital,
                win_rate=0.5,  # Simplified assumption
                win_loss_ratio=2.0,  # Simplified assumption
                current_volatility=0.2,
            )

            position_size = sizing_result["position_size"]
            pnl = trade["pnl"]

            # Update capital
            capital += pnl
            portfolio_values.append(capital)

        return portfolio_values

    def get_risk_warnings(self, kelly_result: KellyResult) -> List[str]:
        """Get risk warnings based on Kelly calculation"""
        warnings = []

        if kelly_result.kelly_full > 1.0:
            warnings.append("⚠️ Kelly fraction > 100% - High risk of ruin")

        if kelly_result.kelly_full > 0.5:
            warnings.append("⚠️ Kelly fraction > 50% - Aggressive sizing")

        if kelly_result.kelly_fraction < 0.1:
            warnings.append("⚠️ Kelly fraction < 10% - Very conservative")

        if kelly_result.expected_growth_rate < 0:
            warnings.append("❌ Negative expected growth - Avoid trading")

        return warnings

    # =========================================================================
    # ÁREA 3: Métodos de ajuste por régimen y correlación serial
    # =========================================================================
    
    def calculate_regime_adjusted_kelly(
        self,
        win_rate: float,
        win_loss_ratio: float,
        price_data: pd.DataFrame = None,
        trade_history: pd.DataFrame = None,
        market_impact_pct: float = 0.0,
    ) -> Dict:
        """
        ÁREA 3 FIX: Kelly ajustado por régimen de mercado y correlación serial.
        
        Args:
            win_rate: Tasa de éxito (0.0-1.0)
            win_loss_ratio: Ratio ganancia/pérdida promedio
            price_data: DataFrame con precios para detectar régimen (opcional)
            trade_history: DataFrame con historial de trades para detectar rachas
            market_impact_pct: Costo estimado de market impact
            
        Returns:
            Dict con kelly ajustado y detalles de ajustes
        """
        # 1. Calcular Kelly base
        kelly_result = self.calculate_kelly_fraction(win_rate, win_loss_ratio, market_impact_pct)
        kelly_base = kelly_result.kelly_fraction
        
        if kelly_base <= 0:
            return {
                "kelly_final": 0.0,
                "kelly_base": kelly_base,
                "regime": "unknown",
                "regime_multiplier": 0.0,
                "streak_penalty": 0.0,
                "adjustments": ["Kelly base <= 0, no trade recommended"],
            }
        
        adjustments = []
        
        # 2. Detectar y ajustar por régimen de mercado
        regime = "unknown"
        regime_multiplier = REGIME_MULTIPLIERS["unknown"]
        
        if price_data is not None and ANALYSIS_ENGINES_AVAILABLE:
            try:
                regime = self._detect_current_regime(price_data)
                regime_multiplier = REGIME_MULTIPLIERS.get(regime, 0.5)
                adjustments.append(f"Régimen detectado: {regime} (×{regime_multiplier})")
            except Exception as e:
                logger.warning(f"Error detecting regime: {e}")
                adjustments.append(f"Error en detección de régimen: {str(e)}")
        else:
            adjustments.append("Sin datos de precio - usando régimen unknown")
        
        kelly_after_regime = kelly_base * regime_multiplier
        
        # 3. Detectar y ajustar por correlación serial (rachas)
        streak_penalty = 0.0
        consecutive_wins = 0
        
        if trade_history is not None and len(trade_history) >= 2:
            consecutive_wins = self._count_consecutive_wins(trade_history)
            streak_penalty = STREAK_PENALTIES.get(min(consecutive_wins, 5), 0.0)
            
            if streak_penalty > 0:
                adjustments.append(
                    f"Racha detectada: {consecutive_wins} wins seguidos "
                    f"(penalización: -{streak_penalty*100:.0f}%)"
                )
        
        kelly_after_streak = kelly_after_regime * (1 - streak_penalty)
        
        # 4. Aplicar límites finales
        kelly_final = max(0.0, min(kelly_after_streak, self.max_position_pct))
        
        # 5. Generar warnings si aplica
        warnings = []
        if regime in ["bear", "chop", "sideways"]:
            warnings.append(f"⚠️ Mercado {regime} - Kelly reducido significativamente")
        if consecutive_wins >= 3:
            warnings.append(f"⚠️ Racha de {consecutive_wins} wins - Posible reversión a media")
        if kelly_final < kelly_base * 0.3:
            warnings.append("⚠️ Kelly final muy reducido por ajustes")
        
        return {
            "kelly_final": kelly_final,
            "kelly_base": kelly_base,
            "kelly_full": kelly_result.kelly_full,
            "regime": regime,
            "regime_multiplier": regime_multiplier,
            "kelly_after_regime": kelly_after_regime,
            "consecutive_wins": consecutive_wins,
            "streak_penalty": streak_penalty,
            "kelly_after_streak": kelly_after_streak,
            "adjustments": adjustments,
            "warnings": warnings,
            "expected_growth_rate": kelly_result.expected_growth_rate,
        }
    
    def _detect_current_regime(self, price_data: pd.DataFrame) -> str:
        """
        Detectar régimen actual usando HMM.
        
        Args:
            price_data: DataFrame con columna 'Close'
            
        Returns:
            str: 'bull', 'bear', 'chop', o 'unknown'
        """
        if not ANALYSIS_ENGINES_AVAILABLE:
            return "unknown"
        
        try:
            analyzer = AnalysisEngines()
            result = analyzer.detect_regime_hmm(price_data)
            
            if isinstance(result, dict) and "error" in result:
                logger.warning(f"Regime detection error: {result['error']}")
                return "unknown"
            
            # Obtener último régimen
            if "regime_name" in result.columns:
                last_regime = result["regime_name"].dropna().iloc[-1]
                return last_regime if last_regime else "unknown"
            
            return "unknown"
            
        except Exception as e:
            logger.warning(f"Error in regime detection: {e}")
            return "unknown"
    
    def _count_consecutive_wins(self, trade_history: pd.DataFrame) -> int:
        """
        Contar wins consecutivos desde el último trade.
        
        Args:
            trade_history: DataFrame con columna 'pnl'
            
        Returns:
            int: Número de wins consecutivos
        """
        if trade_history.empty or 'pnl' not in trade_history.columns:
            return 0
        
        # Ordenar por fecha si hay columna de timestamp
        if 'timestamp' in trade_history.columns:
            trade_history = trade_history.sort_values('timestamp')
        
        # Contar wins consecutivos desde el final
        pnl_values = trade_history['pnl'].values
        consecutive = 0
        
        for pnl in reversed(pnl_values):
            if pnl > 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def calculate_adaptive_lookback(
        self,
        trade_history: pd.DataFrame,
        min_trades: int = 20,
        max_trades: int = 200,
        volatility_window: int = 20,
    ) -> int:
        """
        ÁREA 3 FIX: Calcular lookback adaptativo basado en régimen.
        
        En alta volatilidad: lookback corto (más reciente)
        En baja volatilidad: lookback largo (más trades)
        
        Args:
            trade_history: DataFrame con historial de trades
            min_trades: Mínimo de trades a considerar
            max_trades: Máximo de trades a considerar
            volatility_window: Ventana para calcular volatilidad
            
        Returns:
            int: Número de trades para lookback
        """
        if trade_history.empty or 'pnl' not in trade_history.columns:
            return min_trades
        
        n_trades = len(trade_history)
        
        if n_trades < min_trades:
            return n_trades  # Usar todos los disponibles
        
        # Calcular volatilidad de PnL reciente
        recent_pnl = trade_history['pnl'].tail(volatility_window)
        if len(recent_pnl) < 5:
            return min(n_trades, min_trades)
        
        pnl_volatility = recent_pnl.std()
        pnl_mean = abs(recent_pnl.mean()) if recent_pnl.mean() != 0 else 1
        
        # Coeficiente de variación normalizado
        cv = pnl_volatility / pnl_mean if pnl_mean > 0 else 1.0
        
        # Mapear CV a lookback
        # CV alto (>1.0) -> lookback corto
        # CV bajo (<0.5) -> lookback largo
        if cv > 1.0:
            lookback = min_trades  # Alta volatilidad: usar datos recientes
        elif cv < 0.3:
            lookback = max_trades  # Baja volatilidad: usar más datos
        else:
            # Interpolación lineal
            lookback = int(min_trades + (max_trades - min_trades) * (1 - cv))
        
        return min(n_trades, max(min_trades, lookback))
    
    def get_statistics_with_adaptive_lookback(
        self,
        trade_history: pd.DataFrame,
    ) -> Dict:
        """
        Obtener estadísticas usando lookback adaptativo.
        
        Returns:
            Dict con win_rate, win_loss_ratio, y lookback usado
        """
        if trade_history.empty or 'pnl' not in trade_history.columns:
            return {
                "win_rate": 0.5,
                "win_loss_ratio": 1.0,
                "lookback_used": 0,
                "total_trades": 0,
            }
        
        # Calcular lookback adaptativo
        lookback = self.calculate_adaptive_lookback(trade_history)
        
        # Usar últimos N trades
        recent_trades = trade_history.tail(lookback)
        
        # Calcular estadísticas
        wins = recent_trades[recent_trades['pnl'] > 0]
        losses = recent_trades[recent_trades['pnl'] < 0]
        
        win_rate = len(wins) / len(recent_trades) if len(recent_trades) > 0 else 0.5
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 1
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        return {
            "win_rate": win_rate,
            "win_loss_ratio": win_loss_ratio,
            "lookback_used": lookback,
            "total_trades": len(trade_history),
            "trades_analyzed": len(recent_trades),
            "n_wins": len(wins),
            "n_losses": len(losses),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }

