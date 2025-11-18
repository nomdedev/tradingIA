"""
Market Impact Model for Realistic Order Execution Simulation

Implements the Almgren-Chriss model for permanent and temporary market impact,
plus bid-ask spread costs. Critical for realistic backtesting.

References:
- Almgren & Chriss (2000): "Optimal Execution of Portfolio Transactions"
- Bertsimas & Lo (1998): "Optimal control of execution costs"
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging


class MarketImpactModel:
    """
    Calculates realistic market impact costs based on:
    1. Order size relative to average volume (permanent impact)
    2. Urgency of execution (temporary impact)
    3. Bid-ask spread
    4. Current volatility regime
    5. Time of day (liquidity patterns)
    """

    def __init__(
        self,
        permanent_impact_factor: float = 0.1,
        temporary_impact_factor: float = 0.5,
        spread_factor: float = 0.5,
        volatility_scaling: bool = True
    ):
        """
        Initialize market impact model.

        Args:
            permanent_impact_factor: Coefficient for permanent impact (default 0.1)
            temporary_impact_factor: Coefficient for temporary impact (default 0.5)
            spread_factor: Multiplier for bid-ask spread (0.5 = mid-market execution)
            volatility_scaling: Scale impact by current volatility regime
        """
        self.permanent_factor = permanent_impact_factor
        self.temporary_factor = temporary_impact_factor
        self.spread_factor = spread_factor
        self.volatility_scaling = volatility_scaling
        self.logger = logging.getLogger(__name__)

    def calculate_impact(
        self,
        order_size: float,
        price: float,
        avg_volume: float,
        volatility: float,
        bid_ask_spread: float,
        time_of_day: Optional[int] = None,
        urgency: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate total market impact cost.

        Args:
            order_size: Number of units to trade
            price: Current market price
            avg_volume: Average trading volume (same timeframe)
            volatility: Current volatility (ATR or stddev)
            bid_ask_spread: Current bid-ask spread (absolute)
            time_of_day: Hour of day (0-23) for liquidity adjustment
            urgency: Execution urgency multiplier (0.5-2.0)

        Returns:
            Dictionary with impact breakdown:
            - permanent_impact: Price change that persists
            - temporary_impact: Execution slippage
            - spread_cost: Bid-ask crossing cost
            - liquidity_penalty: Additional cost in low-liquidity periods
            - total_impact_pct: Total as % of price
            - total_impact_dollars: Total in dollar terms
        """
        # Validate inputs
        if order_size <= 0 or price <= 0 or avg_volume <= 0:
            self.logger.warning(f"Invalid inputs: size={order_size}, price={price}, vol={avg_volume}")
            return self._zero_impact()

        # 1. PERMANENT IMPACT (Square-root model)
        # Impact scales with sqrt(order_size / volume)
        volume_ratio = order_size / avg_volume
        
        # Almgren-Chriss square root model
        permanent_impact = self.permanent_factor * volatility * np.sqrt(volume_ratio)

        # 2. TEMPORARY IMPACT (Linear in volume ratio)
        # Temporary impact depends on execution speed
        temporary_impact = self.temporary_factor * volatility * volume_ratio * urgency

        # 3. SPREAD COST
        # Cost of crossing bid-ask spread
        spread_pct = bid_ask_spread / price
        spread_cost = self.spread_factor * spread_pct

        # 4. LIQUIDITY PENALTY (Time-of-day adjustment)
        liquidity_penalty = 0.0
        if time_of_day is not None:
            liquidity_penalty = self._calculate_liquidity_penalty(time_of_day)

        # 5. VOLATILITY SCALING
        vol_multiplier = 1.0
        if self.volatility_scaling:
            # Scale impact by volatility regime (percentile-based)
            vol_multiplier = self._calculate_volatility_multiplier(volatility)

        # Total impact
        total_impact_pct = (
            permanent_impact + temporary_impact + spread_cost + liquidity_penalty
        ) * vol_multiplier

        # Convert to dollars
        order_value = order_size * price
        total_impact_dollars = order_value * total_impact_pct

        return {
            'permanent_impact': permanent_impact,
            'temporary_impact': temporary_impact,
            'spread_cost': spread_cost,
            'liquidity_penalty': liquidity_penalty,
            'vol_multiplier': vol_multiplier,
            'total_impact_pct': total_impact_pct,
            'total_impact_dollars': total_impact_dollars,
            'volume_ratio': volume_ratio
        }

    def calculate_execution_price(
        self,
        side: str,
        price: float,
        impact_pct: float
    ) -> float:
        """
        Calculate actual execution price including impact.

        Args:
            side: 'buy' or 'sell'
            price: Market price
            impact_pct: Total impact as percentage

        Returns:
            Execution price after impact
        """
        if side.lower() == 'buy':
            # Buying pushes price up
            execution_price = price * (1 + impact_pct)
        elif side.lower() == 'sell':
            # Selling pushes price down
            execution_price = price * (1 - impact_pct)
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'")

        return execution_price

    def _calculate_liquidity_penalty(self, hour: int) -> float:
        """
        Calculate liquidity penalty based on time of day.

        Assumes lower liquidity during:
        - Night hours (0-4 UTC): +100% impact
        - Early morning (4-8 UTC): +50% impact
        - Lunch (11-13 UTC): +30% impact
        - After hours (20-24 UTC): +70% impact

        Args:
            hour: Hour of day (0-23)

        Returns:
            Additional impact percentage
        """
        # Liquidity profile (lower = better liquidity)
        liquidity_profile = {
            0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0,  # Night: very low liquidity
            4: 0.5, 5: 0.5, 6: 0.4, 7: 0.3,  # Early morning: improving
            8: 0.1, 9: 0.0, 10: 0.0,          # Morning: best liquidity
            11: 0.3, 12: 0.3, 13: 0.2,        # Lunch: moderate
            14: 0.0, 15: 0.0, 16: 0.0,        # Afternoon: best liquidity
            17: 0.1, 18: 0.2, 19: 0.3,        # Late afternoon
            20: 0.7, 21: 0.8, 22: 0.9, 23: 1.0  # After hours: low liquidity
        }

        penalty = liquidity_profile.get(hour, 0.0)
        return penalty * 0.001  # Scale to reasonable percentage

    def _calculate_volatility_multiplier(self, current_vol: float) -> float:
        """
        Scale impact by volatility regime.

        Higher volatility = wider spreads and more impact.

        Args:
            current_vol: Current volatility (ATR or similar)

        Returns:
            Multiplier (1.0 = normal, >1.0 = high vol)
        """
        # This would ideally use historical percentile
        # For now, simple scaling
        # Assumes current_vol is normalized (e.g., ATR as % of price)
        
        if current_vol < 0.01:  # Very low vol (<1%)
            return 0.8
        elif current_vol < 0.02:  # Normal vol (1-2%)
            return 1.0
        elif current_vol < 0.03:  # Elevated vol (2-3%)
            return 1.3
        else:  # High vol (>3%)
            return 1.6

    def _zero_impact(self) -> Dict[str, float]:
        """Return zero impact dict for error cases"""
        return {
            'permanent_impact': 0.0,
            'temporary_impact': 0.0,
            'spread_cost': 0.0,
            'liquidity_penalty': 0.0,
            'vol_multiplier': 1.0,
            'total_impact_pct': 0.0,
            'total_impact_dollars': 0.0,
            'volume_ratio': 0.0
        }

    def estimate_optimal_order_size(
        self,
        available_capital: float,
        price: float,
        avg_volume: float,
        volatility: float,
        max_impact_pct: float = 0.005  # 0.5% max impact
    ) -> Dict[str, float]:
        """
        Estimate optimal order size to keep impact below threshold.

        Uses binary search to find largest order that stays under max_impact.

        Args:
            available_capital: Total capital available
            price: Current price
            avg_volume: Average volume
            volatility: Current volatility
            max_impact_pct: Maximum acceptable impact (default 0.5%)

        Returns:
            Dictionary with:
            - optimal_size: Order size in units
            - optimal_value: Order value in dollars
            - expected_impact_pct: Expected impact
            - capital_utilization: % of available capital used
        """
        max_size = available_capital / price

        # Binary search for optimal size
        low, high = 0.0, max_size
        optimal_size = 0.0
        
        for _ in range(20):  # 20 iterations = ~0.0001% precision
            mid = (low + high) / 2
            
            # Calculate impact at this size
            impact = self.calculate_impact(
                order_size=mid,
                price=price,
                avg_volume=avg_volume,
                volatility=volatility,
                bid_ask_spread=price * 0.001  # Assume 0.1% spread
            )
            
            if impact['total_impact_pct'] <= max_impact_pct:
                optimal_size = mid
                low = mid
            else:
                high = mid

        optimal_value = optimal_size * price
        capital_utilization = optimal_value / available_capital

        # Get final impact
        final_impact = self.calculate_impact(
            order_size=optimal_size,
            price=price,
            avg_volume=avg_volume,
            volatility=volatility,
            bid_ask_spread=price * 0.001
        )

        return {
            'optimal_size': optimal_size,
            'optimal_value': optimal_value,
            'expected_impact_pct': final_impact['total_impact_pct'],
            'capital_utilization': capital_utilization
        }


class VolumeProfileAnalyzer:
    """
    Analyzes historical volume patterns to estimate average volume
    for different timeframes and times of day.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_average_volume(
        self,
        df: pd.DataFrame,
        lookback_periods: int = 20
    ) -> pd.Series:
        """
        Calculate rolling average volume.

        Args:
            df: DataFrame with 'volume' column
            lookback_periods: Number of periods for rolling average

        Returns:
            Series with average volume
        """
        if 'volume' not in df.columns:
            self.logger.error("DataFrame missing 'volume' column")
            return pd.Series(index=df.index, data=0)

        avg_volume = df['volume'].rolling(window=lookback_periods, min_periods=1).mean()
        return avg_volume

    def calculate_volume_profile_by_hour(
        self,
        df: pd.DataFrame
    ) -> Dict[int, float]:
        """
        Calculate average volume by hour of day.

        Args:
            df: DataFrame with 'volume' column and datetime index

        Returns:
            Dictionary mapping hour (0-23) to average volume
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.error("DataFrame index must be DatetimeIndex")
            return {}

        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour

        volume_by_hour = df_copy.groupby('hour')['volume'].mean().to_dict()
        return volume_by_hour


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    impact_model = MarketImpactModel()

    # Example: Calculate impact for a BTC trade
    print("=" * 60)
    print("Market Impact Model - Example Calculation")
    print("=" * 60)

    # Scenario: Buy $50,000 worth of BTC
    price = 50000  # BTC price
    order_value = 50000  # $50k order
    order_size = order_value / price  # 1 BTC
    avg_volume = 10  # 10 BTC average volume per bar
    volatility = 0.02  # 2% volatility (ATR)
    bid_ask_spread = 50  # $50 spread

    impact = impact_model.calculate_impact(
        order_size=order_size,
        price=price,
        avg_volume=avg_volume,
        volatility=volatility,
        bid_ask_spread=bid_ask_spread,
        time_of_day=14,  # 2 PM (high liquidity)
        urgency=1.0
    )

    print(f"\nOrder Details:")
    print(f"  Size: {order_size:.4f} BTC")
    print(f"  Value: ${order_value:,.2f}")
    print(f"  Price: ${price:,.2f}")
    print(f"  Volume Ratio: {impact['volume_ratio']:.2%}")

    print(f"\nImpact Breakdown:")
    print(f"  Permanent Impact: {impact['permanent_impact']:.4%}")
    print(f"  Temporary Impact: {impact['temporary_impact']:.4%}")
    print(f"  Spread Cost: {impact['spread_cost']:.4%}")
    print(f"  Liquidity Penalty: {impact['liquidity_penalty']:.4%}")
    print(f"  Volatility Multiplier: {impact['vol_multiplier']:.2f}x")

    print(f"\nTotal Impact:")
    print(f"  Percentage: {impact['total_impact_pct']:.4%}")
    print(f"  Dollars: ${impact['total_impact_dollars']:,.2f}")

    # Calculate execution price
    exec_price = impact_model.calculate_execution_price(
        side='buy',
        price=price,
        impact_pct=impact['total_impact_pct']
    )
    print(f"\nExecution Price: ${exec_price:,.2f} (vs ${price:,.2f} mid)")
    print(f"Slippage: ${exec_price - price:,.2f} ({(exec_price/price - 1):.4%})")

    # Optimal sizing
    print("\n" + "=" * 60)
    print("Optimal Order Sizing")
    print("=" * 60)

    optimal = impact_model.estimate_optimal_order_size(
        available_capital=100000,
        price=price,
        avg_volume=avg_volume,
        volatility=volatility,
        max_impact_pct=0.005  # 0.5% max impact
    )

    print(f"\nAvailable Capital: $100,000")
    print(f"Optimal Order Size: {optimal['optimal_size']:.4f} BTC")
    print(f"Optimal Order Value: ${optimal['optimal_value']:,.2f}")
    print(f"Expected Impact: {optimal['expected_impact_pct']:.4%}")
    print(f"Capital Utilization: {optimal['capital_utilization']:.2%}")

    print("\n" + "=" * 60)
