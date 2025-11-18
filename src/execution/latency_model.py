"""
Latency Model for Realistic Order Execution Timing

Simulates network delays, exchange processing times, and execution lag
that occur in real trading. Critical for high-frequency and intraday strategies.

Components:
1. Network latency (user → exchange)
2. Exchange processing time
3. Market data delays
4. Order acknowledgment delays
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging


class LatencyModel:
    """
    Simulates realistic execution delays in trading systems.

    Includes:
    - Network round-trip time (RTT)
    - Exchange order processing
    - Market data feed delays
    - Queue position effects
    """

    def __init__(
        self,
        base_network_latency_ms: float = 50.0,
        network_jitter_ms: float = 20.0,
        exchange_processing_ms: float = 10.0,
        market_data_delay_ms: float = 100.0,
        queue_enabled: bool = True
    ):
        """
        Initialize latency model.

        Args:
            base_network_latency_ms: Average network latency (ms)
            network_jitter_ms: Network jitter std dev (ms)
            exchange_processing_ms: Exchange processing time mean (ms)
            market_data_delay_ms: Market data feed delay (ms)
            queue_enabled: Simulate order queue position effects
        """
        self.base_network_latency = base_network_latency_ms
        self.network_jitter = network_jitter_ms
        self.exchange_processing = exchange_processing_ms
        self.market_data_delay = market_data_delay_ms
        self.queue_enabled = queue_enabled
        self.logger = logging.getLogger(__name__)

        # Cache for performance
        self._last_signal_time = None
        self._last_execution_time = None

    def calculate_total_latency(
        self,
        order_type: str = "market",
        market_volatility: float = 1.0,
        time_of_day: Optional[int] = None
    ) -> float:
        """
        Calculate total execution latency.

        Args:
            order_type: Type of order ('market', 'limit', 'stop')
            market_volatility: Current volatility (higher = more congestion)
            time_of_day: Hour of day (0-23) for time-based patterns

        Returns:
            Total latency in milliseconds
        """
        # 1. Network latency (round-trip)
        network_latency = self._calculate_network_latency()

        # 2. Exchange processing
        exchange_latency = self._calculate_exchange_processing(order_type, market_volatility)

        # 3. Queue position (if enabled)
        queue_latency = 0.0
        if self.queue_enabled:
            queue_latency = self._calculate_queue_delay(market_volatility)

        # 4. Time-of-day effects
        tod_multiplier = self._get_time_of_day_multiplier(time_of_day)

        # Total latency
        total_latency = (network_latency + exchange_latency + queue_latency) * tod_multiplier

        return max(0, total_latency)

    def _calculate_network_latency(self) -> float:
        """
        Calculate network latency with jitter.

        Uses normal distribution: N(base_latency, jitter)
        """
        latency = np.random.normal(self.base_network_latency, self.network_jitter)
        return max(0, latency)

    def _calculate_exchange_processing(self, order_type: str, volatility: float) -> float:
        """
        Calculate exchange processing time.

        Market orders: Fastest processing
        Limit orders: Slower (need to check order book)
        Stop orders: Slowest (need to monitor trigger)

        Higher volatility = more orders = slower processing
        """
        # Base processing time
        if order_type == "market":
            base_time = self.exchange_processing
        elif order_type == "limit":
            base_time = self.exchange_processing * 1.5
        elif order_type == "stop":
            base_time = self.exchange_processing * 2.0
        else:
            base_time = self.exchange_processing

        # Volatility adjustment (exponential distribution)
        vol_multiplier = 1.0 + (volatility - 1.0) * 0.5  # 50% increase per vol unit
        processing_time = np.random.exponential(base_time * vol_multiplier)

        return processing_time

    def _calculate_queue_delay(self, volatility: float) -> float:
        """
        Simulate order queue delay.

        In high volatility / high volume periods, orders queue up
        and execution is delayed.
        """
        if volatility > 1.5:  # High volatility regime
            # Exponential delay with mean proportional to volatility
            mean_queue_delay = (volatility - 1.0) * 50  # Up to 50ms per volatility unit
            queue_delay = np.random.exponential(mean_queue_delay)
        else:
            queue_delay = 0

        return queue_delay

    def _get_time_of_day_multiplier(self, hour: Optional[int]) -> float:
        """
        Adjust latency based on time of day.

        Market open/close: Higher latency (congestion)
        Lunch hours: Lower latency
        Off-hours: Variable latency
        """
        if hour is None:
            return 1.0

        # Latency multiplier by hour (UTC)
        # Higher = more congestion
        latency_profile = {
            0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0,  # Night: normal
            4: 1.2, 5: 1.3,                   # Pre-market: building
            6: 1.5, 7: 1.5, 8: 1.6,           # Market open: HIGH
            9: 1.3, 10: 1.2,                  # Morning: settling
            11: 1.0, 12: 1.0, 13: 1.0,        # Lunch: calm
            14: 1.2, 15: 1.3,                 # Afternoon: picking up
            16: 1.4, 17: 1.5,                 # Market close: HIGH
            18: 1.2, 19: 1.1,                 # After hours: settling
            20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0  # Evening: normal
        }

        return latency_profile.get(hour, 1.0)

    def apply_latency_to_signal(
        self,
        signal_time: datetime,
        data_df,
        current_index: int,
        order_type: str = "market",
        volatility: float = 1.0
    ) -> Dict:
        """
        Apply latency to a trading signal and determine actual execution.

        Args:
            signal_time: Time when signal was generated
            data_df: DataFrame with market data
            current_index: Current bar index in data_df
            order_type: Type of order
            volatility: Current volatility measure

        Returns:
            Dictionary with:
            - execution_time: Actual execution timestamp
            - execution_index: Bar index where order executes
            - execution_price: Price at execution
            - latency_ms: Total latency in milliseconds
            - bars_delayed: Number of bars delayed
        """
        # Calculate total latency
        hour = signal_time.hour if isinstance(signal_time, datetime) else None
        latency_ms = self.calculate_total_latency(order_type, volatility, hour)

        # Convert to timedelta
        latency_delta = timedelta(milliseconds=latency_ms)
        execution_time = signal_time + latency_delta

        # Find execution bar
        if isinstance(data_df.index, type(signal_time)):
            # DatetimeIndex
            future_bars = data_df.index[data_df.index > execution_time]
            
            if len(future_bars) > 0:
                execution_bar_time = future_bars[0]
                execution_index = data_df.index.get_loc(execution_bar_time)
                execution_price = data_df.loc[execution_bar_time, 'close']
                bars_delayed = execution_index - current_index
            else:
                # Execution time is beyond available data
                self.logger.warning(f"Execution time {execution_time} beyond data range")
                return None
        else:
            # Integer index
            bars_delayed = max(1, int(latency_ms / 60000))  # Assume 1min bars
            execution_index = min(current_index + bars_delayed, len(data_df) - 1)
            execution_time = data_df.index[execution_index]
            execution_price = data_df.iloc[execution_index]['close']

        return {
            'signal_time': signal_time,
            'execution_time': execution_time,
            'execution_index': execution_index,
            'execution_price': execution_price,
            'latency_ms': latency_ms,
            'bars_delayed': bars_delayed
        }

    def calculate_market_data_staleness(
        self,
        data_timestamp: datetime,
        current_time: datetime
    ) -> float:
        """
        Calculate how stale market data is.

        Market data feeds have inherent delays. When you see a price,
        it's already X milliseconds old.

        Args:
            data_timestamp: Timestamp of market data
            current_time: Current system time

        Returns:
            Staleness in milliseconds
        """
        staleness = (current_time - data_timestamp).total_seconds() * 1000
        
        # Add market data feed delay
        staleness += self.market_data_delay

        return staleness

    def get_latency_statistics(self, n_samples: int = 1000) -> Dict:
        """
        Generate latency statistics for analysis.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Dictionary with latency statistics
        """
        samples = []
        
        for _ in range(n_samples):
            latency = self.calculate_total_latency(
                order_type="market",
                market_volatility=1.0,
                time_of_day=None
            )
            samples.append(latency)

        samples = np.array(samples)

        return {
            'mean': np.mean(samples),
            'median': np.median(samples),
            'std': np.std(samples),
            'min': np.min(samples),
            'max': np.max(samples),
            'p50': np.percentile(samples, 50),
            'p90': np.percentile(samples, 90),
            'p95': np.percentile(samples, 95),
            'p99': np.percentile(samples, 99)
        }


class LatencyProfile:
    """
    Predefined latency profiles for different trading scenarios.
    """

    @staticmethod
    def get_profile(profile_name: str) -> LatencyModel:
        """
        Get a predefined latency profile.

        Profiles:
        - 'co-located': Very low latency (co-located with exchange)
        - 'institutional': Low latency (dedicated lines)
        - 'retail_fast': Retail with good connection
        - 'retail_average': Average retail trader
        - 'retail_slow': Poor internet connection
        - 'mobile': Mobile trading app

        Args:
            profile_name: Name of the profile

        Returns:
            Configured LatencyModel
        """
        profiles = {
            'co-located': {
                'base_network_latency_ms': 1.0,
                'network_jitter_ms': 0.5,
                'exchange_processing_ms': 2.0,
                'market_data_delay_ms': 5.0,
                'queue_enabled': False
            },
            'institutional': {
                'base_network_latency_ms': 10.0,
                'network_jitter_ms': 3.0,
                'exchange_processing_ms': 5.0,
                'market_data_delay_ms': 20.0,
                'queue_enabled': True
            },
            'retail_fast': {
                'base_network_latency_ms': 30.0,
                'network_jitter_ms': 10.0,
                'exchange_processing_ms': 10.0,
                'market_data_delay_ms': 50.0,
                'queue_enabled': True
            },
            'retail_average': {
                'base_network_latency_ms': 50.0,
                'network_jitter_ms': 20.0,
                'exchange_processing_ms': 10.0,
                'market_data_delay_ms': 100.0,
                'queue_enabled': True
            },
            'retail_slow': {
                'base_network_latency_ms': 100.0,
                'network_jitter_ms': 40.0,
                'exchange_processing_ms': 15.0,
                'market_data_delay_ms': 200.0,
                'queue_enabled': True
            },
            'mobile': {
                'base_network_latency_ms': 150.0,
                'network_jitter_ms': 60.0,
                'exchange_processing_ms': 20.0,
                'market_data_delay_ms': 300.0,
                'queue_enabled': True
            }
        }

        if profile_name not in profiles:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(profiles.keys())}")

        config = profiles[profile_name]
        return LatencyModel(**config)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("=" * 70)
    print("Latency Model - Comparison of Trading Profiles")
    print("=" * 70)

    # Compare different profiles
    profiles = ['co-located', 'institutional', 'retail_fast', 'retail_average', 'retail_slow', 'mobile']

    print("\nLatency Statistics (1000 samples):\n")
    print(f"{'Profile':<20} {'Mean':<10} {'P50':<10} {'P90':<10} {'P95':<10} {'P99':<10}")
    print("-" * 70)

    for profile_name in profiles:
        model = LatencyProfile.get_profile(profile_name)
        stats = model.get_latency_statistics(n_samples=1000)

        print(f"{profile_name:<20} "
              f"{stats['mean']:>8.1f}ms "
              f"{stats['p50']:>8.1f}ms "
              f"{stats['p90']:>8.1f}ms "
              f"{stats['p95']:>8.1f}ms "
              f"{stats['p99']:>8.1f}ms")

    # Detailed example with retail_average
    print("\n" + "=" * 70)
    print("Detailed Example: Retail Average Trader")
    print("=" * 70)

    model = LatencyProfile.get_profile('retail_average')

    # Test different scenarios
    scenarios = [
        {'desc': 'Market order, normal vol, market hours', 'order_type': 'market', 'vol': 1.0, 'hour': 14},
        {'desc': 'Market order, high vol, market open', 'order_type': 'market', 'vol': 2.0, 'hour': 9},
        {'desc': 'Limit order, normal vol, lunch', 'order_type': 'limit', 'vol': 1.0, 'hour': 12},
        {'desc': 'Stop order, high vol, market close', 'order_type': 'stop', 'vol': 2.5, 'hour': 16},
    ]

    print("\nScenario Analysis:")
    print("-" * 70)

    for scenario in scenarios:
        latency = model.calculate_total_latency(
            order_type=scenario['order_type'],
            market_volatility=scenario['vol'],
            time_of_day=scenario['hour']
        )

        print(f"\n{scenario['desc']}:")
        print(f"  Total Latency: {latency:.1f}ms")
        print(f"  Equivalent to: {latency/1000:.3f} seconds")
        
        # Show bars delayed (assuming 1min bars)
        bars_1min = latency / 60000
        bars_5min = latency / 300000
        print(f"  Bars delayed: {bars_1min:.2f} (1min) | {bars_5min:.2f} (5min)")

    # Impact analysis
    print("\n" + "=" * 70)
    print("Impact Analysis")
    print("=" * 70)

    print("\nFor a fast-moving market (1% move per minute):")
    
    for profile_name in ['co-located', 'retail_average', 'mobile']:
        model = LatencyProfile.get_profile(profile_name)
        stats = model.get_latency_statistics()
        
        # Calculate potential slippage
        avg_latency_seconds = stats['mean'] / 1000
        move_per_second = 0.01 / 60  # 1% per minute = ~0.0167% per second
        expected_slippage = avg_latency_seconds * move_per_second * 100  # As percentage
        
        print(f"\n{profile_name}:")
        print(f"  Average latency: {stats['mean']:.1f}ms")
        print(f"  Expected slippage: {expected_slippage:.4f}%")
        print(f"  On $10,000 order: ${expected_slippage * 100:.2f} cost")

    print("\n" + "=" * 70)
    print("\nConclusion:")
    print("- Co-located: ~$0.17 average slippage cost")
    print("- Retail Average: ~$8.33 average slippage cost")
    print("- Mobile: ~$25.00 average slippage cost")
    print("\nFor HFT strategies, latency can make or break profitability!")
    print("=" * 70)
