import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_stocks_data():
    """Generate synthetic stocks data (S&P 500 style)"""
    np.random.seed(42)
    n_bars = 2000  # More data for stocks

    # Base price around 4000 (typical S&P 500 level)
    base_price = 4000.0

    # Generate realistic price movements (stocks have moderate volatility)
    returns = np.random.normal(0, 0.0012, n_bars)  # Daily returns ~0.12%
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.008, n_bars)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.008, n_bars)))
    opens = prices + np.random.normal(0, prices * 0.003, n_bars)
    opens = np.clip(opens, lows, highs)

    # Generate volume (stocks volume varies significantly)
    base_volume = 2000000  # 2M base volume
    volume = base_volume * (1 + np.random.exponential(0.6, n_bars))

    # Create DataFrame
    dates = pd.date_range('2020-01-01', periods=n_bars, freq='1D')
    data = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volume
    }, index=dates)

    return data


class MomentumMACDADXStocks(Strategy):
    """
    Momentum MACD+ADX Strategy for Stocks
    Uses MACD for momentum signals and ADX for trend strength confirmation
    """

    # Strategy parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    adx_length = 14
    adx_threshold = 25  # ADX above this indicates strong trend
    risk_per_trade = 0.02  # 2% risk per trade for stocks
    atr_length = 14

    def init(self):
        # Initialize indicators
        self.macd_line, self.signal_line, self.histogram = self.I(self.calculate_macd)
        self.adx = self.I(self.calculate_adx)
        self.atr = self.I(self.calculate_atr)

        # Track position
        self.position_size = 0

    def calculate_macd(self):
        """Calculate MACD indicator"""
        ema_fast = pd.Series(self.data.Close).ewm(span=self.macd_fast).mean()
        ema_slow = pd.Series(self.data.Close).ewm(span=self.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        histogram = macd_line - signal_line

        return macd_line.values, signal_line.values, histogram.values

    def calculate_adx(self):
        """Calculate ADX (Average Directional Index)"""
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)

        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), high - high.shift(1), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), low.shift(1) - low, 0)

        # Calculate Directional Indicators
        atr = tr.rolling(self.adx_length).mean()
        di_plus = 100 * (pd.Series(dm_plus).rolling(self.adx_length).mean() / atr)
        di_minus = 100 * (pd.Series(dm_minus).rolling(self.adx_length).mean() / atr)

        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(self.adx_length).mean()

        return adx.fillna(20).values  # Default ADX value

    def calculate_atr(self):
        """Calculate ATR indicator"""
        high_low = self.data.High - self.data.Low
        high_close = np.abs(self.data.High - np.roll(self.data.Close, 1))
        low_close = np.abs(self.data.Low - np.roll(self.data.Close, 1))
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return pd.Series(tr).rolling(self.atr_length).mean().fillna(1).values

    def next(self):
        if len(self.data) < self.macd_slow + self.adx_length:
            return

        # Get current indicator values
        current_macd = self.macd_line[-1]
        current_signal = self.signal_line[-1]
        current_adx = self.adx[-1]
        current_atr = self.atr[-1]

        # MACD crossover signals
        # Previous values for crossover detection
        prev_macd = self.macd_line[-2] if len(self.macd_line) > 1 else current_macd
        prev_signal = self.signal_line[-2] if len(self.signal_line) > 1 else current_signal

        macd_cross_up = (prev_macd <= prev_signal) and (current_macd > current_signal)
        macd_cross_down = (prev_macd >= prev_signal) and (current_macd < current_signal)

        # ADX trend strength filter
        strong_trend = current_adx > self.adx_threshold

        # Entry conditions
        long_condition = (
            macd_cross_up and  # MACD bullish crossover
            strong_trend and   # Strong trend confirmed by ADX
            self.position.size == 0
        )

        short_condition = (
            macd_cross_down and  # MACD bearish crossover
            strong_trend and    # Strong trend confirmed by ADX
            self.position.size == 0
        )

        # Exit conditions (opposite signals or trend weakening)
        exit_long = (
            self.position.size > 0 and
            (macd_cross_down or current_adx < self.adx_threshold)
        )

        exit_short = (
            self.position.size < 0 and
            (macd_cross_up or current_adx < self.adx_threshold)
        )

        # Execute trades
        if long_condition:
            risk_amount = self.equity * self.risk_per_trade
            stop_distance = current_atr * 2.0  # 2 ATR stop for stocks
            if stop_distance > 0:
                position_size_calc = max(1, int(risk_amount / stop_distance))
                self.buy(size=position_size_calc)
                print(
                    f"ENTERING LONG: MACD cross up, ADX={current_adx:.1f}, close={self.data.Close[-1]:.2f}")

        elif short_condition:
            risk_amount = self.equity * self.risk_per_trade
            stop_distance = current_atr * 2.0
            if stop_distance > 0:
                position_size_calc = max(1, int(risk_amount / stop_distance))
                self.sell(size=position_size_calc)
                print(
                    f"ENTERING SHORT: MACD cross down, ADX={current_adx:.1f}, close={self.data.Close[-1]:.2f}")

        # Exit positions
        if exit_long or exit_short:
            self.position.close()
            direction = "LONG" if exit_long else "SHORT"
            print(f"EXITING {direction}: condition met, close={self.data.Close[-1]:.2f}")


class MACDBaselineStocks(Strategy):
    """Baseline MACD strategy for A/B testing (without ADX filter)"""

    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    risk_per_trade = 0.02

    def init(self):
        self.macd_line, self.signal_line, self.histogram = self.I(self.calculate_macd)
        self.atr = self.I(self.calculate_atr)

    def calculate_macd(self):
        """Calculate MACD indicator"""
        ema_fast = pd.Series(self.data.Close).ewm(span=self.macd_fast).mean()
        ema_slow = pd.Series(self.data.Close).ewm(span=self.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        histogram = macd_line - signal_line

        return macd_line.values, signal_line.values, histogram.values

    def calculate_atr(self):
        """Calculate ATR indicator"""
        high_low = self.data.High - self.data.Low
        high_close = np.abs(self.data.High - np.roll(self.data.Close, 1))
        low_close = np.abs(self.data.Low - np.roll(self.data.Close, 1))
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return pd.Series(tr).rolling(14).mean().fillna(1).values

    def next(self):
        if len(self.data) < self.macd_slow:
            return

        # Get current indicator values
        current_macd = self.macd_line[-1]
        current_signal = self.signal_line[-1]
        current_atr = self.atr[-1]

        # MACD crossover signals
        prev_macd = self.macd_line[-2] if len(self.macd_line) > 1 else current_macd
        prev_signal = self.signal_line[-2] if len(self.signal_line) > 1 else current_signal

        macd_cross_up = (prev_macd <= prev_signal) and (current_macd > current_signal)
        macd_cross_down = (prev_macd >= prev_signal) and (current_macd < current_signal)

        # Entry conditions (without ADX filter)
        if self.position.size == 0:
            if macd_cross_up:
                risk_amount = self.equity * self.risk_per_trade
                stop_distance = current_atr * 2.0
                if stop_distance > 0:
                    position_size_calc = max(1, int(risk_amount / stop_distance))
                    self.buy(size=position_size_calc)

            elif macd_cross_down:
                risk_amount = self.equity * self.risk_per_trade
                stop_distance = current_atr * 2.0
                if stop_distance > 0:
                    position_size_calc = max(1, int(risk_amount / stop_distance))
                    self.sell(size=position_size_calc)

        # Exit on opposite signal
        else:
            if self.position.size > 0 and macd_cross_down:
                self.position.close()
            elif self.position.size < 0 and macd_cross_up:
                self.position.close()


def run_momentum_macd_adx_stocks():
    """Run Momentum MACD+ADX Stocks strategy with walk-forward testing and A/B analysis"""

    print("Running Momentum MACD+ADX Strategy for Stocks (S&P 500)")
    print("=" * 60)

    # Generate synthetic data
    data = generate_synthetic_stocks_data()

    # Run backtest
    bt = Backtest(data, MomentumMACDADXStocks, cash=10000, commission=.001)  # Stock commission
    result = bt.run()

    print("MACD+ADX Strategy Results:")
    print(f"Sharpe Ratio: {result['Sharpe Ratio']:.3f}")
    print(f"Win Rate: {result['Win Rate [%]']:.1f}%")
    print(f"Total Return: {result['Return [%]']:.1f}%")
    print(f"Max Drawdown: {result['Max. Drawdown [%]']:.1f}%")
    print(f"Total Trades: {result['# Trades']}")

    # A/B Testing vs MACD Baseline
    print("\nRunning A/B Test vs MACD Baseline...")

    bt_baseline = Backtest(data, MACDBaselineStocks, cash=10000, commission=.001)
    result_baseline = bt_baseline.run()

    print("MACD Baseline Strategy Results:")
    print(f"Sharpe Ratio: {result_baseline['Sharpe Ratio']:.3f}")
    print(f"Win Rate: {result_baseline['Win Rate [%]']:.1f}%")
    print(f"Total Return: {result_baseline['Return [%]']:.1f}%")
    print(f"Max Drawdown: {result_baseline['Max. Drawdown [%]']:.1f}%")
    print(f"Total Trades: {result_baseline['# Trades']}")

    # Comparison
    print("\nA/B Test Results:")
    print(f"Sharpe Improvement: {result['Sharpe Ratio'] - result_baseline['Sharpe Ratio']:.3f}")
    print(f"Win Rate Improvement: {result['Win Rate [%]'] - result_baseline['Win Rate [%]']:.1f}%")
    print(f"Return Improvement: {result['Return [%]'] - result_baseline['Return [%]']:.1f}%")

    # Walk-forward testing simulation
    print("\nPerforming Walk-Forward Analysis...")
    wf_results = perform_walk_forward_analysis(data, MomentumMACDADXStocks)

    # Save results
    save_results(result, result_baseline, wf_results)

    # Plot results
    plot_results(bt, result, result_baseline)

    return result, result_baseline


def perform_walk_forward_analysis(data, strategy_class):
    """Perform walk-forward analysis"""
    n_splits = 5
    results = []

    # Simple walk-forward simulation
    total_bars = len(data)

    for i in range(n_splits):
        start_idx = i * (total_bars // n_splits)
        end_idx = (i + 1) * (total_bars // n_splits)

        if end_idx > total_bars:
            end_idx = total_bars

        train_data = data.iloc[start_idx:end_idx]

        bt = Backtest(train_data, strategy_class, cash=10000, commission=.001)
        result = bt.run()

        results.append({
            'period': i + 1,
            'sharpe': result['Sharpe Ratio'],
            'win_rate': result['Win Rate [%]'],
            'return': result['Return [%]'],
            'trades': result['# Trades']
        })

    return results


def save_results(macd_adx_result, baseline_result, wf_results):
    """Save results to files"""

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save main results
    results = {
        'macd_adx_strategy': {
            'sharpe_ratio': macd_adx_result['Sharpe Ratio'],
            'win_rate': macd_adx_result['Win Rate [%]'],
            'total_return': macd_adx_result['Return [%]'],
            'max_drawdown': macd_adx_result['Max. Drawdown [%]'],
            'total_trades': macd_adx_result['# Trades']
        },
        'baseline_strategy': {
            'sharpe_ratio': baseline_result['Sharpe Ratio'],
            'win_rate': baseline_result['Win Rate [%]'],
            'total_return': baseline_result['Return [%]'],
            'max_drawdown': baseline_result['Max. Drawdown [%]'],
            'total_trades': baseline_result['# Trades']
        },
        'walk_forward_results': wf_results
    }

    with open(results_dir / "macd_adx_stocks_results.json", 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {results_dir}/macd_adx_stocks_results.json")


def plot_results(bt, macd_adx_result, baseline_result):
    """Plot strategy results"""

    # Create plots directory
    plots_dir = Path("results/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot equity curve
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    bt.plot()
    plt.title('MACD+ADX Stocks Strategy - Equity Curve')

    plt.subplot(2, 1, 2)
    # Simple comparison plot
    plt.plot([macd_adx_result['Sharpe Ratio'], baseline_result['Sharpe Ratio']],
             label=['MACD+ADX', 'MACD Baseline'])
    plt.title('Strategy Comparison - Sharpe Ratio')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "macd_adx_stocks_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {plots_dir}/macd_adx_stocks_analysis.png")


if __name__ == "__main__":
    run_momentum_macd_adx_stocks()
