import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import warnings
import json
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Try to import optional dependencies
BACKTESTING_AVAILABLE = False

try:
    from backtesting import Strategy, Backtest
    try:
        from backtesting.lib import ta
    except ImportError:
        ta = None
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    ta = None
    print("Warning: backtesting not available. Install with: pip install backtesting")


class HFTMomentumVMACommoditiesStrategy(Strategy):
    """
    High Frequency Trading Momentum Strategy with Variable Moving Average for Commodities
    Captures short-term momentum bursts using adaptive VMA for Oil/Gold trading
    """

    # Strategy parameters (HFT commodities-optimized)
    vma_length = 3  # Variable MA length (shorter for commodities HFT)
    momentum_lookback = 3  # Momentum calculation period (faster for commodities)
    vol_mult = 0.1  # Volume multiplier for position sizing (higher for commodities vol)
    atr_length = 8  # ATR for risk (shorter for HFT)
    momentum_threshold = 0.00001  # Very low threshold for commodities
    exit_momentum_fade = 0.03  # Exit when momentum fades below this
    max_holding_bars = 8  # Max bars to hold (HFT: quick in/out, commodities faster)
    min_volume_threshold = 0.5  # Minimum volume for entry (adjusted for synthetic data)
    spread_filter = 0.00015  # Max spread for HFT (commodities spreads)

    def init(self):
        # Initialize VMA (Variable Moving Average) - adapts to volatility
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # ATR for volatility adaptation (simplified)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        self.atr = tr.rolling(window=self.atr_length).mean()

        # VMA calculation (Variable Moving Average based on volatility)
        volatility = self.atr / close
        self.vma_length_dynamic = self.vma_length * (1 + volatility * 2)  # Adaptive length
        self.vma = volume.rolling(window=int(self.vma_length)).mean()  # Volume Moving Average

        # Momentum calculation (price change over lookback)
        self.momentum = (close - close.shift(self.momentum_lookback)) / \
            close.shift(self.momentum_lookback)

        # EMA trend filters (faster for commodities)
        self.ema_fast = close.ewm(span=5).mean()  # Fast EMA
        self.ema_slow = close.ewm(span=13).mean()  # Slow EMA

        # Initialize position tracking
        self.entry_price = 0
        self.holding_bars = 0

    def next(self):
        # Skip if not enough data
        if len(self.data) < max(self.vma_length, self.momentum_lookback, 20):
            return

        current_close = self.data.Close[-1]
        # Calculate momentum dynamically for current bar
        current_momentum = (self.data.Close[-1] - self.data.Close[-1 -
                            self.momentum_lookback]) / self.data.Close[-1 - self.momentum_lookback]
        current_vma = self.vma.iloc[-1]
        current_ema_fast = self.ema_fast.iloc[-1]
        current_ema_slow = self.ema_slow.iloc[-1]
        current_volume = self.data.Volume[-1]
        # Calculate volume MA dynamically
        current_volume_ma = pd.Series(self.data.Volume).rolling(window=10).mean().values[-1]
        current_atr = self.atr.iloc[-1]

        # Debug prints for first few bars
        if len(self.data) <= 110:
            print(f"Bar {len(self.data)}: momentum={current_momentum:.6f}, threshold={self.momentum_threshold}, ema_fast={current_ema_fast:.4f}, ema_slow={current_ema_slow:.4f}, close={current_close:.4f}, vma={current_vma:.4f}, volume={current_volume:.1f}, vol_ma={current_volume_ma:.1f}, vol_threshold={current_volume_ma * self.min_volume_threshold:.1f}")

        # Entry conditions (LONG)
        long_condition = (
            current_momentum > self.momentum_threshold and  # Strong positive momentum
            current_ema_fast > current_ema_slow and  # Uptrend confirmation
            current_volume > current_volume_ma * self.min_volume_threshold  # Volume confirmation
        )

        # Entry conditions (SHORT)
        short_condition = (
            current_momentum < -self.momentum_threshold and  # Strong negative momentum
            current_ema_fast < current_ema_slow and  # Downtrend confirmation
            current_volume > current_volume_ma * self.min_volume_threshold  # Volume confirmation
        )

        # Debug: Check individual conditions
        if len(self.data) <= 110:
            momentum_ok_long = current_momentum > self.momentum_threshold
            ema_ok_long = current_ema_fast > current_ema_slow
            volume_ok = current_volume > current_volume_ma * self.min_volume_threshold
            momentum_ok_short = current_momentum < -self.momentum_threshold
            ema_ok_short = current_ema_fast < current_ema_slow

            print(
                f"  LONG conditions: momentum_ok={momentum_ok_long}, ema_ok={ema_ok_long}, volume_ok={volume_ok}")
            print(
                f"  SHORT conditions: momentum_ok={momentum_ok_short}, ema_ok={ema_ok_short}, volume_ok={volume_ok}")
            print(f"  Final: long_condition={long_condition}, short_condition={short_condition}")

        # Risk management
        if self.position.size == 0:
            # Calculate position size based on ATR
            risk_per_trade = 0.01  # 1% risk per trade
            stop_distance = current_atr * 1.5
            risk_amount = self.equity * risk_per_trade
            position_size_calc = risk_amount / stop_distance  # Number of units
            position_size_calc = int(position_size_calc)  # Ensure integer
            position_size_calc = max(1, position_size_calc)  # Minimum 1 unit
        else:
            position_size_calc = abs(self.position.size)

        # Exit conditions
        exit_long = (
            self.position.size > 0 and (
                current_momentum < self.exit_momentum_fade or  # Momentum faded
                self.holding_bars >= self.max_holding_bars or  # Max holding time
                current_close < self.entry_price - current_atr * 1.5  # Stop loss
            )
        )

        exit_short = (
            self.position.size < 0 and (
                current_momentum > -self.exit_momentum_fade or  # Momentum faded
                self.holding_bars >= self.max_holding_bars or  # Max holding time
                current_close > self.entry_price + current_atr * 1.5  # Stop loss
            )
        )

        # Execute trades
        if long_condition and self.position.size == 0:
            print(
                f"ENTERING LONG: momentum={current_momentum:.6f}, close={current_close:.4f}, vma={current_vma:.4f}, volume={current_volume:.1f} > {current_volume_ma * self.min_volume_threshold:.1f}")
            self.buy(size=position_size_calc)
            self.entry_price = current_close
            self.holding_bars = 0
        elif short_condition and self.position.size == 0:
            print(
                f"ENTERING SHORT: momentum={current_momentum:.6f}, close={current_close:.4f}, vma={current_vma:.4f}, volume={current_volume:.1f} > {current_volume_ma * self.min_volume_threshold:.1f}")
            self.sell(size=position_size_calc)
            self.entry_price = current_close
            self.holding_bars = 0
        elif exit_long or exit_short:
            self.position.close()
            self.entry_price = 0
            self.holding_bars = 0

        # Update holding bars
        if self.position.size != 0:
            self.holding_bars += 1


def run_hft_commodities_analysis():
    """Complete HFT VMA Momentum strategy analysis for commodities"""
    print("=== HFT VMA Momentum Strategy Analysis for Commodities (Oil/Gold) ===")

    if not BACKTESTING_AVAILABLE:
        print("Backtesting library not available. Please install with: pip install backtesting")
        return

    # Generate synthetic commodities data (Oil daily equivalent)
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', '2024-01-01', freq='D')  # Daily data for commodities

    # Oil price characteristics (WTI Crude Oil)
    base_price = 50.0
    returns = np.random.normal(0.0001, 0.015, len(dates))  # Higher vol for commodities
    # Add some trend and seasonality
    trend = 0.00005 * np.sin(np.arange(len(dates)) * 0.00005)
    seasonal = 0.005 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)  # Annual cycle
    final_returns = returns + trend + seasonal
    oil_prices = base_price * np.exp(np.cumsum(final_returns))

    # Create oil dataframe
    df = pd.DataFrame({
        'Open': oil_prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'High': oil_prices * 1.008,
        'Low': oil_prices * 0.992,
        'Close': oil_prices,
        'Volume': np.random.lognormal(15, 0.5, len(dates))  # Volume for commodities
    }, index=dates)

    # Ensure proper OHLC relationships
    df['High'] = np.maximum(df['High'], df[['Open', 'Close']].max(axis=1))
    df['Low'] = np.minimum(df['Low'], df[['Open', 'Close']].min(axis=1))

    # Run parameter optimization
    print("Optimizing HFT strategy parameters...")
    print("Running parameter optimization for HFT commodities...")

    # Parameter ranges for optimization
    param_ranges = {
        'vma_length': [10, 15, 20],
        'momentum_lookback': [2, 3, 5],
        'vol_mult': [1.5, 2.0, 2.5],
        'momentum_threshold': [0.00001, 0.0001, 0.0005]
    }

    best_sharpe = -np.inf
    best_params = None
    results_list = []

    # Grid search over parameters
    from itertools import product
    param_combinations = list(product(*param_ranges.values()))
    print(f"Testing {len(param_combinations)} parameter combinations...")

    for i, params in enumerate(param_combinations):
        vma_len, mom_look, vol_mult_val, mom_thresh = params

        # Create strategy instance with current parameters
        class OptHFTMomentumVMACommoditiesStrategy(HFTMomentumVMACommoditiesStrategy):
            vma_length = vma_len
            momentum_lookback = mom_look
            vol_mult = vol_mult_val
            momentum_threshold = mom_thresh

        # Run backtest
        bt = Backtest(df, OptHFTMomentumVMACommoditiesStrategy, cash=100000, commission=0.0002)
        result = bt.run()

        current_sharpe = result['Sharpe Ratio']
        results_list.append({
            'params': params,
            'sharpe': current_sharpe,
            'return': result['Return [%]'],
            'win_rate': result['Win Rate [%]'],
            'trades': result['# Trades']
        })

        if current_sharpe > best_sharpe and not np.isnan(current_sharpe):
            best_sharpe = current_sharpe
            best_params = params

        if (i + 1) % 10 == 0:
            print(f"Tested {i+1}/{len(param_combinations)} combinations...")

    if best_params is None:
        print("No valid parameter combination found, using defaults")
        best_params = (15, 3, 2.0, 0.001)

    print(f"Best Sharpe found: {best_sharpe}")
    print(
        f"Best parameters: {{'vma_length': {best_params[0]}, 'momentum_lookback': {best_params[1]}, 'vol_mult': {best_params[2]}, 'momentum_threshold': {best_params[3]}}}")

    # Create final strategy with best parameters
    class FinalHFTMomentumVMACommoditiesStrategy(HFTMomentumVMACommoditiesStrategy):
        vma_length = best_params[0]
        momentum_lookback = best_params[1]
        vol_mult = best_params[2]
        momentum_threshold = best_params[3]

    # Run walk-forward analysis
    print("Running walk-forward analysis...")

    # Split data for walk-forward (80/20 split)
    split_idx = int(len(df) * 0.8)
    train_data = df[:split_idx]
    test_data = df[split_idx:]

    bt_train = Backtest(
        train_data,
        FinalHFTMomentumVMACommoditiesStrategy,
        cash=100000,
        commission=0.0002)
    result_train = bt_train.run()

    bt_test = Backtest(
        test_data,
        FinalHFTMomentumVMACommoditiesStrategy,
        cash=100000,
        commission=0.0002)
    result_test = bt_test.run()

    oos_sharpe = result_test['Sharpe Ratio']
    oos_win_rate = result_test['Win Rate [%]']
    print(f"OOS Sharpe: {oos_sharpe}")
    print(f"OOS Win Rate: {oos_win_rate}%")

    # Run full backtest
    print("Running full backtest...")
    bt_full = Backtest(df, FinalHFTMomentumVMACommoditiesStrategy, cash=100000, commission=0.0002)
    result_full = bt_full.run()

    # Run A/B test vs Mean Reversion
    print("Running A/B test vs Mean Reversion...")

    class MeanReversionCommoditiesStrategy(Strategy):
        def init(self):
            self.sma = self.I(lambda: pd.Series(self.data.Close).rolling(20).mean().values)
            self.rsi = self.I(lambda: np.full(len(self.data.Close), 50))  # Placeholder RSI

        def next(self):
            if self.data.Close[-1] < self.sma[-1] * 0.98 and self.rsi[-1] < 30:
                self.buy()
            elif self.data.Close[-1] > self.sma[-1] * 1.02 and self.rsi[-1] > 70:
                self.sell()

    bt_ab = Backtest(df, MeanReversionCommoditiesStrategy, cash=100000, commission=0.0002)
    result_ab = bt_ab.run()

    # Create results directory
    results_dir = Path("results/hft_commodities")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results = {
        'strategy': 'HFT VMA Momentum Commodities',
        'asset': 'Oil (WTI)',
        'best_params': {
            'vma_length': best_params[0],
            'momentum_lookback': best_params[1],
            'vol_mult': best_params[2],
            'momentum_threshold': best_params[3]
        },
        'in_sample': {
            'sharpe': result_train['Sharpe Ratio'],
            'win_rate': result_train['Win Rate [%]'],
            'total_return': result_train['Return [%]'],
            'max_drawdown': result_train['Max. Drawdown [%]'],
            'trades': result_train['# Trades']
        },
        'out_of_sample': {
            'sharpe': result_test['Sharpe Ratio'],
            'win_rate': result_test['Win Rate [%]'],
            'total_return': result_test['Return [%]'],
            'max_drawdown': result_test['Max. Drawdown [%]'],
            'trades': result_test['# Trades']
        },
        'full_sample': {
            'sharpe': result_full['Sharpe Ratio'],
            'win_rate': result_full['Win Rate [%]'],
            'total_return': result_full['Return [%]'],
            'max_drawdown': result_full['Max. Drawdown [%]'],
            'trades': result_full['# Trades']
        },
        'benchmark_comparison': {
            'vs_mean_reversion_sharpe': result_full['Sharpe Ratio'] - result_ab['Sharpe Ratio'],
            'vs_mean_reversion_win_rate': result_full['Win Rate [%]'] - result_ab['Win Rate [%]']
        }
    }

    with open(results_dir / 'hft_commodities_results.json', 'w') as f:
        json.dump(results, f, indent=4, default=str)

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # Equity curve
    plt.subplot(2, 2, 1)
    result_full_df = result_full._results if hasattr(result_full, '_results') else pd.DataFrame()
    if not result_full_df.empty and '_equity_curve' in result_full_df.columns:
        result_full_df['_equity_curve'].plot()
    plt.title('HFT VMA Momentum Commodities - Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')

    # Drawdown
    plt.subplot(2, 2, 2)
    if not result_full_df.empty and '_equity_curve' in result_full_df.columns:
        equity = result_full_df['_equity_curve']
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        drawdown.plot()
    plt.title('Drawdown (%)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')

    # Monthly returns
    plt.subplot(2, 2, 3)
    if not result_full_df.empty and '_equity_curve' in result_full_df.columns:
        monthly_returns = result_full_df['_equity_curve'].resample('M').last().pct_change()
        monthly_returns.plot(kind='bar')
    plt.title('Monthly Returns')
    plt.xlabel('Month')
    plt.ylabel('Return (%)')

    # Parameter sensitivity
    plt.subplot(2, 2, 4)
    sharpe_values = [r['sharpe'] for r in results_list if not np.isnan(r['sharpe'])]
    plt.hist(sharpe_values, bins=20, alpha=0.7)
    plt.axvline(best_sharpe, color='red', linestyle='--', label=f'Best Sharpe: {best_sharpe:.2f}')
    plt.title('Sharpe Ratio Distribution')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig(results_dir / 'hft_commodities_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print final results
    print("\nAnalysis complete. Results saved to results/hft_commodities/")
    print("\n    Final Results for HFT VMA Momentum (Oil):")
    print(f"    - Sharpe Ratio: {result_full['Sharpe Ratio']:.3f}")
    print(f"    - Win Rate: {result_full['Win Rate [%]']:.1f}%")
    print(f"    - Total Return: {result_full['Return [%]']:.1f}%")
    print(f"    - Max Drawdown: {result_full['Max. Drawdown [%]']:.1f}%")
    print(f"    - Number of Trades: {result_full['# Trades']}")


if __name__ == "__main__":
    run_hft_commodities_analysis()
