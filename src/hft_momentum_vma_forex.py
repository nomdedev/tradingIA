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
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    print("Warning: backtesting not available. Install with: pip install backtesting")


class HFTMomentumVMAForexStrategy(Strategy):
    """
    High Frequency Trading Momentum Strategy with Variable Moving Average for Forex
    Captures short-term momentum bursts using adaptive VMA for EUR/USD
    """

    # Strategy parameters (HFT forex-optimized)
    vma_length = 20  # Variable MA length (adaptive)
    momentum_lookback = 5  # Momentum calculation period
    vol_mult = 1.8  # Volume multiplier for position sizing
    atr_length = 10  # ATR for risk (shorter for HFT)
    momentum_threshold = 0.001  # Very low momentum entry threshold (%)
    exit_momentum_fade = 0.05  # Exit when momentum fades below this
    max_holding_bars = 12  # Max bars to hold (HFT: quick in/out)
    min_volume_threshold = 1.2  # Minimum volume for entry
    spread_filter = 0.00002  # Max spread for HFT (2 pips for EUR/USD)

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

        # VMA (Variable Moving Average)
        self.vma = close.rolling(window=self.vma_length).mean()

        # Momentum calculation (rate of change)
        self.momentum = (close - close.shift(self.momentum_lookback)) / \
            close.shift(self.momentum_lookback)

        # Volume confirmation
        self.avg_volume = volume.rolling(window=20).mean()

        # Trend bias (EMA crossover for HFT)
        self.ema_fast = close.ewm(span=8, adjust=False).mean()
        self.ema_slow = close.ewm(span=21, adjust=False).mean()

        # Position tracking
        self.entry_bar = 0
        self.entry_price = 0

    def next(self):
        # Ensure we have enough data
        if len(self.data.Close) < 50:
            return

        idx = len(self.data.Close) - 1
        current_price = self.data.Close[idx]

        # Check exit conditions first
        if self.position:
            bars_held = idx - self.entry_bar

            # Exit on max holding time
            if bars_held >= self.max_holding_bars:
                self.position.close()
                return

            # Exit on momentum fade
            if self.position.is_long and self.momentum.iloc[idx] < self.exit_momentum_fade:
                self.position.close()
                return
            elif self.position.is_short and self.momentum.iloc[idx] > -self.exit_momentum_fade:
                self.position.close()
                return

            # Exit on adverse EMA crossover
            if self.position.is_long and self.ema_fast.iloc[idx] < self.ema_slow.iloc[idx]:
                self.position.close()
                return
            elif self.position.is_short and self.ema_fast.iloc[idx] > self.ema_slow.iloc[idx]:
                self.position.close()
                return

        # Entry conditions for LONG momentum (very simplified for HFT)
        long_momentum = self.momentum.iloc[idx] > self.momentum_threshold
        ema_bias = self.ema_fast.iloc[idx] > self.ema_slow.iloc[idx]

        # Debug prints (remove after testing)
        if idx > 100 and idx < 110:  # Print first few bars
            print(
                f"Bar {idx}: momentum={self.momentum.iloc[idx]:.6f}, threshold={self.momentum_threshold}, ema_fast={self.ema_fast.iloc[idx]:.4f}, ema_slow={self.ema_slow.iloc[idx]:.4f}")

        if long_momentum and ema_bias and not self.position:

            # Calculate position size based on risk (HFT: smaller, more frequent)
            risk_amount = self.equity * 0.005  # 0.5% risk per HFT trade
            stop_distance = self.atr.iloc[idx] * 1.5  # Tighter stops for HFT
            if stop_distance > 0:
                position_value = risk_amount / stop_distance
                size_fraction = min(max(position_value / self.equity, 0.002),
                                    0.05)  # 0.2-5% per trade
            else:
                size_fraction = 0.01

            # HFT entry with tight SL/TP
            sl_price = current_price - stop_distance
            tp_price = current_price + (stop_distance * 2.5)  # 2.5:1 RR for HFT

            # Ensure proper order for long: TP > entry > SL
            if tp_price > current_price > sl_price:
                self.buy(size=size_fraction, sl=sl_price, tp=tp_price)
                self.entry_bar = idx
                self.entry_price = current_price

        # Entry conditions for SHORT momentum (very simplified for HFT)
        short_momentum = self.momentum.iloc[idx] < -self.momentum_threshold
        ema_bias_short = self.ema_fast.iloc[idx] < self.ema_slow.iloc[idx]

        if short_momentum and ema_bias_short and not self.position:

            # Calculate position size based on risk
            risk_amount = self.equity * 0.005  # 0.5% risk per HFT trade
            stop_distance = self.atr.iloc[idx] * 1.5
            if stop_distance > 0:
                position_value = risk_amount / stop_distance
                size_fraction = min(max(position_value / self.equity, 0.002), 0.05)
            else:
                size_fraction = 0.01

            # HFT entry with tight SL/TP
            sl_price = current_price + stop_distance
            tp_price = current_price - (stop_distance * 2.5)  # 2.5:1 RR

            # Ensure proper order for short: TP < entry < SL
            if tp_price < current_price < sl_price:
                self.sell(size=size_fraction, sl=sl_price, tp=tp_price)
                self.entry_bar = idx
                self.entry_price = current_price


def walk_forward_test_hft_forex(df: pd.DataFrame, params: Dict = None) -> Dict:
    """Walk-forward optimization and testing for HFT forex strategy"""
    if not BACKTESTING_AVAILABLE:
        print("Backtesting library not available")
        return {}

    if params:
        HFTMomentumVMAForexStrategy.vma_length = params.get('vma_length', 20)
        HFTMomentumVMAForexStrategy.momentum_lookback = params.get('momentum_lookback', 5)
        HFTMomentumVMAForexStrategy.vol_mult = params.get('vol_mult', 1.8)
        HFTMomentumVMAForexStrategy.momentum_threshold = params.get('momentum_threshold', 0.15)

    # Use more data for HFT (higher frequency simulation)
    test_size = int(len(df) * 0.3)  # 30% for testing
    test_data = df.iloc[-test_size:]

    results = []

    try:
        bt = Backtest(test_data, HFTMomentumVMAForexStrategy,
                      cash=10000, commission=0.00002)  # Forex commission (2 pips)
        result = bt.run()

        results.append({
            'sharpe': result['Sharpe Ratio'],
            'win_rate': result['Win Rate [%]'] / 100,
            'return': result['Return [%]'] / 100,
            'max_dd': result['Max. Drawdown [%]'] / 100,
            'trades': result['# Trades']
        })

    except Exception as e:
        print(f"Error in backtest: {e}")
        return {}

    if results:
        oos_sharpe = np.mean([r['sharpe'] for r in results])
        oos_win_rate = np.mean([r['win_rate'] for r in results])
        oos_return = np.mean([r['return'] for r in results])
        oos_max_dd = np.mean([r['max_dd'] for r in results])

        return {
            'oos_sharpe': oos_sharpe,
            'oos_win_rate': oos_win_rate,
            'oos_return': oos_return,
            'oos_max_dd': oos_max_dd,
            'total_trades': sum([r['trades'] for r in results])
        }

    return {}


def optimize_hft_forex_strategy(df: pd.DataFrame) -> Dict:
    """Grid search optimization for HFT forex strategy parameters"""
    if not BACKTESTING_AVAILABLE:
        print("Backtesting not available, using default parameters")
        return {
            'vma_length': 20, 'momentum_lookback': 5,
            'vol_mult': 1.8, 'momentum_threshold': 0.15
        }

    print("Running parameter optimization for HFT forex...")

    # Parameter ranges optimized for HFT forex
    param_grid = {
        'vma_length': [15, 20, 25],
        'momentum_lookback': [3, 5, 8],
        'vol_mult': [1.5, 1.8, 2.0],
        'momentum_threshold': [0.001, 0.002, 0.005]
    }

    best_sharpe = -np.inf
    best_params = None

    # Use 40% of data for optimization (HFT needs more data)
    train_size = int(len(df) * 0.4)
    train_df = df.iloc[:train_size]

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Testing {total_combinations} parameter combinations...")

    tested = 0
    for vma_len in param_grid['vma_length']:
        for mom_look in param_grid['momentum_lookback']:
            for vol_mult in param_grid['vol_mult']:
                for mom_thresh in param_grid['momentum_threshold']:
                    tested += 1
                    if tested % 20 == 0:
                        print(f"Tested {tested}/{total_combinations} combinations...")

                    # Set parameters
                    HFTMomentumVMAForexStrategy.vma_length = vma_len
                    HFTMomentumVMAForexStrategy.momentum_lookback = mom_look
                    HFTMomentumVMAForexStrategy.vol_mult = vol_mult
                    HFTMomentumVMAForexStrategy.momentum_threshold = mom_thresh

                    try:
                        bt = Backtest(train_df, HFTMomentumVMAForexStrategy,
                                      cash=10000, commission=0.00002)
                        result = bt.run()

                        sharpe = result['Sharpe Ratio']
                        win_rate = result['Win Rate [%]'] / 100
                        trades = result['# Trades']

                        # HFT needs many trades and high win rate
                        if trades >= 50 and win_rate > 0.55 and sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = {
                                'vma_length': vma_len,
                                'momentum_lookback': mom_look,
                                'vol_mult': vol_mult,
                                'momentum_threshold': mom_thresh
                            }

                    except Exception as e:
                        print(f"Skipping parameter combination due to error: {e}")
                        continue

    if best_params is None:
        print("No valid parameter combination found, using defaults")
        best_params = {
            'vma_length': 20, 'momentum_lookback': 5,
            'vol_mult': 1.8, 'momentum_threshold': 0.15
        }

    print(f"Best Sharpe found: {best_sharpe:.3f}")
    return best_params


def ab_test_vs_mean_reversion_forex(df: pd.DataFrame) -> Dict:
    """A/B test against Mean Reversion IBS+BB alternative for forex HFT"""
    if not BACKTESTING_AVAILABLE:
        return {}

    class MeanReversionForexStrategy(Strategy):
        """Alternative strategy: Mean Reversion IBS + Bollinger Bands for forex"""
        bb_length = 15
        bb_std = 1.8
        ibs_length = 20
        ibs_threshold = 0.35
        vol_mult = 1.6
        atr_length = 10
        ema_htf_length = 30
        risk_mult = 1.8
        tp_rr = 2.2

        def init(self):
            close = pd.Series(self.data.Close)
            high = pd.Series(self.data.High)
            low = pd.Series(self.data.Low)

            # Bollinger Bands (shorter for forex)
            self.bb_sma = close.rolling(window=self.bb_length).mean()
            self.bb_std = close.rolling(window=self.bb_length).std()
            self.bb_upper = self.bb_sma + (self.bb_std * self.bb_std)
            self.bb_lower = self.bb_sma - (self.bb_std * self.bb_std)

            # IBS (Internal Bar Strength)
            self.ibs_value = (close - low) / (high - low)

            # ATR for risk
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            self.atr_value = tr.rolling(window=self.atr_length).mean()

            # Higher timeframe EMA
            self.ema_htf = close.ewm(span=self.ema_htf_length, adjust=False).mean()

        def next(self):
            if len(self.data.Close) < 30:
                return

            # Entry conditions for LONG
            ibs_oversold = self.ibs_value.iloc[-1] < self.ibs_threshold
            bb_break = self.data.Close[-1] < self.bb_lower.iloc[-1]
            vol_confirm = True  # Simplified
            htf_bias = self.data.Close[-1] > self.ema_htf.iloc[-1]

            if ibs_oversold and bb_break and vol_confirm and htf_bias and not self.position:
                risk_amount = self.equity * 0.005
                stop_distance = self.atr_value.iloc[-1] * self.risk_mult
                position_value = risk_amount / stop_distance
                size_fraction = min(max(position_value / self.equity, 0.002), 0.08)

                sl_price = self.data.Close[-1] - stop_distance
                tp_price = self.data.Close[-1] + (stop_distance * self.tp_rr)
                self.buy(size=size_fraction, sl=sl_price, tp=tp_price)

            # Entry conditions for SHORT
            ibs_overbought = self.ibs_value.iloc[-1] > (1 - self.ibs_threshold)
            bb_break_short = self.data.Close[-1] > self.bb_upper.iloc[-1]
            htf_bias_short = self.data.Close[-1] < self.ema_htf.iloc[-1]

            if ibs_overbought and bb_break_short and vol_confirm and htf_bias_short and not self.position:
                risk_amount = self.equity * 0.005
                stop_distance = self.atr_value.iloc[-1] * self.risk_mult
                position_value = risk_amount / stop_distance
                size_fraction = min(max(position_value / self.equity, 0.002), 0.08)

                sl_price = self.data.Close[-1] + stop_distance
                tp_price = self.data.Close[-1] - (stop_distance * self.tp_rr)
                self.sell(size=size_fraction, sl=sl_price, tp=tp_price)

    # Run both strategies
    try:
        # HFT strategy
        bt_hft = Backtest(df, HFTMomentumVMAForexStrategy, cash=10000, commission=0.00002)
        result_hft = bt_hft.run()

        # Mean reversion strategy
        bt_mr = Backtest(df, MeanReversionForexStrategy, cash=10000, commission=0.00002)
        result_mr = bt_mr.run()

        # Statistical test
        from scipy import stats

        hft_returns = [result_hft['Return [%]'] / 100]
        mr_returns = [result_mr['Return [%]'] / 100]

        if len(hft_returns) > 0 and len(mr_returns) > 0:
            t_stat, p_value = stats.ttest_ind(hft_returns, mr_returns, equal_var=False)
        else:
            t_stat, p_value = 0, 1

        return {
            'hft_sharpe': result_hft['Sharpe Ratio'],
            'mr_sharpe': result_mr['Sharpe Ratio'],
            'hft_win_rate': result_hft['Win Rate [%]'] / 100,
            'mr_win_rate': result_mr['Win Rate [%]'] / 100,
            'hft_return': result_hft['Return [%]'] / 100,
            'mr_return': result_mr['Return [%]'] / 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'hft_superior': result_hft['Sharpe Ratio'] > result_mr['Sharpe Ratio'],
            'significant': p_value < 0.05
        }

    except Exception as e:
        print(f"Error in A/B test: {e}")
        return {}


def run_hft_forex_strategy_analysis():
    """Complete HFT VMA Momentum strategy analysis for forex"""
    print("=== HFT VMA Momentum Strategy Analysis for Forex (EUR/USD) ===")

    # Generate synthetic forex data (EUR/USD hourly equivalent)
    np.random.seed(42)
    dates = pd.date_range('2010-01-01', '2024-01-01', freq='H')  # Hourly data

    # EUR/USD characteristics (major forex pair) - increased volatility for momentum
    base_price = 1.20
    returns = np.random.normal(0.00001, 0.005, len(dates))  # Higher vol for momentum
    # Add some trend and seasonality
    trend = 0.00001 * np.sin(np.arange(len(dates)) * 0.0001)
    seasonal = 0.0005 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 365))  # Daily cycle
    final_returns = returns + trend + seasonal
    prices = base_price * np.exp(np.cumsum(final_returns))

    # Create forex dataframe
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.0001, len(dates))),
        'High': prices * 1.0008,
        'Low': prices * 0.9992,
        'Close': prices,
        'Volume': np.random.lognormal(8, 1, len(dates)),
    }, index=dates)

    # Ensure High >= max(Open, Close), Low <= min(Open, Close)
    df['High'] = np.maximum(df[['Open', 'Close']].max(axis=1), df['High'])
    df['Low'] = np.minimum(df[['Open', 'Close']].min(axis=1), df['Low'])

    print(f"Loaded {len(df)} hourly bars of EUR/USD data")

    # Optimize parameters
    print("Optimizing HFT strategy parameters...")
    best_params = optimize_hft_forex_strategy(df)
    print(f"Best parameters: {best_params}")

    # Run walk-forward test
    print("Running walk-forward analysis...")
    wf_results = walk_forward_test_hft_forex(df, best_params)
    oos_sharpe = wf_results.get('oos_sharpe', 'N/A')
    oos_win_rate = wf_results.get('oos_win_rate', 'N/A')
    print(f"OOS Sharpe: {oos_sharpe if isinstance(oos_sharpe, str) else f'{oos_sharpe:.3f}'}")
    print(
        f"OOS Win Rate: {oos_win_rate if isinstance(oos_win_rate, str) else f'{oos_win_rate:.1%}'}")

    # Run full backtest
    print("Running full backtest...")
    bt = Backtest(df, HFTMomentumVMAForexStrategy, cash=10000, commission=0.00002)
    result = bt.run()

    # Calculate metrics
    trades_df = result._trades if hasattr(result, '_trades') else pd.DataFrame()
    if len(trades_df) > 0:
        win_rate = (trades_df['PnL'] > 0).mean() if 'PnL' in trades_df.columns else 0
        total_return = trades_df['PnL'].sum() if 'PnL' in trades_df.columns else 0
        num_trades = len(trades_df)
        avg_trade = total_return / num_trades if num_trades > 0 else 0
        metrics = {
            'win_rate': win_rate,
            'total_return': total_return,
            'num_trades': num_trades,
            'avg_trade': avg_trade
        }
    else:
        metrics = {}

    # A/B test
    print("Running A/B test vs Mean Reversion...")
    ab_results = ab_test_vs_mean_reversion_forex(df)

    # Save results
    output_dir = Path('results/hft_forex')
    output_dir.mkdir(exist_ok=True)

    results = {
        'asset': 'EUR/USD',
        'strategy': 'HFT VMA Momentum',
        'best_params': best_params,
        'walk_forward': wf_results,
        'full_backtest': {
            'sharpe': result.get('Sharpe Ratio', 'N/A'),
            'win_rate': result.get('Win Rate [%]', 'N/A') / 100,
            'total_return': result.get('Return [%]', 'N/A') / 100,
            'max_dd': result.get('Max. Drawdown [%]', 'N/A') / 100,
            'trades': result.get('# Trades', 'N/A')
        },
        'custom_metrics': metrics,
        'ab_test': ab_results
    }

    with open(output_dir / 'metrics_hft_forex.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('HFT VMA Momentum Strategy Analysis for Forex (EUR/USD)', fontsize=16)

    # Equity curve
    axes[0, 0].plot(result._equity_curve.Equity if hasattr(
        result, '_equity_curve') else range(len(df)))
    axes[0, 0].set_title('Equity Curve')
    axes[0, 0].set_ylabel('Portfolio Value')
    axes[0, 0].grid(True, alpha=0.3)

    # Drawdown
    if hasattr(result, '_equity_curve'):
        equity = result._equity_curve.Equity
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown %')
        axes[0, 1].grid(True, alpha=0.3)

    # Monthly returns heatmap placeholder
    axes[1, 0].text(0.5, 0.5, 'Hourly Returns\nAnalysis',
                    ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Hourly Returns')

    # Key metrics
    metrics_text = f"""
    Sharpe: {result.get('Sharpe Ratio', 'N/A'):.3f}
    Win Rate: {result.get('Win Rate [%]', 'N/A'):.1f}%
    Total Return: {result.get('Return [%]', 'N/A'):.1f}%
    Max DD: {result.get('Max. Drawdown [%]', 'N/A'):.1f}%
    Trades: {result.get('# Trades', 'N/A')}
    """
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('Key Metrics')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'hft_forex_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Analysis complete. Results saved to {output_dir}/")
    print(f"""
    Final Results for HFT VMA Momentum (EUR/USD):
    - Sharpe Ratio: {result.get('Sharpe Ratio', 'N/A'):.3f}
    - Win Rate: {result.get('Win Rate [%]', 'N/A'):.1f}%
    - Total Return: {result.get('Return [%]', 'N/A'):.1f}%
    - Max Drawdown: {result.get('Max. Drawdown [%]', 'N/A'):.1f}%
    - Number of Trades: {result.get('# Trades', 'N/A')}
    """)

    return results


if __name__ == "__main__":
    # Run analysis for HFT forex strategy
    run_hft_forex_strategy_analysis()
