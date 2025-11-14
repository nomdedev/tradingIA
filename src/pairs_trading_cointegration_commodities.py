import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import warnings
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
warnings.filterwarnings('ignore')

# Try to import optional dependencies
BACKTESTING_AVAILABLE = False

try:
    from backtesting import Strategy, Backtest
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    print("Warning: backtesting not available. Install with: pip install backtesting")


class PairsTradingCommoditiesStrategy(Strategy):
    """
    Pairs Trading Cointegration Strategy for Commodities (Oil vs Gold)
    Market-neutral strategy exploiting mean-reversion in cointegrated pairs
    """

    # Strategy parameters (commodities-optimized)
    spread_zscore_threshold = 2.0  # Entry threshold for spread deviation
    half_life_target = 5  # Target half-life in days for mean reversion
    vol_mult = 1.2  # Volume multiplier for position sizing
    ema_htf_length = 50  # Higher timeframe EMA for trend bias
    risk_mult = 1.5  # Risk multiplier for stop loss
    tp_rr = 2.0  # Take profit risk-reward ratio
    max_holding_period = 10  # Max bars to hold position (commodities daily = 10 days)

    def init(self):
        # Initialize spread tracking (Oil vs Gold)
        oil = self.data.Close  # Using Close as proxy for Oil
        gold = self.data.Open  # Using Open as proxy for Gold (simplified)

        # Calculate spread: Oil - hedge_ratio * Gold
        # In practice, this would be calculated from actual cointegration
        spread_noise = np.random.normal(0, 0.01, len(oil))  # Add some noise
        self.spread = oil - 0.8 * gold + spread_noise  # Synthetic spread

        # Calculate spread statistics
        spread_series = pd.Series(self.spread)
        self.spread_mean = spread_series.rolling(
            window=50).mean().values  # Shorter window for commodities
        self.spread_std = spread_series.rolling(window=50).std().values

        # Z-score of spread
        self.spread_zscore = ((spread_series - spread_series.rolling(window=50).mean()) /
                              spread_series.rolling(window=50).std()).values

        # Higher timeframe trend bias (5-day equivalent for daily data)
        htf_close = pd.Series(oil).rolling(window=5).mean()
        self.htf_trend = htf_close.values

        # ATR for risk management (using Oil volatility)
        high_series = pd.Series(self.data.High)
        low_series = pd.Series(self.data.Low)
        close_series = pd.Series(self.data.Close)
        tr = pd.concat([
            high_series - low_series,
            (high_series - close_series.shift(1)).abs(),
            (low_series - close_series.shift(1)).abs()
        ], axis=1).max(axis=1)
        self.atr_value = tr.rolling(window=14).mean().values

        # Position tracking
        self.entry_bar = 0
        self.position_type = 0  # 0=none, 1=long spread (Oil over Gold), -1=short spread

    def next(self):
        # Ensure we have enough data
        if len(self.data.Close) < 100:  # Need at least 100 periods for spread stats
            return

        idx = len(self.data.Close) - 1

        # Check if we should exit existing position
        if self.position:
            bars_held = idx - self.entry_bar
            if bars_held >= self.max_holding_period:
                self.position.close()
                self.position_type = 0
                return

            # Exit if spread reverts to mean (simplified exit condition)
            if self.position_type == 1 and self.spread_zscore[idx] >= 0.5:
                self.position.close()
                self.position_type = 0
                return
            elif self.position_type == -1 and self.spread_zscore[idx] <= -0.5:
                self.position.close()
                self.position_type = 0
                return

        # Entry conditions for LONG spread (Oil outperforms Gold)
        spread_wide = self.spread_zscore[idx] <= -self.spread_zscore_threshold
        trend_bias = self.data.Close[idx] > self.htf_trend[idx]  # Uptrend bias
        not_reversing = self.spread_zscore[idx] > self.spread_zscore[idx - 1] if idx > 0 else False

        if spread_wide and trend_bias and not_reversing and not self.position:
            # Calculate position size based on risk
            risk_amount = self.equity * 0.02  # 2% risk per trade
            stop_distance = self.atr_value[idx] * self.risk_mult  # Use ATR for stop distance
            if stop_distance > 0:
                position_size = min(
                    risk_amount / stop_distance,
                    self.equity * 0.10)  # Max 10% of equity
                size_fraction = position_size / self.equity
                size_fraction = min(max(size_fraction, 0.005), 0.10)
            else:
                size_fraction = 0.02

            # Entry with SL/TP based on price levels (not spread)
            entry_price = self.data.Close[idx]
            # For long orders: TP above entry, SL below entry
            sl_level = entry_price - stop_distance  # SL below entry
            tp_level = entry_price + (stop_distance * self.tp_rr)  # TP above entry

            # For backtesting, ensure proper order: for long orders TP > entry > SL
            if tp_level > entry_price > sl_level:
                self.buy(size=size_fraction, sl=sl_level, tp=tp_level)
                self.entry_bar = idx
                self.position_type = 1

        # Entry conditions for SHORT spread (Gold outperforms Oil)
        spread_wide_short = self.spread_zscore[idx] >= self.spread_zscore_threshold
        trend_bias_short = self.data.Close[idx] < self.htf_trend[idx]  # Downtrend bias
        not_reversing_short = self.spread_zscore[idx] < self.spread_zscore[idx -
                                                                           1] if idx > 0 else False

        if spread_wide_short and trend_bias_short and not_reversing_short and not self.position:
            # Calculate position size based on risk
            risk_amount = self.equity * 0.02  # 2% risk per trade
            stop_distance = self.atr_value[idx] * self.risk_mult  # Use ATR for stop distance
            if stop_distance > 0:
                position_size = min(
                    risk_amount / stop_distance,
                    self.equity * 0.10)  # Max 10% of equity
                size_fraction = position_size / self.equity
                size_fraction = min(max(size_fraction, 0.005), 0.10)
            else:
                size_fraction = 0.02

            # Entry with SL/TP based on price levels (not spread)
            entry_price = self.data.Close[idx]
            # For short orders: TP below entry, SL above entry
            sl_level = entry_price + stop_distance  # SL above entry
            tp_level = entry_price - (stop_distance * self.tp_rr)  # TP below entry

            # For backtesting, ensure proper order: for short orders TP < entry < SL
            if tp_level < entry_price < sl_level:
                self.sell(size=size_fraction, sl=sl_level, tp=tp_level)
                self.entry_bar = idx
                self.position_type = -1


def calculate_cointegration(oil_data: pd.DataFrame,
                            gold_data: pd.DataFrame) -> Tuple[bool, float, float, float]:
    """Calculate cointegration between Oil and Gold"""
    # Ensure same length
    min_len = min(len(oil_data), len(gold_data))
    oil_close = oil_data['Close'].iloc[-min_len:]
    gold_close = gold_data['Close'].iloc[-min_len:]

    # Test cointegration
    try:
        coint_t, p_value, crit_values = coint(oil_close, gold_close)
        cointegrated = p_value < 0.05  # 5% significance

        # Calculate hedge ratio
        model = OLS(oil_close, gold_close).fit()
        hedge_ratio = model.params[0]

        # Calculate spread
        spread = oil_close - hedge_ratio * gold_close
        spread_mean = spread.mean()
        spread_std = spread.std()

        return cointegrated, hedge_ratio, spread_mean, spread_std

    except Exception as e:
        print(f"Cointegration calculation failed: {e}")
        return False, 1.0, 0.0, 1.0


def walk_forward_test_commodities(df: pd.DataFrame, params: Dict = None) -> Dict:
    """Walk-forward optimization and testing for commodities pairs strategy"""
    if not BACKTESTING_AVAILABLE:
        print("Backtesting library not available")
        return {}

    if params:
        PairsTradingCommoditiesStrategy.spread_zscore_threshold = params.get(
            'spread_zscore_threshold', 2.0)
        PairsTradingCommoditiesStrategy.vol_mult = params.get('vol_mult', 1.2)
        PairsTradingCommoditiesStrategy.risk_mult = params.get('risk_mult', 1.5)
        PairsTradingCommoditiesStrategy.tp_rr = params.get('tp_rr', 2.0)

    # Simple train/test split (80/20) for commodities
    split_idx = int(len(df) * 0.8)
    test_data = df.iloc[split_idx:]

    results = []

    # Test on the last 20% of data
    try:
        bt = Backtest(test_data, PairsTradingCommoditiesStrategy,
                      cash=10000, commission=0.001)  # Futures commission
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

    # Calculate OOS metrics
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


def optimize_commodities_pairs_strategy(df: pd.DataFrame) -> Dict:
    """Grid search optimization for commodities pairs strategy parameters"""
    if not BACKTESTING_AVAILABLE:
        print("Backtesting not available, using default parameters")
        return {
            'spread_zscore_threshold': 2.0, 'vol_mult': 1.2,
            'risk_mult': 1.5, 'tp_rr': 2.0
        }

    print("Running parameter optimization for commodities pairs...")

    # Parameter ranges optimized for commodities pairs
    param_grid = {
        'spread_zscore_threshold': [1.5, 2.0, 2.5],
        'vol_mult': [1.0, 1.2],
        'risk_mult': [1.2, 1.5],
        'tp_rr': [1.5, 2.0]
    }

    best_sharpe = -np.inf
    best_params = None

    # Use 50% of data for optimization (faster for commodities)
    train_size = int(len(df) * 0.5)
    train_df = df.iloc[:train_size]

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Testing {total_combinations} parameter combinations...")

    tested = 0
    for spread_thresh in param_grid['spread_zscore_threshold']:
        for vol_mult in param_grid['vol_mult']:
            for risk_mult in param_grid['risk_mult']:
                for tp_rr in param_grid['tp_rr']:
                    tested += 1
                    if tested % 10 == 0:
                        print(f"Tested {tested}/{total_combinations} combinations...")

                    # Set parameters
                    PairsTradingCommoditiesStrategy.spread_zscore_threshold = spread_thresh
                    PairsTradingCommoditiesStrategy.vol_mult = vol_mult
                    PairsTradingCommoditiesStrategy.risk_mult = risk_mult
                    PairsTradingCommoditiesStrategy.tp_rr = tp_rr

                    try:
                        # Quick backtest on training data
                        bt = Backtest(train_df, PairsTradingCommoditiesStrategy,
                                      cash=10000, commission=0.001)
                        result = bt.run()

                        sharpe = result['Sharpe Ratio']
                        win_rate = result['Win Rate [%]'] / 100
                        trades = result['# Trades']

                        # Only consider if enough trades and reasonable win rate
                        if trades >= 10 and win_rate > 0.3 and sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = {
                                'spread_zscore_threshold': spread_thresh,
                                'vol_mult': vol_mult,
                                'risk_mult': risk_mult,
                                'tp_rr': tp_rr
                            }

                    except Exception as e:
                        print(f"Skipping parameter combination due to error: {e}")
                        continue

    if best_params is None:
        print("No valid parameter combination found, using defaults")
        best_params = {
            'spread_zscore_threshold': 2.0, 'vol_mult': 1.2,
            'risk_mult': 1.5, 'tp_rr': 2.0
        }

    print(f"Best Sharpe found: {best_sharpe:.3f}")
    return best_params


def ab_test_vs_mean_reversion(df: pd.DataFrame) -> Dict:
    """A/B test against Mean Reversion IBS+BB alternative for commodities"""
    if not BACKTESTING_AVAILABLE:
        return {}

    class MeanReversionCommoditiesStrategy(Strategy):
        """Alternative strategy: Mean Reversion IBS + Bollinger Bands for commodities"""
        bb_length = 20
        bb_std = 2.0
        ibs_length = 25
        ibs_threshold = 0.35
        vol_mult = 1.4
        atr_length = 14
        ema_htf_length = 50
        risk_mult = 1.5
        tp_rr = 2.0

        def init(self):
            close = pd.Series(self.data.Close)
            high = pd.Series(self.data.High)
            low = pd.Series(self.data.Low)

            # Bollinger Bands
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
            if len(self.data.Close) < 50:
                return

            # Entry conditions for LONG
            ibs_oversold = self.ibs_value.iloc[-1] < self.ibs_threshold
            bb_break = self.data.Close[-1] < self.bb_lower.iloc[-1]
            vol_confirm = True  # Simplified for commodities
            htf_bias = self.data.Close[-1] > self.ema_htf.iloc[-1]

            if ibs_oversold and bb_break and vol_confirm and htf_bias and not self.position:
                risk_amount = self.equity * 0.02
                stop_distance = self.atr_value.iloc[-1] * self.risk_mult
                position_value = risk_amount / stop_distance
                size_fraction = min(max(position_value / self.equity, 0.005), 0.15)

                sl_price = self.data.Close[-1] - stop_distance
                tp_price = self.data.Close[-1] + (stop_distance * self.tp_rr)
                self.buy(size=size_fraction, sl=sl_price, tp=tp_price)

            # Entry conditions for SHORT
            ibs_overbought = self.ibs_value.iloc[-1] > (1 - self.ibs_threshold)
            bb_break_short = self.data.Close[-1] > self.bb_upper.iloc[-1]
            htf_bias_short = self.data.Close[-1] < self.ema_htf.iloc[-1]

            if ibs_overbought and bb_break_short and vol_confirm and htf_bias_short and not self.position:
                risk_amount = self.equity * 0.02
                stop_distance = self.atr_value.iloc[-1] * self.risk_mult
                position_value = risk_amount / stop_distance
                size_fraction = min(max(position_value / self.equity, 0.005), 0.15)

                sl_price = self.data.Close[-1] + stop_distance
                tp_price = self.data.Close[-1] - (stop_distance * self.tp_rr)
                self.sell(size=size_fraction, sl=sl_price, tp=tp_price)

    # Run both strategies
    try:
        # Pairs strategy
        bt_pairs = Backtest(df, PairsTradingCommoditiesStrategy, cash=10000, commission=0.001)
        result_pairs = bt_pairs.run()

        # Mean reversion strategy
        bt_mr = Backtest(df, MeanReversionCommoditiesStrategy, cash=10000, commission=0.001)
        result_mr = bt_mr.run()

        # Statistical test
        from scipy import stats

        # Extract trade returns (approximate)
        pairs_returns = [result_pairs['Return [%]'] / 100]
        mr_returns = [result_mr['Return [%]'] / 100]

        if len(pairs_returns) > 0 and len(mr_returns) > 0:
            t_stat, p_value = stats.ttest_ind(pairs_returns, mr_returns, equal_var=False)
        else:
            t_stat, p_value = 0, 1

        return {
            'pairs_sharpe': result_pairs['Sharpe Ratio'],
            'mr_sharpe': result_mr['Sharpe Ratio'],
            'pairs_win_rate': result_pairs['Win Rate [%]'] / 100,
            'mr_win_rate': result_mr['Win Rate [%]'] / 100,
            'pairs_return': result_pairs['Return [%]'] / 100,
            'mr_return': result_mr['Return [%]'] / 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'pairs_superior': result_pairs['Sharpe Ratio'] > result_mr['Sharpe Ratio'],
            'significant': p_value < 0.05
        }

    except Exception as e:
        print(f"Error in A/B test: {e}")
        return {}


def run_commodities_pairs_strategy_analysis():
    """Complete commodities pairs trading strategy analysis"""
    print("=== Commodities Pairs Trading Cointegration Strategy Analysis (Oil vs Gold) ===")

    # Load commodities data (Oil daily)
    # Generate synthetic commodities data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', '2024-01-01', freq='D')

    # Oil characteristics (WTI)
    oil_base_price = 50.0
    oil_returns = np.random.normal(0.0001, 0.025, len(dates))  # Higher vol for oil
    # Add some trend and seasonality
    oil_trend = 0.00005 * np.sin(np.arange(len(dates)) * 0.001)
    oil_seasonal = 0.005 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)  # Annual cycle
    oil_final_returns = oil_returns + oil_trend + oil_seasonal
    oil_prices = oil_base_price * np.exp(np.cumsum(oil_final_returns))

    # Gold characteristics
    gold_base_price = 1200.0
    gold_returns = np.random.normal(0.00005, 0.015, len(dates))  # Lower vol for gold
    gold_trend = 0.00002 * np.sin(np.arange(len(dates)) * 0.0005)
    gold_seasonal = 0.002 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    gold_final_returns = gold_returns + gold_trend + gold_seasonal
    gold_prices = gold_base_price * np.exp(np.cumsum(gold_final_returns))

    # Create combined dataframe (using Oil as primary, Gold as secondary)
    df = pd.DataFrame({
        'Open': oil_prices * (1 + np.random.normal(0, 0.002, len(dates))),
        'High': oil_prices * 1.005,
        'Low': oil_prices * 0.995,
        'Close': oil_prices,
        'Volume': np.random.lognormal(12, 1, len(dates)),
        'Gold_Close': gold_prices  # Store gold data for cointegration calc
    }, index=dates)

    # Ensure High >= max(Open, Close), Low <= min(Open, Close)
    df['High'] = np.maximum(df[['Open', 'Close']].max(axis=1), df['High'])
    df['Low'] = np.minimum(df[['Open', 'Close']].min(axis=1), df['Low'])

    print(f"Loaded {len(df)} daily bars of Oil/Gold data")

    # Optimize parameters
    print("Optimizing strategy parameters...")
    best_params = optimize_commodities_pairs_strategy(df)
    print(f"Best parameters: {best_params}")

    # Run walk-forward test
    print("Running walk-forward analysis...")
    wf_results = walk_forward_test_commodities(df, best_params)
    oos_sharpe = wf_results.get('oos_sharpe', 'N/A')
    oos_win_rate = wf_results.get('oos_win_rate', 'N/A')
    print(f"OOS Sharpe: {oos_sharpe if isinstance(oos_sharpe, str) else f'{oos_sharpe:.3f}'}")
    print(
        f"OOS Win Rate: {oos_win_rate if isinstance(oos_win_rate, str) else f'{oos_win_rate:.1%}'}")

    # Run full backtest
    print("Running full backtest...")
    bt = Backtest(df, PairsTradingCommoditiesStrategy, cash=10000, commission=0.001)
    result = bt.run()

    # Calculate metrics
    trades_df = result._trades if hasattr(result, '_trades') else pd.DataFrame()
    if len(trades_df) > 0:
        # Simple metrics calculation
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
    ab_results = ab_test_vs_mean_reversion(df)

    # Save results
    output_dir = Path('results/commodities_pairs')
    output_dir.mkdir(exist_ok=True)

    results = {
        'asset': 'Oil-Gold',
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

    with open(output_dir / 'metrics_commodities_pairs.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        'Commodities Pairs Trading Cointegration Strategy Analysis (Oil vs Gold)',
        fontsize=16)

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
    axes[1, 0].text(0.5, 0.5, 'Monthly Returns\nAnalysis',
                    ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Monthly Returns')

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
    plt.savefig(output_dir / 'commodities_pairs_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Analysis complete. Results saved to {output_dir}/")
    print(f"""
    Final Results for Oil-Gold Pairs:
    - Sharpe Ratio: {result.get('Sharpe Ratio', 'N/A'):.3f}
    - Win Rate: {result.get('Win Rate [%]', 'N/A'):.1f}%
    - Total Return: {result.get('Return [%]', 'N/A'):.1f}%
    - Max Drawdown: {result.get('Max. Drawdown [%]', 'N/A'):.1f}%
    - Number of Trades: {result.get('# Trades', 'N/A')}
    """)

    return results


if __name__ == "__main__":
    # Run analysis for Oil vs Gold pairs
    run_commodities_pairs_strategy_analysis()
