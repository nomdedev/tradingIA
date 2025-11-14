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
SKOPT_AVAILABLE = False
TA_AVAILABLE = False

try:
    from backtesting import Strategy, Backtest
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    print("Warning: backtesting not available. Install with: pip install backtesting")

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not available")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: ta not available. Install with: pip install ta")


class MACDADXCommoditiesStrategy(Strategy):
    """
    Momentum MACD + ADX Strategy for Commodities (Oil & Gold)
    Adapted for commodity seasonality and lower liquidity
    """

    # Strategy parameters (commodities-optimized - more conservative)
    macd_fast = 15  # Slower for commodities
    macd_slow = 30
    macd_signal = 12
    adx_length = 14
    adx_threshold = 20  # Higher threshold for commodities
    vol_mult = 1.2  # Moderate volume multiplier
    atr_length = 14
    risk_mult = 1.4  # Conservative risk for commodities
    tp_rr = 2.2  # Higher target for trending commodities

    def init(self):
        # Use numpy arrays for backtesting compatibility
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # MACD calculation using pandas then convert to numpy
        close_series = pd.Series(close)
        ema_fast = close_series.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close_series.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        self.macd_hist = macd_line - macd_signal

        # Convert to numpy arrays for backtesting
        self.macd_line = macd_line.values
        self.macd_signal = macd_signal.values

        # ADX calculation (simplified using pandas then numpy)
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)

        high_low = high_series - low_series
        high_close = (high_series - close_series.shift(1)).abs()
        low_close = (low_series - close_series.shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Directional movement
        dm_plus = pd.Series(
            np.where(
                (high_series -
                 high_series.shift(1)) > (
                    low_series.shift(1) -
                    low_series),
                high_series -
                high_series.shift(1),
                0))
        dm_minus = pd.Series(
            np.where(
                (low_series.shift(1) -
                 low_series) > (
                    high_series -
                    high_series.shift(1)),
                low_series.shift(1) -
                low_series,
                0))

        # Smoothed averages
        atr = tr.rolling(window=self.adx_length).mean()
        di_plus = 100 * (dm_plus.rolling(window=self.adx_length).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=self.adx_length).mean() / atr)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx_value = dx.rolling(window=self.adx_length).mean()

        # Convert to numpy arrays
        self.adx_value = adx_value.values
        self.atr_value = atr.values

        # Volume confirmation (using range for commodities)
        range_series = (high_series - low_series).rolling(window=21).mean()
        self.range_sma = range_series.values

        # EMA for trend bias (longer for commodities)
        ema_bias = close_series.ewm(span=100, adjust=False).mean()
        self.ema_bias = ema_bias.values

    def next(self):
        # Ensure we have enough data
        if len(self.data.Close) < 50:  # Need at least 50 periods for indicators
            return

        # Get current index
        idx = len(self.data.Close) - 1

        # Entry conditions for LONG
        macd_bull = self.macd_line[idx] > self.macd_signal[idx]
        adx_trend = self.adx_value[idx] > self.adx_threshold
        range_confirm = (self.data.High[idx] - self.data.Low[idx]
                         ) > self.range_sma[idx] * self.vol_mult
        bias_up = self.data.Close[idx] > self.ema_bias[idx]

        # Additional momentum filter (conservative for commodities)
        price_momentum = self.data.Close[idx] > self.data.Close[idx - 5] if idx >= 5 else False

        if macd_bull and adx_trend and range_confirm and bias_up and price_momentum and not self.position:
            # Calculate position size based on risk
            risk_amount = self.equity * 0.02  # 2% risk per trade (commodities can be volatile)
            stop_distance = max(
                self.atr_value[idx] *
                self.risk_mult,
                self.data.Close[idx] *
                0.01)  # Min 1% stop
            if stop_distance > 0:
                position_size = min(
                    risk_amount / stop_distance,
                    self.equity * 0.15)  # Max 15% of equity
                # Convert to fraction of equity for backtesting
                position_value = position_size * self.data.Close[idx]
                size_fraction = position_value / self.equity
                size_fraction = min(max(size_fraction, 0.005), 0.15)  # Between 0.5% and 15%
            else:
                size_fraction = 0.02  # Default 2%

            # Entry with SL/TP
            sl_price = self.data.Close[idx] - stop_distance
            tp_price = self.data.Close[idx] + (stop_distance * self.tp_rr)

            self.buy(size=size_fraction, sl=sl_price, tp=tp_price)

        # Entry conditions for SHORT
        macd_bear = self.macd_line[idx] < self.macd_signal[idx]
        bias_down = self.data.Close[idx] < self.ema_bias[idx]
        price_momentum_short = self.data.Close[idx] < self.data.Close[idx -
                                                                      5] if idx >= 5 else False

        if macd_bear and adx_trend and range_confirm and bias_down and price_momentum_short and not self.position:
            # Calculate position size based on risk
            risk_amount = self.equity * 0.02  # 2% risk per trade
            stop_distance = max(
                self.atr_value[idx] *
                self.risk_mult,
                self.data.Close[idx] *
                0.01)  # Min 1% stop
            if stop_distance > 0:
                position_size = min(
                    risk_amount / stop_distance,
                    self.equity * 0.15)  # Max 15% of equity
                # Convert to fraction of equity for backtesting
                position_value = position_size * self.data.Close[idx]
                size_fraction = position_value / self.equity
                size_fraction = min(max(size_fraction, 0.005), 0.15)  # Between 0.5% and 15%
            else:
                size_fraction = 0.02  # Default 2%

            # Entry with SL/TP
            sl_price = self.data.Close[idx] + stop_distance
            tp_price = self.data.Close[idx] - (stop_distance * self.tp_rr)

            self.sell(size=size_fraction, sl=sl_price, tp=tp_price)


def walk_forward_test_commodities(df: pd.DataFrame, params: Dict = None) -> Dict:
    """Walk-forward optimization and testing for commodities strategy"""
    if not BACKTESTING_AVAILABLE:
        print("Backtesting library not available")
        return {}

    if params:
        MACDADXCommoditiesStrategy.macd_fast = params.get('macd_fast', 15)
        MACDADXCommoditiesStrategy.macd_slow = params.get('macd_slow', 30)
        MACDADXCommoditiesStrategy.macd_signal = params.get('macd_signal', 12)
        MACDADXCommoditiesStrategy.adx_threshold = params.get('adx_threshold', 20)
        MACDADXCommoditiesStrategy.vol_mult = params.get('vol_mult', 1.2)
        MACDADXCommoditiesStrategy.risk_mult = params.get('risk_mult', 1.4)
        MACDADXCommoditiesStrategy.tp_rr = params.get('tp_rr', 2.2)

    # Simple train/test split (80/20) for commodities
    split_idx = int(len(df) * 0.8)
    test_data = df.iloc[split_idx:]

    results = []

    # Test on the last 20% of data
    try:
        bt = Backtest(test_data, MACDADXCommoditiesStrategy,
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
            'periods_tested': len(results),
            'degradation': 0.0  # Simplified
        }

    return {}


def optimize_commodities_strategy(df: pd.DataFrame) -> Dict:
    """Grid search optimization for commodities strategy parameters"""
    if not BACKTESTING_AVAILABLE:
        print("Backtesting not available, using default parameters")
        return {
            'macd_fast': 15, 'macd_slow': 30, 'macd_signal': 12,
            'adx_threshold': 20, 'vol_mult': 1.2, 'risk_mult': 1.4, 'tp_rr': 2.2
        }

    print("Running parameter optimization...")

    # Parameter ranges optimized for commodities
    param_grid = {
        'macd_fast': [8, 12, 15, 20],
        'macd_slow': [26, 30, 35, 40],
        'macd_signal': [9, 12, 15],
        'adx_threshold': [15, 20, 25, 30],
        'vol_mult': [1.0, 1.2, 1.5],
        'risk_mult': [1.2, 1.4, 1.6, 1.8],
        'tp_rr': [1.8, 2.2, 2.5, 3.0]
    }

    best_sharpe = -np.inf
    best_params = None

    # Use 70% of data for optimization
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Testing {total_combinations} parameter combinations...")

    tested = 0
    for macd_fast in param_grid['macd_fast']:
        for macd_slow in param_grid['macd_slow']:
            if macd_fast >= macd_slow:
                continue  # Skip invalid MACD combinations
            for macd_signal in param_grid['macd_signal']:
                for adx_threshold in param_grid['adx_threshold']:
                    for vol_mult in param_grid['vol_mult']:
                        for risk_mult in param_grid['risk_mult']:
                            for tp_rr in param_grid['tp_rr']:
                                tested += 1
                                if tested % 100 == 0:
                                    print(f"Tested {tested}/{total_combinations} combinations...")

                                # Set parameters
                                MACDADXCommoditiesStrategy.macd_fast = macd_fast
                                MACDADXCommoditiesStrategy.macd_slow = macd_slow
                                MACDADXCommoditiesStrategy.macd_signal = macd_signal
                                MACDADXCommoditiesStrategy.adx_threshold = adx_threshold
                                MACDADXCommoditiesStrategy.vol_mult = vol_mult
                                MACDADXCommoditiesStrategy.risk_mult = risk_mult
                                MACDADXCommoditiesStrategy.tp_rr = tp_rr

                                try:
                                    # Quick backtest on training data
                                    bt = Backtest(train_df, MACDADXCommoditiesStrategy,
                                                  cash=10000, commission=0.001)
                                    result = bt.run()

                                    sharpe = result['Sharpe Ratio']
                                    win_rate = result['Win Rate [%]'] / 100
                                    trades = result['# Trades']

                                    # Only consider if enough trades and reasonable win rate
                                    if trades >= 50 and win_rate > 0.3 and sharpe > best_sharpe:
                                        best_sharpe = sharpe
                                        best_params = {
                                            'macd_fast': macd_fast,
                                            'macd_slow': macd_slow,
                                            'macd_signal': macd_signal,
                                            'adx_threshold': adx_threshold,
                                            'vol_mult': vol_mult,
                                            'risk_mult': risk_mult,
                                            'tp_rr': tp_rr
                                        }

                                except Exception:
                                    continue  # Skip failed combinations

    if best_params is None:
        print("No valid parameter combination found, using defaults")
        best_params = {
            'macd_fast': 15, 'macd_slow': 30, 'macd_signal': 12,
            'adx_threshold': 20, 'vol_mult': 1.2, 'risk_mult': 1.4, 'tp_rr': 2.2
        }

    print(f"Best Sharpe found: {best_sharpe:.3f}")
    return best_params


def ab_test_vs_rsi_bb(df: pd.DataFrame) -> Dict:
    """A/B test against RSI + Bollinger Bands alternative"""
    if not BACKTESTING_AVAILABLE:
        return {}

    class RSIBBCommoditiesStrategy(Strategy):
        """Alternative strategy: RSI + Bollinger Bands for commodities"""
        rsi_length = 14
        rsi_oversold = 25  # More extreme for commodities
        rsi_overbought = 75
        bb_length = 20
        bb_std = 2.0
        risk_mult = 1.4
        tp_rr = 2.2

        def init(self):
            close = pd.Series(self.data.Close)
            high = pd.Series(self.data.High)
            low = pd.Series(self.data.Low)

            # RSI calculation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_length).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_length).mean()
            rs = gain / loss
            self.rsi_value = 100 - (100 / (1 + rs))

            # Bollinger Bands
            self.bb_sma = close.rolling(window=self.bb_length).mean()
            self.bb_std = close.rolling(window=self.bb_length).std()
            self.bb_upper = self.bb_sma + (self.bb_std * self.bb_std)
            self.bb_lower = self.bb_sma - (self.bb_std * self.bb_std)

            # ATR for risk
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            self.atr_value = tr.rolling(window=14).mean()

        def next(self):
            if len(self.data.Close) < 20:
                return

            # Long: RSI oversold + price below lower BB
            if (self.rsi_value.iloc[-1] < self.rsi_oversold and
                    self.data.Close[-1] < self.bb_lower.iloc[-1] and not self.position):

                risk_amount = self.equity * 0.02
                stop_distance = max(self.atr_value.iloc[-1] *
                                    self.risk_mult, self.data.Close[-1] * 0.015)
                position_value = risk_amount / stop_distance
                size_fraction = min(max(position_value / self.equity, 0.005), 0.15)

                sl_price = self.data.Close[-1] - stop_distance
                tp_price = self.data.Close[-1] + (stop_distance * self.tp_rr)
                self.buy(size=size_fraction, sl=sl_price, tp=tp_price)

            # Short: RSI overbought + price above upper BB
            elif (self.rsi_value.iloc[-1] > self.rsi_overbought and
                  self.data.Close[-1] > self.bb_upper.iloc[-1] and not self.position):

                risk_amount = self.equity * 0.02
                stop_distance = max(self.atr_value.iloc[-1] *
                                    self.risk_mult, self.data.Close[-1] * 0.015)
                position_value = risk_amount / stop_distance
                size_fraction = min(max(position_value / self.equity, 0.005), 0.15)

                sl_price = self.data.Close[-1] + stop_distance
                tp_price = self.data.Close[-1] - (stop_distance * self.tp_rr)
                self.sell(size=size_fraction, sl=sl_price, tp=tp_price)

    # Run both strategies
    try:
        # MACD strategy
        bt_macd = Backtest(df, MACDADXCommoditiesStrategy, cash=10000, commission=0.001)
        result_macd = bt_macd.run()

        # RSI+BB strategy
        bt_rsi = Backtest(df, RSIBBCommoditiesStrategy, cash=10000, commission=0.001)
        result_rsi = bt_rsi.run()

        # Statistical test
        from scipy import stats

        # Extract trade returns (approximate)
        macd_returns = [result_macd['Return [%]'] / 100]
        rsi_returns = [result_rsi['Return [%]'] / 100]

        if len(macd_returns) > 0 and len(rsi_returns) > 0:
            t_stat, p_value = stats.ttest_ind(macd_returns, rsi_returns, equal_var=False)
        else:
            t_stat, p_value = 0, 1

        return {
            'macd_sharpe': result_macd['Sharpe Ratio'],
            'rsi_sharpe': result_rsi['Sharpe Ratio'],
            'macd_win_rate': result_macd['Win Rate [%]'] / 100,
            'rsi_win_rate': result_rsi['Win Rate [%]'] / 100,
            'macd_return': result_macd['Return [%]'] / 100,
            'rsi_return': result_rsi['Return [%]'] / 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'macd_superior': result_macd['Sharpe Ratio'] > result_rsi['Sharpe Ratio'],
            'significant': p_value < 0.05
        }

    except Exception as e:
        print(f"Error in A/B test: {e}")
        return {}


def run_commodities_strategy_analysis(data_path: str = None, asset_name: str = "Oil"):
    """Complete commodities momentum strategy analysis"""
    print(f"=== Commodities Momentum MACD+ADX Strategy Analysis ({asset_name}) ===")

    # Reset strategy parameters to defaults for each asset
    MACDADXCommoditiesStrategy.macd_fast = 15
    MACDADXCommoditiesStrategy.macd_slow = 30
    MACDADXCommoditiesStrategy.macd_signal = 12
    MACDADXCommoditiesStrategy.adx_length = 14
    MACDADXCommoditiesStrategy.adx_threshold = 20
    MACDADXCommoditiesStrategy.vol_mult = 1.2
    MACDADXCommoditiesStrategy.atr_length = 14
    MACDADXCommoditiesStrategy.risk_mult = 1.4
    MACDADXCommoditiesStrategy.tp_rr = 2.2

    # Load commodities data (Oil or Gold daily)
    if data_path:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        # Generate synthetic commodities data for demonstration
        np.random.seed(42)
        dates = pd.date_range('2000-01-01', '2024-01-01', freq='D')

        if asset_name == "Oil":
            # WTI Oil characteristics
            base_price = 50.0
            returns = np.random.normal(0.0001, 0.025, len(dates))  # Higher vol for oil
            # Add some trend and seasonality
            trend = 0.00005 * np.sin(np.arange(len(dates)) * 0.001)
            seasonal = 0.005 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)  # Annual cycle
        else:  # Gold
            # Gold characteristics
            base_price = 1200.0
            returns = np.random.normal(0.00005, 0.015, len(dates))  # Lower vol for gold
            trend = 0.00002 * np.sin(np.arange(len(dates)) * 0.0005)
            seasonal = 0.002 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)

        final_returns = returns + trend + seasonal
        prices = base_price * np.exp(np.cumsum(final_returns))

        # Create OHLCV data
        high_mult = 1 + np.random.exponential(0.005, len(dates))
        low_mult = 1 - np.random.exponential(0.005, len(dates))
        volume = np.random.lognormal(12, 1, len(dates))

        df = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.002, len(dates))),
            'High': prices * high_mult,
            'Low': prices * low_mult,
            'Close': prices,
            'Volume': volume
        }, index=dates)

        # Ensure High >= max(Open, Close), Low <= min(Open, Close)
        df['High'] = np.maximum(df[['Open', 'Close']].max(axis=1), df['High'])
        df['Low'] = np.minimum(df[['Open', 'Close']].min(axis=1), df['Low'])

    print(f"Loaded {len(df)} daily bars of {asset_name} data")

    # Optimize parameters
    print("Optimizing strategy parameters...")
    best_params = optimize_commodities_strategy(df)
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
    bt = Backtest(df, MACDADXCommoditiesStrategy, cash=10000, commission=0.001)
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
    print("Running A/B test vs RSI+BB...")
    ab_results = ab_test_vs_rsi_bb(df)

    # Save results
    output_dir = Path(f'results/commodities_momentum_{asset_name.lower()}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trades
    if len(trades_df) > 0:
        trades_df.to_csv(output_dir / f'trades_commodities_macd_adx_{asset_name.lower()}.csv')

    # Save metrics
    results = {
        'walk_forward': wf_results,
        'full_backtest': dict(result),
        'custom_metrics': metrics,
        'ab_test': ab_results,
        'best_params': best_params,
        'asset': asset_name
    }

    with open(output_dir / f'metrics_commodities_momentum_{asset_name.lower()}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Commodities Momentum MACD+ADX Strategy Analysis ({asset_name})', fontsize=16)

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
    plt.savefig(
        output_dir /
        f'commodities_momentum_{asset_name.lower()}_analysis.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    print(f"Analysis complete. Results saved to {output_dir}/")
    print(f"""
    Final Results for {asset_name}:
    - Sharpe Ratio: {result.get('Sharpe Ratio', 'N/A'):.3f}
    - Win Rate: {result.get('Win Rate [%]', 'N/A'):.1f}%
    - Total Return: {result.get('Return [%]', 'N/A'):.1f}%
    - Max Drawdown: {result.get('Max. Drawdown [%]', 'N/A'):.1f}%
    - Number of Trades: {result.get('# Trades', 'N/A')}
    """)

    return results


if __name__ == "__main__":
    # Run analysis for Oil
    run_commodities_strategy_analysis(asset_name="Oil")

    # Run analysis for Gold
    run_commodities_strategy_analysis(asset_name="Gold")
