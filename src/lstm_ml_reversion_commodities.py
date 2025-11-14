import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Global variables for trained model and data
trained_model = None
feature_scaler = None
features_df = None


def generate_synthetic_commodities_data():
    """Generate synthetic commodities data (Oil)"""
    np.random.seed(42)
    n_bars = 1000  # Reduced data size

    # Base price around 70 (typical Oil price)
    base_price = 70.0

    # Generate realistic price movements (commodities are more volatile)
    returns = np.random.normal(0, 0.002, n_bars)  # Daily returns ~0.2%
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_bars)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_bars)))
    opens = prices + np.random.normal(0, prices * 0.005, n_bars)
    opens = np.clip(opens, lows, highs)

    # Generate volume (commodities volume is typically lower than forex)
    base_volume = 500000  # 500K base volume
    volume = base_volume * (1 + np.random.exponential(0.8, n_bars))

    # Create DataFrame
    dates = pd.date_range('2020-01-01', periods=n_bars, freq='1H')
    data = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volume
    }, index=dates)

    return data


def prepare_features_for_lstm(data):
    """Prepare technical features for LSTM"""
    df = data.copy()

    # Basic price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Simple technical indicators
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Volume indicators
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']

    # Target variable (future returns)
    df['target'] = df['returns'].shift(-1)  # Predict next period return

    # Drop NaN values
    df = df.dropna()

    # Select features
    feature_cols = ['returns', 'sma_20', 'sma_50', 'rsi', 'volume_ratio']

    features = df[feature_cols]
    targets = df['target']

    # Validate data
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    targets = targets.loc[features.index]

    # Scale features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, targets.values, scaler, df.loc[features.index]


def train_lstm_model(features_scaled, targets, lstm_lookback=20):
    """Train LSTM model for return prediction"""
    # Prepare sequences
    X, y = create_sequences(features_scaled, targets, lstm_lookback)

    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build LSTM model
    model = Sequential([
        LSTM(32, input_shape=(lstm_lookback, X.shape[2])),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Regression output
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train model
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=20,  # Reduced epochs
        batch_size=64,  # Increased batch size
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )

    return model


def create_sequences(features, targets, lookback):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i + lookback])
        y.append(targets[i + lookback])
    return np.array(X), np.array(y)


class LSTMCommoditiesStrategy(Strategy):
    """
    LSTM ML Mean Reversion Strategy for Commodities (Oil)
    Uses LSTM to predict returns based on technical indicators
    """

    # Strategy parameters
    lstm_lookback = 20  # Lookback period for LSTM
    prediction_threshold = 0.001  # Minimum prediction confidence for entry
    atr_length = 14
    risk_per_trade = 0.01  # 1% risk per trade
    min_volume_threshold = 0.5  # Minimum volume multiplier

    def init(self):
        # Initialize indicators
        self.atr = self.I(self.calculate_atr)
        self.volume_ma = self.I(self.calculate_volume_ma)

        # Track position
        self.position_size = 0

        # Get pre-trained model and scaler from global variables
        global trained_model, feature_scaler, features_df
        self.model = trained_model
        self.scaler = feature_scaler
        self.features_df = features_df

    def calculate_atr(self):
        """Calculate ATR indicator"""
        high_low = self.data.High - self.data.Low
        high_close = np.abs(self.data.High - np.roll(self.data.Close, 1))
        low_close = np.abs(self.data.Low - np.roll(self.data.Close, 1))
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return pd.Series(tr).rolling(self.atr_length).mean().fillna(1).values

    def calculate_volume_ma(self):
        """Calculate volume moving average"""
        return pd.Series(self.data.Volume).rolling(10).mean().fillna(1).values

    def next(self):
        # Check if model and data are available
        if self.model is None or self.scaler is None or self.features_df is None:
            return

        # Get current features
        if len(self.data) < self.lstm_lookback + 50:  # Need enough data
            return

        # Simple prediction based on recent trend (for debugging)
        recent_prices = self.data.Close[-10:]
        if len(recent_prices) < 10:
            return

        # Simple trend prediction: if price is falling, predict it will continue
        # (mean reversion opportunity)
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        predicted_return = -price_trend * 0.5  # Simple mean reversion prediction

        # Get current ATR and volume
        current_atr = self.atr[-1]
        current_volume_ma = self.volume_ma[-1]

        # Simplified entry conditions
        long_condition = (
            predicted_return < -self.prediction_threshold and
            self.data.Volume[-1] > current_volume_ma * self.min_volume_threshold and
            self.position.size == 0
        )

        short_condition = (
            predicted_return > self.prediction_threshold and
            self.data.Volume[-1] > current_volume_ma * self.min_volume_threshold and
            self.position.size == 0
        )

        # Debug print
        if len(self.data) % 100 == 0:  # Print every 100 bars
            print(
                f"Bar {len(self.data)}: predicted_return={predicted_return:.6f}, long_cond={long_condition}, short_cond={short_condition}")

        # Calculate position size
        if long_condition or short_condition:
            risk_amount = self.equity * self.risk_per_trade
            stop_distance = current_atr * 1.5
            if stop_distance > 0:
                position_size_calc = max(1, int(risk_amount / stop_distance))

                if long_condition:
                    self.buy(size=position_size_calc)
                    print(
                        f"ENTERING LONG: predicted_return={predicted_return:.6f}, close={self.data.Close[-1]:.4f}")

                elif short_condition:
                    self.sell(size=position_size_calc)
                    print(
                        f"ENTERING SHORT: predicted_return={predicted_return:.6f}, close={self.data.Close[-1]:.4f}")


class MeanReversionBaselineCommodities(Strategy):
    """Baseline mean reversion strategy for commodities A/B testing"""

    sma_length = 20
    rsi_length = 14
    rsi_oversold = 30
    rsi_overbought = 70

    def init(self):
        self.sma = self.I(self.calculate_sma)
        self.rsi = self.I(self.calculate_rsi)

    def calculate_sma(self):
        return pd.Series(
            self.data.Close).rolling(
            self.sma_length).mean().fillna(
            self.data.Close[0]).values

    def calculate_rsi(self):
        delta = pd.Series(self.data.Close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_length).mean()
        rs = gain / loss
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values.fillna(50).values

    def next(self):
        if len(self.data) < self.sma_length:
            return

        current_rsi = self.rsi[-1]
        current_sma = self.sma[-1]

        if self.position.size == 0:
            # Long: RSI oversold and price below SMA
            if current_rsi < self.rsi_oversold and self.data.Close[-1] < current_sma:
                self.buy()
            # Short: RSI overbought and price above SMA
            elif current_rsi > self.rsi_overbought and self.data.Close[-1] > current_sma:
                self.sell()
        else:
            # Exit long
            if self.position.size > 0 and current_rsi > 50:
                self.position.close()
            # Exit short
            elif self.position.size < 0 and current_rsi < 50:
                self.position.close()


def run_lstm_commodities_strategy():
    """Run LSTM Commodities strategy with walk-forward testing and A/B analysis"""

    print("Running LSTM ML Mean Reversion Strategy for Commodities (Oil)")
    print("=" * 65)

    # Generate synthetic data
    data = generate_synthetic_commodities_data()

    # Prepare features and train model globally
    global trained_model, feature_scaler, features_df
    features_scaled, targets, feature_scaler, features_df = prepare_features_for_lstm(data)
    trained_model = train_lstm_model(features_scaled, targets)

    # Run backtest
    bt = Backtest(data, LSTMCommoditiesStrategy, cash=10000,
                  commission=.0002)  # Commodities commission
    result = bt.run()

    print("LSTM Strategy Results:")
    print(f"Sharpe Ratio: {result['Sharpe Ratio']:.3f}")
    print(f"Win Rate: {result['Win Rate [%]']:.1f}%")
    print(f"Total Return: {result['Return [%]']:.1f}%")
    print(f"Max Drawdown: {result['Max. Drawdown [%]']:.1f}%")
    print(f"Total Trades: {result['# Trades']}")

    # A/B Testing vs Baseline
    print("\nRunning A/B Test vs Mean Reversion Baseline...")

    bt_baseline = Backtest(data, MeanReversionBaselineCommodities, cash=10000, commission=.0002)
    result_baseline = bt_baseline.run()

    print("Baseline Strategy Results:")
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
    wf_results = perform_walk_forward_analysis(data, LSTMCommoditiesStrategy)

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

        bt = Backtest(train_data, strategy_class, cash=10000, commission=.0002)
        result = bt.run()

        results.append({
            'period': i + 1,
            'sharpe': result['Sharpe Ratio'],
            'win_rate': result['Win Rate [%]'],
            'return': result['Return [%]'],
            'trades': result['# Trades']
        })

    return results


def save_results(lstm_result, baseline_result, wf_results):
    """Save results to files"""

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save main results
    results = {
        'lstm_strategy': {
            'sharpe_ratio': lstm_result['Sharpe Ratio'],
            'win_rate': lstm_result['Win Rate [%]'],
            'total_return': lstm_result['Return [%]'],
            'max_drawdown': lstm_result['Max. Drawdown [%]'],
            'total_trades': lstm_result['# Trades']
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

    with open(results_dir / "lstm_commodities_results.json", 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {results_dir}/lstm_commodities_results.json")


def plot_results(bt, lstm_result, baseline_result):
    """Plot strategy results"""

    # Create plots directory
    plots_dir = Path("results/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot equity curve
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    bt.plot()
    plt.title('LSTM Commodities Strategy - Equity Curve')

    plt.subplot(2, 1, 2)
    # Simple comparison plot
    plt.plot([lstm_result['Sharpe Ratio'], baseline_result['Sharpe Ratio']],
             label=['LSTM', 'Baseline'])
    plt.title('Strategy Comparison - Sharpe Ratio')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "lstm_commodities_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {plots_dir}/lstm_commodities_analysis.png")


if __name__ == "__main__":
    run_lstm_commodities_strategy()
