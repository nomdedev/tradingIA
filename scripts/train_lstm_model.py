#!/usr/bin/env python3
"""
LSTM Model Training Script for TradingIA
Trains LSTM models for price prediction using historical data.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.lstm_strategy import LSTMStrategy


def load_data(filepath: str, symbol: str = "BTCUSD") -> pd.DataFrame:
    """Load and preprocess historical data"""
    print(f"Loading data from {filepath}")

    # Read CSV file
    df = pd.read_csv(filepath)

    # Handle different CSV formats
    # If first column is unnamed and looks like datetime, use it as timestamp
    if df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
        df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Ensure required columns exist
    required_cols = ['timestamp', 'open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        required_cols.append('volume')

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Sort by timestamp
    df.sort_index(inplace=True)

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    print(f"Loaded {len(df)} rows of data")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


def prepare_training_data(df: pd.DataFrame, train_split: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and validation sets"""
    split_idx = int(len(df) * train_split)

    train_data = df.iloc[:split_idx]
    val_data = df.iloc[split_idx:]

    print(f"Training data: {len(train_data)} rows")
    print(f"Validation data: {len(val_data)} rows")

    return train_data, val_data


def train_lstm_model(data_file: str, symbol: str = "BTCUSD", epochs: int = 50,
                    sequence_length: int = 60, prediction_horizon: int = 5):
    """Train LSTM model with given parameters"""

    print("=" * 60)
    print("LSTM MODEL TRAINING")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Data file: {data_file}")
    print(f"Sequence length: {sequence_length}")
    print(f"Prediction horizon: {prediction_horizon}")
    print(f"Epochs: {epochs}")
    print()

    # Load data
    df = load_data(data_file, symbol)

    # Prepare training data
    train_data, val_data = prepare_training_data(df)

    # Initialize strategy
    strategy_name = f"LSTM_{symbol}_{sequence_length}_{prediction_horizon}"
    strategy = LSTMStrategy(name=strategy_name)

    # Set custom parameters
    strategy.sequence_length = sequence_length
    strategy.prediction_horizon = prediction_horizon
    strategy.epochs = epochs

    print("\nTraining LSTM model...")
    print("-" * 40)

    # Train model
    start_time = datetime.now()
    history = strategy.train(train_data, validation_split=0.2)
    training_time = datetime.now() - start_time

    print(f"\nTraining completed in {training_time}")
    print(f"Model saved as: {strategy.model_path}")
    print(f"Scaler saved as: {strategy.scaler_path}")

    # Evaluate on validation data
    print("\nEvaluating on validation data...")
    val_signals = strategy.generate_signals(val_data)

    if val_signals:
        # Calculate basic metrics
        buy_signals = [s for s in val_signals if s['signal'] == 'BUY']
        sell_signals = [s for s in val_signals if s['signal'] == 'SELL']

        print(f"Buy signals: {len(buy_signals)}")
        print(f"Sell signals: {len(sell_signals)}")

        if buy_signals:
            avg_buy_confidence = np.mean([s['confidence'] for s in buy_signals])
            print(".3f")
        if sell_signals:
            avg_sell_confidence = np.mean([s['confidence'] for s in sell_signals])
            print(".3f")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return strategy


def main():
    parser = argparse.ArgumentParser(description='Train LSTM model for trading')
    parser.add_argument('--data-file', type=str, default='data/btc_1H.csv',
                       help='Path to historical data file')
    parser.add_argument('--symbol', type=str, default='BTCUSD',
                       help='Trading symbol')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--sequence-length', type=int, default=60,
                       help='LSTM sequence length')
    parser.add_argument('--prediction-horizon', type=int, default=5,
                       help='Prediction horizon in bars')

    args = parser.parse_args()

    try:
        train_lstm_model(
            data_file=args.data_file,
            symbol=args.symbol,
            epochs=args.epochs,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon
        )
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


