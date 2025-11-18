"""
LSTM Trading Strategy for TradingIA Platform
Implements a Long Short-Term Memory neural network for price prediction and trading signals.

Based on the top-performing LSTM strategy from the analysis results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import talib
import os
import joblib
from strategies.base_strategy import BaseStrategy


class LSTMStrategy(BaseStrategy):
    """
    LSTM-based trading strategy that uses deep learning to predict price movements
    and generate trading signals based on historical patterns.
    """

    def __init__(self, name: str = "LSTM_Strategy"):
        super().__init__(name)

        # LSTM Parameters
        self.sequence_length = 60  # Lookback period for LSTM
        self.prediction_horizon = 5  # Steps ahead to predict
        self.confidence_threshold = 0.6  # Minimum confidence for signal

        # Model parameters
        self.lstm_units = 50
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 50

        # Technical indicators for feature engineering
        self.sma_periods = [10, 20, 50]
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2.0

        # Risk management
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04
        self.max_holding_period = 20

        # Model and scaler paths
        self.model_dir = "models"
        self.model_path = os.path.join(self.model_dir, f"{name}_model.h5")
        self.scaler_path = os.path.join(self.model_dir, f"{name}_scaler.pkl")

        # Initialize model and scaler
        self.model = None
        self.scaler = None
        self.is_trained = False

        # Load existing model if available
        self._load_model()

    def _load_model(self):
        """Load pre-trained model and scaler if they exist"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                self.is_trained = True
                print(f"Loaded pre-trained LSTM model from {self.model_path}")

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"Loaded scaler from {self.scaler_path}")

        except Exception as e:
            print(f"Error loading model/scaler: {e}")
            self.model = None
            self.scaler = None
            self.is_trained = False

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and features for LSTM input"""
        df = df.copy()

        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        for period in self.sma_periods:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].pct_change(5)

        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        df['rsi_divergence'] = df['rsi'] - df['rsi'].rolling(20).mean()

        # MACD
        macd, macdsignal, macdhist = talib.MACD(df['close'],
                                               fastperiod=self.macd_fast,
                                               slowperiod=self.macd_slow,
                                               signalperiod=self.macd_signal)
        df['macd'] = macd
        df['macd_signal'] = macdsignal
        df['macd_hist'] = macdhist

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'],
                                           timeperiod=self.bb_period,
                                           nbdevup=self.bb_std,
                                           nbdevdn=self.bb_std)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['volatility'] = df['returns'].rolling(20).std()

        # Volume-based features
        if 'volume' in df.columns:
            df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Target variable: future returns
        df['future_return'] = df['close'].shift(-self.prediction_horizon) / df['close'] - 1

        # Drop NaN values
        df = df.dropna()

        return df

    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training/prediction"""
        X, y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            # Target: 1 if price will go up, 0 if down
            future_return = data[i + self.sequence_length, -1]  # Last column is future_return
            y.append(1 if future_return > 0 else 0)

        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2),
            Dropout(self.dropout_rate),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

        return model

    def train(self, df: pd.DataFrame, validation_split: float = 0.2):
        """Train the LSTM model"""
        print("Training LSTM model...")

        # Create features
        df_features = self._create_features(df)

        if len(df_features) < self.sequence_length + self.prediction_horizon:
            raise ValueError("Not enough data for training")

        # Prepare data for training
        feature_columns = [col for col in df_features.columns
                          if col not in ['future_return'] and not col.startswith('target')]
        data = df_features[feature_columns].values

        # Scale data
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data)

        # Prepare sequences
        X, y = self._prepare_sequences(data_scaled)

        # Build model
        self.model = self._build_model((X.shape[1], X.shape[2]))

        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=1
        )

        self.is_trained = True

        # Save model and scaler
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        print(f"Model trained and saved to {self.model_path}")
        return history

    def predict_signal(self, df: pd.DataFrame) -> Dict:
        """Generate trading signal using LSTM predictions"""
        if not self.is_trained or self.model is None:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'Model not trained'
            }

        try:
            # Create features for recent data
            df_features = self._create_features(df)

            if len(df_features) < self.sequence_length:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Insufficient data'
                }

            # Get latest sequence
            feature_columns = [col for col in df_features.columns
                              if col not in ['future_return'] and not col.startswith('target')]
            recent_data = df_features[feature_columns].tail(self.sequence_length).values

            # Scale data
            if self.scaler is None:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Scaler not available'
                }

            recent_data_scaled = self.scaler.transform(recent_data)

            # Reshape for LSTM input
            X = recent_data_scaled.reshape(1, self.sequence_length, -1)

            # Make prediction
            prediction = self.model.predict(X, verbose=0)[0][0]

            # Determine signal
            if prediction > 0.5 + self.confidence_threshold / 2:
                signal = 'BUY'
                confidence = prediction
            elif prediction < 0.5 - self.confidence_threshold / 2:
                signal = 'SELL'
                confidence = 1 - prediction
            else:
                signal = 'HOLD'
                confidence = 0.5

            return {
                'signal': signal,
                'confidence': float(confidence),
                'prediction': float(prediction),
                'reason': f'LSTM prediction: {prediction:.3f}'
            }

        except Exception as e:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': f'Prediction error: {str(e)}'
            }

    def generate_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Generate trading signals for the entire dataset"""
        signals = []

        if not self.is_trained:
            return signals

        # Process data in windows
        for i in range(self.sequence_length, len(df)):
            window_df = df.iloc[:i+1]
            signal = self.predict_signal(window_df)

            signals.append({
                'timestamp': df.index[i],
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'price': df.iloc[i]['close']
            })

        return signals

    def get_parameters(self) -> Dict:
        """Get strategy parameters"""
        return {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'confidence_threshold': self.confidence_threshold,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_holding_period': self.max_holding_period
        }

    def set_parameters(self, params: Dict):
        """Set strategy parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

