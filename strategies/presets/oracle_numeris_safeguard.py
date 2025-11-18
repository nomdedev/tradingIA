"""
Oracle Numeris and Safeguard Strategy
Advanced strategy combining numerical oracle predictions with safeguard risk management
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_strategy import BaseStrategy


class OracleNumerisSafeguardStrategy(BaseStrategy):
    """
    Oracle Numeris and Safeguard strategy

    Features:
    - Oracle Numeris: Numerical prediction using statistical models
    - Safeguard: Advanced risk management with dynamic stops and position sizing

    Signals:
    - BUY: Oracle predicts upward movement with safeguard confirmation
    - SELL: Oracle predicts downward movement with safeguard confirmation
    - HOLD: Otherwise
    """

    def __init__(self):
        super().__init__(name="Oracle Numeris Safeguard")
        self.parameters = {
            # Oracle Numeris parameters
            'oracle_window': 20,  # Lookback window for predictions
            'oracle_threshold': 0.02,  # Minimum prediction confidence
            'numeris_smoothing': 5,  # Smoothing period for numerical calculations

            # Safeguard parameters
            'safeguard_atr_period': 14,  # ATR period for volatility
            'safeguard_stop_mult': 1.5,  # Stop loss multiplier
            'safeguard_profit_mult': 2.0,  # Take profit multiplier
            'safeguard_max_drawdown': 0.05,  # Maximum drawdown threshold

            # Signal filters
            'require_volume_confirmation': True,
            'min_volume_ratio': 1.2,  # Minimum volume ratio vs average
            'trend_filter_period': 50  # Trend filter MA period
        }

    def generate_signals(self, df_multi_tf: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Generate Oracle Numeris signals with Safeguard protection"""
        # Use 5min data for signals
        df = df_multi_tf.get('5min', df_multi_tf.get('5m', list(df_multi_tf.values())[0]))

        if not self.validate_data(df):
            raise ValueError("Invalid data format")

        df = df.copy()

        # Calculate Oracle Numeris predictions
        df = self._calculate_oracle_numeris(df)

        # Calculate Safeguard risk metrics
        df = self._calculate_safeguard_metrics(df)

        # Apply signal filters
        df = self._apply_filters(df)

        # Generate final signals
        df['signal'] = 0

        # BUY signal: Oracle predicts up + Safeguard allows
        buy_condition = (
            (df['oracle_prediction'] > self.parameters['oracle_threshold']) &
            (df['safeguard_risk_score'] < 0.7) &
            (df['volume_confirmed'] if self.parameters['require_volume_confirmation'] else True)
        )
        df.loc[buy_condition, 'signal'] = 1

        # SELL signal: Oracle predicts down + Safeguard allows
        sell_condition = (
            (df['oracle_prediction'] < -self.parameters['oracle_threshold']) &
            (df['safeguard_risk_score'] < 0.7) &
            (df['volume_confirmed'] if self.parameters['require_volume_confirmation'] else True)
        )
        df.loc[sell_condition, 'signal'] = -1

        # Calculate signal strength
        df['signal_strength'] = 0
        signal_mask = df['signal'] != 0
        if signal_mask.any():
            # Strength based on oracle confidence and safeguard score
            confidence = abs(df.loc[signal_mask, 'oracle_prediction'])
            risk_score = df.loc[signal_mask, 'safeguard_risk_score']
            df.loc[signal_mask, 'signal_strength'] = (
                (confidence * (1 - risk_score)) * 100
            ).clip(1, 5)

        # Return signals in expected format
        return {
            'entries': (df['signal'] == 1).astype(int),
            'exits': (df['signal'] == -1).astype(int),
            'signals': df['signal']
        }

    def _calculate_oracle_numeris(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Oracle Numeris predictions"""
        window = self.parameters['oracle_window']

        # Calculate returns
        df['returns'] = df['close'].pct_change()

        # Oracle prediction using linear regression on price momentum
        for i in range(window, len(df)):
            # Simple linear regression over window
            y = df['close'].iloc[i-window:i].values
            x = np.arange(len(y))

            # Calculate slope (trend)
            slope = np.polyfit(x, y, 1)[0]

            # Normalize by current price for prediction strength
            df.loc[df.index[i], 'oracle_prediction'] = slope / df['close'].iloc[i]

        # Smooth the predictions
        smoothing = self.parameters['numeris_smoothing']
        df['oracle_prediction'] = df['oracle_prediction'].rolling(window=smoothing).mean()

        # Fill NaN values
        df['oracle_prediction'] = df['oracle_prediction'].fillna(0)

        return df

    def _calculate_safeguard_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Safeguard risk management metrics"""
        # Calculate ATR for volatility
        atr_period = self.parameters['safeguard_atr_period']
        df['atr'] = self._calculate_atr(df, atr_period)

        # Calculate drawdown
        df['peak'] = df['close'].expanding().max()
        df['drawdown'] = (df['close'] - df['peak']) / df['peak']

        # Safeguard risk score (0-1, higher = more risky)
        df['safeguard_risk_score'] = 0.0

        # Risk based on ATR volatility
        atr_risk = df['atr'] / df['close']
        df['safeguard_risk_score'] += atr_risk * 0.4

        # Risk based on drawdown
        drawdown_risk = abs(df['drawdown'])
        df['safeguard_risk_score'] += drawdown_risk * 0.6

        # Cap at 1.0
        df['safeguard_risk_score'] = df['safeguard_risk_score'].clip(0, 1)

        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply additional signal filters"""
        # Volume confirmation
        if self.parameters['require_volume_confirmation']:
            avg_volume = df['volume'].rolling(window=20).mean()
            df['volume_confirmed'] = df['volume'] > (avg_volume * self.parameters['min_volume_ratio'])
        else:
            df['volume_confirmed'] = True

        # Trend filter
        trend_period = self.parameters['trend_filter_period']
        df['trend_ma'] = df['close'].rolling(window=trend_period).mean()
        df['trend_up'] = df['close'] > df['trend_ma']

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def get_parameters(self) -> Dict:
        """Get strategy parameters"""
        return self.parameters.copy()

    def set_parameters(self, params: Dict) -> None:
        """Update strategy parameters"""
        for key, value in params.items():
            if key in self.parameters:
                self.parameters[key] = value
    
    def get_description(self) -> str:
        """Get strategy description"""
        return (
            "Estrategia avanzada que combina Oracle Numeris (predicción numérica) "
            "con Safeguard (gestión de riesgo dinámica). Usa regresión lineal para predecir movimientos "
            "y un sistema de puntuación de riesgo basado en ATR y drawdown."
        )
    
    def get_detailed_info(self) -> Dict:
        """Get detailed strategy information"""
        return {
            'name': self.name,
            'description': self.get_description(),
            'buy_signals': (
                "📈 COMPRA cuando:\n"
                f"  • Oracle predice movimiento alcista (>+{self.parameters['oracle_threshold']*100}%)\n"
                "  • Safeguard: puntuación de riesgo baja (<0.7)\n"
                f"  • Opcionalmente: volumen {self.parameters['min_volume_ratio']}x sobre promedio\n"
                "  • Sistema de predicción basado en regresión lineal"
            ),
            'sell_signals': (
                "📉 VENTA cuando:\n"
                f"  • Oracle predice movimiento bajista (<-{self.parameters['oracle_threshold']*100}%)\n"
                "  • Safeguard: puntuación de riesgo baja (<0.7)\n"
                f"  • Opcionalmente: volumen {self.parameters['min_volume_ratio']}x sobre promedio\n"
                "  • Protección contra alta volatilidad y drawdown"
            ),
            'parameters': {
                'oracle_window': f"{self.parameters['oracle_window']} - Ventana para predicciones Oracle",
                'oracle_threshold': f"{self.parameters['oracle_threshold']*100}% - Umbral de confianza Oracle",
                'numeris_smoothing': f"{self.parameters['numeris_smoothing']} - Período suavizado Numeris",
                'safeguard_atr_period': f"{self.parameters['safeguard_atr_period']} - Período ATR Safeguard",
                'safeguard_stop_mult': f"{self.parameters['safeguard_stop_mult']}x - Multiplicador stop loss",
                'safeguard_profit_mult': f"{self.parameters['safeguard_profit_mult']}x - Multiplicador take profit",
                'safeguard_max_drawdown': f"{self.parameters['safeguard_max_drawdown']*100}% - Drawdown máximo",
                'require_volume_confirmation': f"{self.parameters['require_volume_confirmation']} - Confirmación de volumen",
                'min_volume_ratio': f"{self.parameters['min_volume_ratio']}x - Ratio mínimo de volumen",
                'trend_filter_period': f"{self.parameters['trend_filter_period']} - Período filtro tendencia"
            },
            'risk_level': 'Equilibrado',
            'timeframe': '5min',
            'indicators': ['Linear Regression', 'ATR', 'Drawdown Monitor', 'Volume MA', 'Trend MA']
        }


# Create preset configurations
PRESETS = {
    'conservative': {
        'oracle_window': 25,
        'oracle_threshold': 0.03,
        'numeris_smoothing': 7,
        'safeguard_atr_period': 20,
        'safeguard_stop_mult': 1.2,
        'safeguard_profit_mult': 1.5,
        'safeguard_max_drawdown': 0.03,
        'require_volume_confirmation': True,
        'min_volume_ratio': 1.5,
        'trend_filter_period': 100
    },
    'balanced': {
        'oracle_window': 20,
        'oracle_threshold': 0.02,
        'numeris_smoothing': 5,
        'safeguard_atr_period': 14,
        'safeguard_stop_mult': 1.5,
        'safeguard_profit_mult': 2.0,
        'safeguard_max_drawdown': 0.05,
        'require_volume_confirmation': True,
        'min_volume_ratio': 1.2,
        'trend_filter_period': 50
    },
    'aggressive': {
        'oracle_window': 15,
        'oracle_threshold': 0.015,
        'numeris_smoothing': 3,
        'safeguard_atr_period': 10,
        'safeguard_stop_mult': 1.8,
        'safeguard_profit_mult': 2.5,
        'safeguard_max_drawdown': 0.08,
        'require_volume_confirmation': False,
        'min_volume_ratio': 1.0,
        'trend_filter_period': 25
    }
}


if __name__ == "__main__":
    # Test strategy
    print("Oracle Numeris and Safeguard Strategy")
    print("=" * 50)

    strategy = OracleNumerisSafeguardStrategy()
    print(f"\nStrategy: {strategy}")
    print(f"Parameters: {strategy.get_parameters()}")

    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=300, freq='1h')
    df = pd.DataFrame({
        'open': np.random.randn(300).cumsum() + 100,
        'high': np.random.randn(300).cumsum() + 102,
        'low': np.random.randn(300).cumsum() + 98,
        'close': np.random.randn(300).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 300)
    }, index=dates)

    # Generate signals
    result = strategy.generate_signals({'5min': df})

    print(f"\nGenerated {len(result['signals'][result['signals'] != 0])} signals")
    print(f"BUY signals: {len(result['entries'][result['entries'] == 1])}")
    print(f"SELL signals: {len(result['exits'][result['exits'] == 1])}")