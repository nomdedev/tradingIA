"""
Base Strategy Interface
All strategies must inherit from this base class
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Tuple


class BaseStrategy(ABC):
    """
    Base class for all trading strategies
    
    All strategies must implement:
    - generate_signals(): Return BUY/SELL/HOLD signals
    - get_parameters(): Return strategy parameters
    - set_parameters(): Update strategy parameters
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.parameters = {}
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 'signal' column: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict:
        """
        Get current strategy parameters
        
        Returns:
            Dict with parameter names and values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict) -> None:
        """
        Update strategy parameters
        
        Args:
            params: Dict with parameter names and new values
        """
        pass
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if valid, False otherwise
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in required_cols)
    
    def calculate_signal_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate signal strength (0-5 scale)
        Can be overridden by child classes
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with 'signal_strength' column
        """
        if 'signal' not in df.columns:
            df['signal_strength'] = 0
            return df
            
        # Default: binary strength based on signal
        df['signal_strength'] = df['signal'].apply(
            lambda x: 5 if x != 0 else 0
        )
        return df
    
    def __str__(self):
        return f"{self.name} - Parameters: {self.get_parameters()}"
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}')>"
