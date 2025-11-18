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
    def generate_signals(self, df_multi_tf: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Generate trading signals based on market data
        
        Args:
            df_multi_tf: Dictionary with timeframe keys and OHLCV DataFrames
            
        Returns:
            Dict with 'entries', 'exits', 'signals' Series
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
    
    def get_description(self) -> str:
        """
        Get strategy description for users
        
        Returns:
            Human-readable description of the strategy
        """
        return "Trading strategy - no description available"
    
    def get_detailed_info(self) -> Dict:
        """
        Get detailed strategy information for display
        
        Returns:
            Dict with:
                - name: Strategy name
                - description: Brief description
                - buy_signals: How buy signals are generated
                - sell_signals: How sell signals are generated
                - parameters: Current parameters with descriptions
                - risk_level: Conservative/Balanced/Aggressive
                - timeframe: Recommended timeframe
                - indicators: List of indicators used
        """
        return {
            'name': self.name,
            'description': self.get_description(),
            'buy_signals': 'Not specified',
            'sell_signals': 'Not specified',
            'parameters': self.get_parameters(),
            'risk_level': 'Balanced',
            'timeframe': '5min',
            'indicators': []
        }
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if valid, False otherwise
        """
        required_cols_lower = ['open', 'high', 'low', 'close', 'volume']
        required_cols_upper = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check for lowercase columns
        has_lower = all(col in df.columns for col in required_cols_lower)
        # Check for uppercase columns  
        has_upper = all(col in df.columns for col in required_cols_upper)
        
        return has_lower or has_upper
    
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
