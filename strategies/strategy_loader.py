"""
Strategy Loader and Manager
Load pre-configured strategies from the presets folder
"""

import os
import sys
import importlib.util
from typing import Dict, List, Optional
import inspect


class StrategyLoader:
    """
    Load and manage trading strategies from presets folder
    """
    
    def __init__(self, presets_path: Optional[str] = None):
        """
        Initialize strategy loader
        
        Args:
            presets_path: Path to presets folder (default: ./presets/)
        """
        if presets_path is None:
            self.presets_path = os.path.join(
                os.path.dirname(__file__), 
                'presets'
            )
        else:
            self.presets_path = presets_path
        
        self.available_strategies = {}
        self._scan_strategies()
    
    def _scan_strategies(self):
        """Scan presets folder for available strategies"""
        if not os.path.exists(self.presets_path):
            print(f"Warning: Presets path not found: {self.presets_path}")
            return
        
        for filename in os.listdir(self.presets_path):
            if filename.endswith('.py') and not filename.startswith('__'):
                strategy_name = filename[:-3]  # Remove .py
                module_path = os.path.join(self.presets_path, filename)
                
                try:
                    # Load module
                    spec = importlib.util.spec_from_file_location(
                        strategy_name, 
                        module_path
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find strategy class (inherits from BaseStrategy)
                    strategy_class = None
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (hasattr(obj, 'generate_signals') and 
                            name != 'BaseStrategy' and
                            not name.startswith('_')):
                            strategy_class = obj
                            break
                    
                    if strategy_class:
                        # Get presets if available
                        presets = getattr(module, 'PRESETS', {})
                        
                        self.available_strategies[strategy_name] = {
                            'class': strategy_class,
                            'module': module,
                            'presets': presets,
                            'path': module_path
                        }
                
                except Exception as e:
                    print(f"Error loading strategy {strategy_name}: {e}")
    
    def list_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.available_strategies.keys())
    
    def get_strategy(self, strategy_name: str, preset: Optional[str] = None):
        """
        Load and instantiate a strategy
        
        Args:
            strategy_name: Name of strategy (filename without .py)
            preset: Optional preset configuration name
            
        Returns:
            Instantiated strategy object
        """
        if strategy_name not in self.available_strategies:
            raise ValueError(
                f"Strategy '{strategy_name}' not found. "
                f"Available: {self.list_strategies()}"
            )
        
        strategy_info = self.available_strategies[strategy_name]
        strategy = strategy_info['class']()
        
        # Apply preset if specified
        if preset:
            if preset not in strategy_info['presets']:
                raise ValueError(
                    f"Preset '{preset}' not found for {strategy_name}. "
                    f"Available: {list(strategy_info['presets'].keys())}"
                )
            strategy.set_parameters(strategy_info['presets'][preset])
        
        return strategy
    
    def get_presets(self, strategy_name: str) -> Dict:
        """
        Get available presets for a strategy
        
        Args:
            strategy_name: Name of strategy
            
        Returns:
            Dict of preset configurations
        """
        if strategy_name not in self.available_strategies:
            return {}
        
        return self.available_strategies[strategy_name]['presets']
    
    def get_strategy_info(self, strategy_name: str) -> Dict:
        """
        Get detailed information about a strategy
        
        Args:
            strategy_name: Name of strategy
            
        Returns:
            Dict with strategy information
        """
        if strategy_name not in self.available_strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy_info = self.available_strategies[strategy_name]
        strategy = strategy_info['class']()
        
        return {
            'name': strategy.name,
            'class_name': strategy_info['class'].__name__,
            'parameters': strategy.get_parameters(),
            'presets': list(strategy_info['presets'].keys()),
            'path': strategy_info['path']
        }


# Global loader instance
_loader = None

def get_loader() -> StrategyLoader:
    """Get global strategy loader instance"""
    global _loader
    if _loader is None:
        _loader = StrategyLoader()
    return _loader


def load_strategy(strategy_name: str, preset: Optional[str] = None):
    """
    Convenience function to load a strategy
    
    Args:
        strategy_name: Name of strategy
        preset: Optional preset name
        
    Returns:
        Instantiated strategy
    """
    return get_loader().get_strategy(strategy_name, preset)


def list_available_strategies() -> List[str]:
    """Get list of all available strategies"""
    return get_loader().list_strategies()


if __name__ == "__main__":
    print("=" * 60)
    print("STRATEGY LOADER - Available Strategies")
    print("=" * 60)
    
    loader = StrategyLoader()
    
    strategies = loader.list_strategies()
    print(f"\nFound {len(strategies)} strategies:\n")
    
    for strategy_name in strategies:
        try:
            info = loader.get_strategy_info(strategy_name)
            print(f"📊 {info['name']}")
            print(f"   Class: {info['class_name']}")
            print(f"   Presets: {', '.join(info['presets']) if info['presets'] else 'None'}")
            print(f"   Parameters: {len(info['parameters'])} configurable")
            print()
        except Exception as e:
            print(f"⚠️  {strategy_name}: Error loading - {e}\n")
    
    print("=" * 60)
    print("EXAMPLE USAGE:")
    print("=" * 60)
    print("""
from strategies.strategy_loader import load_strategy

# Load default strategy
strategy = load_strategy('rsi_mean_reversion')

# Load with preset
strategy = load_strategy('rsi_mean_reversion', preset='aggressive')

# Use strategy
import pandas as pd
df = pd.DataFrame({...})  # Your OHLCV data
signals = strategy.generate_signals(df)
""")
