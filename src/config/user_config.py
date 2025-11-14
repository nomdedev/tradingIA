"""
User Configuration Manager
Saves and loads user preferences between sessions
"""

import json
import os
from datetime import datetime


class UserConfigManager:
    """Manages user configuration persistence"""
    
    def __init__(self, config_file='config/user_preferences.json'):
        self.config_file = config_file
        self.config_dir = os.path.dirname(config_file)
        self.default_config = {
            'last_session': None,
            'live_trading': {
                'ticker': 'BTC/USD',
                'strategy': 'RSI Mean Reversion',
                'mode': 'Paper Trading',
                'parameters': {
                    'rsi_period': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30,
                    'take_profit': 2.0,
                    'stop_loss': 1.5
                }
            },
            'backtest': {
                'start_date': '2023-01-01',
                'end_date': '2024-01-01',
                'initial_capital': 10000,
                'timeframe': '5min'
            },
            'data_paths': {
                '5min': 'data/raw/BTCUSD_5Min.csv',
                '15min': 'data/raw/BTCUSD_15Min.csv',
                '1hour': 'data/raw/BTCUSD_1Hour.csv',
                '4hour': 'data/raw/BTCUSD_4Hour.csv'
            }
        }
        
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Load or create config
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return self._merge_dicts(self.default_config, loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}")
                return self.default_config.copy()
        else:
            return self.default_config.copy()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            self.config['last_session'] = datetime.now().isoformat()
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, key_path, default=None):
        """
        Get config value using dot notation
        Example: get('live_trading.ticker')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path, value, auto_save=True):
        """
        Set config value using dot notation
        Example: set('live_trading.ticker', 'ETH/USD')
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        
        if auto_save:
            self.save_config()
    
    def get_live_trading_config(self):
        """Get all live trading configuration"""
        return self.config.get('live_trading', {})
    
    def update_live_trading_config(self, ticker=None, strategy=None, mode=None, parameters=None):
        """Update live trading configuration"""
        if ticker:
            self.config['live_trading']['ticker'] = ticker
        if strategy:
            self.config['live_trading']['strategy'] = strategy
        if mode:
            self.config['live_trading']['mode'] = mode
        if parameters:
            self.config['live_trading']['parameters'].update(parameters)
        
        self.save_config()
    
    def get_backtest_config(self):
        """Get backtest configuration"""
        return self.config.get('backtest', {})
    
    def update_backtest_config(self, **kwargs):
        """Update backtest configuration"""
        self.config['backtest'].update(kwargs)
        self.save_config()
    
    def get_data_path(self, timeframe):
        """Get data file path for a specific timeframe"""
        return self.config['data_paths'].get(timeframe)
    
    def _merge_dicts(self, default, loaded):
        """Recursively merge loaded config with defaults"""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.default_config.copy()
        self.save_config()
