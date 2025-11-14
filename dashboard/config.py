"""
Configuration settings for the trading dashboard.
"""

import os
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use environment variables directly

# Dashboard Configuration
DASHBOARD_CONFIG = {
    # Page settings
    'page_title': 'Trading AI Dashboard',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',

    # Auto-refresh settings
    'auto_refresh_interval': 30,  # seconds
    'enable_auto_refresh': True,

    # Data settings
    'logs_directory': 'logs',
    'trades_file': 'trades.csv',
    'retraining_file': 'retraining_history.csv',
    'max_data_points': 10000,  # Maximum data points to load

    # Display settings
    'default_chart_height': 400,
    'default_table_height': 300,
    'max_feed_items': 50,

    # Color scheme
    'colors': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff9896',
        'info': '#aec7e8',
        'background': '#f8f9fa',
        'text': '#212529'
    },

    # Alpaca API Configuration
    'alpaca': {
        'api_key': os.getenv('ALPACA_API_KEY', ''),  # Set via environment variable
        'api_secret': os.getenv('ALPACA_SECRET_KEY', ''),  # Set via environment variable
        'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),  # Paper trading URL
        'use_paper': True,  # Always use paper trading for safety
        'max_orders_history': 100,  # Maximum orders to fetch
        'max_positions_history': 100,  # Maximum positions to fetch
        'data_timeframe': '1D',  # Default timeframe for historical data
        'data_limit': 1000  # Maximum data points for charts
    },

    # System monitoring thresholds
    'system_thresholds': {
        'cpu_warning': 80,     # 80%
        'cpu_danger': 95,      # 95%
        'memory_warning': 85,  # 85%
        'memory_danger': 95,   # 95%
        'disk_warning': 90,    # 90%
        'disk_danger': 95      # 95%
    },

    # Performance targets
    'performance_targets': {
        'target_win_rate': 0.55,       # 55%
        'target_sharpe_ratio': 1.5,
        'target_profit_factor': 1.5,
        'target_max_drawdown': 0.15    # 15%
    }
}

# Navigation menu configuration
NAVIGATION_CONFIG = {
    'pages': [
        {
            'title': 'Live Trading',
            'icon': '📈',
            'file': 'pages/1_live_trading.py',
            'description': 'Real-time trading monitor'
        },
        {
            'title': 'Performance',
            'icon': '📊',
            'file': 'pages/2_performance.py',
            'description': 'Performance analytics'
        },
        {
            'title': 'Retraining History',
            'icon': '🔄',
            'file': 'pages/3_retraining_history.py',
            'description': 'Model retraining timeline'
        },
        {
            'title': 'Risk Analysis',
            'icon': '⚠️',
            'file': 'pages/4_risk_analysis.py',
            'description': 'Risk metrics and analysis'
        },
        {
            'title': 'System Status',
            'icon': '🖥️',
            'file': 'pages/5_system_status.py',
            'description': 'System health monitoring'
        }
    ]
}

# Chart configuration
CHART_CONFIG = {
    'equity_curve': {
        'title': 'Portfolio Equity Curve',
        'x_title': 'Date',
        'y_title': 'Portfolio Value ($)',
        'height': 400
    },

    'r_multiples_histogram': {
        'title': 'R-Multiple Distribution',
        'x_title': 'R-Multiple',
        'y_title': 'Frequency',
        'height': 300,
        'bins': 20
    },

    'performance_metrics': {
        'title': 'Performance Metrics Over Time',
        'height': 400,
        'metrics': ['sharpe', 'win_rate', 'drawdown']
    },

    'correlation_matrix': {
        'title': 'Asset Correlation Matrix',
        'height': 400,
        'colorscale': 'RdBu'
    },

    'drawdown_chart': {
        'title': 'Drawdown Analysis',
        'x_title': 'Date',
        'y_title': 'Drawdown (%)',
        'height': 300
    }
}

# Table configuration
TABLE_CONFIG = {
    'trades_table': {
        'max_rows': 100,
        'height': 400,
        'columns': [
            'timestamp', 'symbol', 'action', 'price',
            'quantity', 'profit', 'r_multiple', 'reason'
        ]
    },

    'positions_table': {
        'max_rows': 50,
        'height': 300,
        'columns': [
            'symbol', 'quantity', 'entry_price', 'current_price',
            'unrealized_pnl', 'stop_loss', 'take_profit'
        ]
    },

    'retraining_table': {
        'max_rows': 50,
        'height': 400,
        'columns': [
            'timestamp', 'model_version', 'accuracy_before',
            'accuracy_after', 'improvement', 'reason'
        ]
    }
}

# Feed configuration
FEED_CONFIG = {
    'signal_feed': {
        'max_items': 20,
        'update_interval': 5,  # seconds
        'retention_hours': 24
    },

    'decision_feed': {
        'max_items': 20,
        'update_interval': 5,
        'retention_hours': 24
    },

    'alert_feed': {
        'max_items': 10,
        'update_interval': 10,
        'retention_hours': 48
    }
}

def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.

    Args:
        key_path: Dot-separated path to config value (e.g., 'colors.primary')
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    config = DASHBOARD_CONFIG

    try:
        for key in keys:
            config = config[key]
        return config
    except (KeyError, TypeError):
        return default

def get_logs_path(filename: str) -> str:
    """
    Get full path for a logs file.

    Args:
        filename: Name of the log file

    Returns:
        Full path to the log file
    """
    logs_dir = get_config_value('logs_directory', 'logs')
    return os.path.join(logs_dir, filename)

def get_color_scheme() -> Dict[str, str]:
    """
    Get the color scheme configuration.

    Returns:
        Dictionary with color definitions
    """
    return get_config_value('colors', {})

def get_risk_thresholds() -> Dict[str, float]:
    """
    Get risk threshold configuration.

    Returns:
        Dictionary with risk thresholds
    """
    return get_config_value('risk_thresholds', {})

def get_system_thresholds() -> Dict[str, float]:
    """
    Get system monitoring threshold configuration.

    Returns:
        Dictionary with system thresholds
    """
    return get_config_value('system_thresholds', {})

def get_alpaca_config() -> Dict[str, Any]:
    """
    Get Alpaca API configuration.

    Returns:
        Dictionary with Alpaca configuration
    """
    return get_config_value('alpaca', {})

def is_alpaca_configured() -> bool:
    """
    Check if Alpaca API is properly configured.

    Returns:
        True if Alpaca credentials are set, False otherwise
    """
    config = get_alpaca_config()
    return bool(config.get('api_key') and config.get('api_secret'))