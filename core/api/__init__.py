"""
TradingIA REST API module initialization.
"""

try:
    from .main import app, TradingAPI, FASTAPI_AVAILABLE
    from .config import api_config, APIConfig
    API_AVAILABLE = FASTAPI_AVAILABLE
except ImportError:
    app = None
    TradingAPI = None
    api_config = None
    APIConfig = None
    API_AVAILABLE = False

__all__ = [
    'app',
    'TradingAPI',
    'api_config',
    'APIConfig',
    'API_AVAILABLE'
]

if API_AVAILABLE:
    __all__.extend(['FASTAPI_AVAILABLE'])