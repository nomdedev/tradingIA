"""
API configuration and utilities for the TradingIA REST API.
"""

import os
from typing import Dict, Any

class APIConfig:
    """Configuration for the TradingIA REST API."""

    def __init__(self):
        # API server configuration
        self.host: str = os.getenv("TRADINGIA_API_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("TRADINGIA_API_PORT", "8000"))
        self.debug: bool = os.getenv("TRADINGIA_API_DEBUG", "false").lower() == "true"

        # Security configuration
        self.enable_cors: bool = os.getenv("TRADINGIA_API_CORS", "true").lower() == "true"
        self.allowed_origins: list = os.getenv("TRADINGIA_API_ALLOWED_ORIGINS", "*").split(",")

        # API features
        self.enable_docs: bool = os.getenv("TRADINGIA_API_DOCS", "true").lower() == "true"
        self.enable_backtest_endpoint: bool = os.getenv("TRADINGIA_API_BACKTEST", "true").lower() == "true"
        self.enable_broker_endpoints: bool = os.getenv("TRADINGIA_API_BROKERS", "true").lower() == "true"

        # Rate limiting (basic implementation)
        self.rate_limit_requests: int = int(os.getenv("TRADINGIA_API_RATE_LIMIT", "100"))
        self.rate_limit_window: int = int(os.getenv("TRADINGIA_API_RATE_WINDOW", "60"))  # seconds

        # Data limits
        self.max_data_points: int = int(os.getenv("TRADINGIA_API_MAX_DATA_POINTS", "10000"))
        self.max_backtest_duration: int = int(os.getenv("TRADINGIA_API_MAX_BACKTEST_DURATION", "300"))  # seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "enable_cors": self.enable_cors,
            "allowed_origins": self.allowed_origins,
            "enable_docs": self.enable_docs,
            "enable_backtest_endpoint": self.enable_backtest_endpoint,
            "enable_broker_endpoints": self.enable_broker_endpoints,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window": self.rate_limit_window,
            "max_data_points": self.max_data_points,
            "max_backtest_duration": self.max_backtest_duration
        }

# Global configuration instance
api_config = APIConfig()