#!/usr/bin/env python3
"""
Standalone server for the TradingIA REST API.
Run this script to start the API server independently.
"""

import logging
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from core.api import app, api_config, API_AVAILABLE
    import uvicorn

    if not API_AVAILABLE:
        print("❌ FastAPI is not available. Please install it with: pip install fastapi uvicorn")
        sys.exit(1)

    def main():
        """Main entry point for the API server."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        logger = logging.getLogger(__name__)

        print("🚀 Starting TradingIA REST API Server")
        print(f"📍 Host: {api_config.host}")
        print(f"🔌 Port: {api_config.port}")
        print(f"📚 Docs: http://{api_config.host if api_config.host != '0.0.0.0' else 'localhost'}:{api_config.port}/docs")
        print(f"🔄 ReDocs: http://{api_config.host if api_config.host != '0.0.0.0' else 'localhost'}:{api_config.port}/redoc")

        try:
            uvicorn.run(
                "core.api.main:app",
                host=api_config.host,
                port=api_config.port,
                reload=api_config.debug,
                log_level="info"
            )
        except KeyboardInterrupt:
            logger.info("API server stopped by user")
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            sys.exit(1)

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)