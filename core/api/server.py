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
        logging.error("❌ FastAPI is not available. Please install it with: pip install fastapi uvicorn")
        sys.exit(1)

    def main():
        """Main entry point for the API server."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        logger = logging.getLogger(__name__)

        logger.info("🚀 Starting TradingIA REST API Server")
        logger.info(f"📍 Host: {api_config.host}")
        logger.info(f"🔌 Port: {api_config.port}")
        docs_host = api_config.host if api_config.host != '0.0.0.0' else 'localhost'
        logger.info(f"📚 Docs: http://{docs_host}:{api_config.port}/docs")
        logger.info(f"🔄 ReDocs: http://{docs_host}:{api_config.port}/redoc")

        try:
            uvicorn.run(
                "core.api.main:app",
                host=api_config.host,
                port=api_config.port,
                reload=api_config.debug,
                log_level="info",
            )
        except KeyboardInterrupt:
            logger.info("API server stopped by user")
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            sys.exit(1)

    if __name__ == "__main__":
        main()

except ImportError as e:
    logging.error(f"❌ Import error: {e}")
    logging.error("Please ensure all dependencies are installed.")
    sys.exit(1)
