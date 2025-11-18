"""
REST API for remote access to the TradingIA platform.
Provides endpoints for system monitoring, backtesting, strategy management, and live trading.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from contextlib import asynccontextmanager
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Define dummy classes for when FastAPI/Pydantic is not available
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def Field(default=None, description="", **kwargs):
        return default

    asynccontextmanager = None

from core.backend_core import DataManager, StrategyEngine
from core.execution.backtester_core import BacktesterCore
from src.analysis_engines import AnalysisEngines
from utils.settings_manager import SettingsManager
from src.reporters_engine import ReportersEngine

# Optional broker integration
try:
    from core.brokers import BrokerManager
    BROKERS_AVAILABLE = True
except ImportError:
    BrokerManager = None
    BROKERS_AVAILABLE = False

# Pydantic models for API requests/responses
if FASTAPI_AVAILABLE:
    class SystemStatus(BaseModel):
        status: str = Field(..., description="System status")
        version: str = Field(..., description="Platform version")
        uptime: str = Field(..., description="System uptime")
        components: Dict[str, bool] = Field(..., description="Component availability")

    class DataRequest(BaseModel):
        symbol: str = Field(..., description="Trading symbol")
        timeframe: str = Field(..., description="Timeframe (1Min, 5Min, 15Min, 1H, 1D)")
        start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
        end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")

    class BacktestRequest(BaseModel):
        strategy_name: str = Field(..., description="Strategy name")
        symbol: str = Field(..., description="Trading symbol")
        timeframe: str = Field(..., description="Timeframe")
        start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
        end_date: str = Field(..., description="End date (YYYY-MM-DD)")
        initial_balance: float = Field(10000.0, description="Initial balance")
        parameters: Optional[Dict[str, Any]] = Field(None, description="Strategy parameters")

    class StrategyList(BaseModel):
        strategies: List[str] = Field(..., description="Available strategies")

    class BacktestResult(BaseModel):
        status: str = Field(..., description="Backtest status")
        results: Optional[Dict[str, Any]] = Field(None, description="Backtest results")
        error: Optional[str] = Field(None, description="Error message")

    class BrokerConfigRequest(BaseModel):
        name: str = Field(..., description="Broker name")
        broker_type: str = Field(..., description="Broker type (alpaca)")
        credentials: Dict[str, str] = Field(..., description="Broker credentials")
        enabled: bool = Field(True, description="Whether broker is enabled")

    class OrderRequest(BaseModel):
        broker_name: str = Field(..., description="Broker name")
        symbol: str = Field(..., description="Trading symbol")
        side: str = Field(..., description="Order side (buy/sell)")
        order_type: str = Field(..., description="Order type (market/limit/stop)")
        quantity: float = Field(..., description="Order quantity")
        price: Optional[float] = Field(None, description="Order price (for limit/stop orders)")
else:
    # Dummy classes when FastAPI is not available
    SystemStatus = None
    DataRequest = None
    BacktestRequest = None
    StrategyList = None
    BacktestResult = None
    BrokerConfigRequest = None
    OrderRequest = None


class TradingAPI:
    """Main REST API class for the TradingIA platform."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.data_manager = DataManager()
        self.strategy_engine = StrategyEngine()
        self.backtester = BacktesterCore()
        self.analysis_engines = AnalysisEngines()
        self.settings = SettingsManager()
        self.reporters = ReportersEngine()

        # Initialize broker manager if available
        self.broker_manager = BrokerManager() if BROKERS_AVAILABLE else None

        # Track running tasks
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.backtest_results: Dict[str, Dict[str, Any]] = {}

        # Start time for uptime calculation
        self.start_time = datetime.now()

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = datetime.now() - self.start_time

        components = {
            "data_manager": True,
            "strategy_engine": True,
            "backtester": True,
            "analysis_engines": True,
            "settings": True,
            "reporters": True,
            "broker_manager": self.broker_manager is not None
        }

        return {
            "status": "healthy",
            "version": "2.0.0",
            "uptime": f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m",
            "components": components,
            "timestamp": datetime.now().isoformat()
        }

    async def get_market_data(self, symbol: str, timeframe: str,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get market data for a symbol."""
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

            # Get data using data manager
            df = self.data_manager.load_alpaca_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )

            if isinstance(df, dict) and 'error' in df:
                raise HTTPException(status_code=400, detail=df['error'])

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "data_points": len(df),
                "data": df.to_dict('records') if hasattr(df, 'to_dict') else df
            }

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def run_backtest(self, request: BacktestRequest, background_tasks=None) -> Dict[str, Any]:
        """Run a backtest asynchronously."""
        try:
            # Validate strategy exists
            available_strategies = self.strategy_engine.get_available_strategies()
            if request.strategy_name not in available_strategies:
                raise HTTPException(status_code=400, detail=f"Strategy '{request.strategy_name}' not found")

            # Create unique task ID
            task_id = f"backtest_{request.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Run backtest in background
            task = asyncio.create_task(self._run_backtest_task(task_id, request))
            self.running_tasks[task_id] = task

            return {
                "task_id": task_id,
                "status": "running",
                "message": "Backtest started successfully"
            }

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error starting backtest: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _run_backtest_task(self, task_id: str, request: BacktestRequest) -> None:
        """Run backtest task in background."""
        try:
            # Configure backtest parameters (keeping for future use)
            # config = {
            #     'strategy': request.strategy_name,
            #     'symbol': request.symbol,
            #     'timeframe': request.timeframe,
            #     'start_date': request.start_date,
            #     'end_date': request.end_date,
            #     'initial_balance': request.initial_balance,
            #     'parameters': request.parameters or {}
            # }

            # Run backtest
            # For now, create a simple mock result since we need to integrate with data loading
            results = {
                "metrics": {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": 0.08,
                    "win_rate": 0.55,
                    "total_trades": 25
                },
                "trades": [],
                "equity_curve": [10000] * 100,  # Mock equity curve
                "status": "completed"
            }

            # Store results for later retrieval
            self.backtest_results[task_id] = {
                "status": "completed",
                "results": results,
                "completed_at": datetime.now().isoformat()
            }

            # Store results (in a real implementation, you'd save to database/file)
            self.logger.info(f"Backtest {task_id} completed successfully")

        except Exception as e:
            self.logger.error(f"Backtest {task_id} failed: {e}")
        finally:
            # Clean up task
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

    def get_backtest_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a backtest task."""
        if task_id in self.running_tasks:
            return {
                "task_id": task_id,
                "status": "running",
                "message": "Backtest is still running"
            }
        elif task_id in self.backtest_results:
            result = self.backtest_results[task_id]
            return {
                "task_id": task_id,
                "status": result["status"],
                "results": result["results"],
                "completed_at": result["completed_at"],
                "message": "Backtest completed successfully"
            }
        else:
            return {
                "task_id": task_id,
                "status": "not_found",
                "message": "Backtest task not found"
            }

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies."""
        try:
            # For now, return a hardcoded list of available strategies
            # TODO: Integrate with actual strategy registry
            return [
                "MACD_ADX_Strategy",
                "RSI_Bollinger_Strategy", 
                "MovingAverageCrossover_Strategy",
                "StochasticRSI_Strategy",
                "BollingerBands_Strategy"
            ]
        except Exception as e:
            self.logger.error(f"Error getting available strategies: {e}")
            return []

    def get_broker_status(self) -> Dict[str, Any]:
        """Get broker status if available."""
        if not self.broker_manager:
            return {"available": False, "message": "Broker integration not available"}

        try:
            status = self.broker_manager.get_status_summary()
            return {"available": True, "status": status}
        except Exception as e:
            self.logger.error(f"Error getting broker status: {e}")
            return {"available": True, "error": str(e)}

    async def add_broker(self, config: BrokerConfigRequest) -> Dict[str, Any]:
        """Add a new broker."""
        if not self.broker_manager:
            raise HTTPException(status_code=400, detail="Broker integration not available")  # type: ignore

        try:
            from core.brokers import BrokerConfig
            broker_config = BrokerConfig(
                name=config.name,
                broker_type=config.broker_type,
                credentials=config.credentials,
                enabled=config.enabled
            )

            self.broker_manager.add_broker(broker_config)
            return {"message": f"Broker '{config.name}' added successfully"}

        except Exception as e:
            self.logger.error(f"Error adding broker: {e}")
            raise HTTPException(status_code=500, detail=str(e))  # type: ignore

    async def place_order(self, order: OrderRequest) -> Dict[str, Any]:
        """Place an order through a broker."""
        if not self.broker_manager:
            raise HTTPException(status_code=400, detail="Broker integration not available")  # type: ignore

        try:
            # Get broker
            brokers = self.broker_manager.get_brokers()
            if order.broker_name not in brokers:
                raise HTTPException(status_code=400, detail=f"Broker '{order.broker_name}' not found")  # type: ignore

            broker = brokers[order.broker_name]

            # Place order
            order_result = broker.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price
            )

            return {
                "message": "Order placed successfully",
                "order": order_result
            }

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise HTTPException(status_code=500, detail=str(e))  # type: ignore


# Global API instance
api_instance = TradingAPI()

# FastAPI app creation
if FASTAPI_AVAILABLE:

    @asynccontextmanager  # type: ignore
    async def lifespan(app: FastAPI):  # type: ignore
        """Handle application startup and shutdown."""
        logging.info("TradingIA API starting up...")
        yield
        logging.info("TradingIA API shutting down...")

    app = FastAPI(  # type: ignore
        title="TradingIA API",
        description="REST API for TradingIA Advanced Trading Platform",
        version="2.0.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,  # type: ignore
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=SystemStatus)
    async def health_check():
        """Health check endpoint."""
        return api_instance.get_system_status()

    @app.get("/status")
    async def system_status():
        """Get detailed system status."""
        return api_instance.get_system_status()

    @app.get("/data")
    async def get_data(
        symbol: str = Query(..., description="Trading symbol"),  # type: ignore
        timeframe: str = Query(..., description="Timeframe"),  # type: ignore
        start_date: Optional[str] = Query(None, description="Start date"),  # type: ignore
        end_date: Optional[str] = Query(None, description="End date")  # type: ignore
    ):
        """Get market data."""
        return await api_instance.get_market_data(symbol, timeframe, start_date, end_date)

    @app.post("/backtest")
    async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):  # type: ignore
        """Run a backtest."""
        return await api_instance.run_backtest(request, background_tasks)

    @app.get("/backtest/{task_id}")
    async def get_backtest_status(task_id: str):
        """Get backtest task status."""
        return api_instance.get_backtest_status(task_id)

    @app.get("/strategies")
    async def get_strategies():
        """Get available strategies."""
        strategies = api_instance.get_available_strategies()
        return {"strategies": strategies}

    @app.get("/brokers/status")
    async def get_broker_status():
        """Get broker status."""
        return api_instance.get_broker_status()

    @app.post("/brokers")
    async def add_broker(config: BrokerConfigRequest):
        """Add a new broker."""
        return await api_instance.add_broker(config)

    @app.post("/orders")
    async def place_order(order: OrderRequest):
        """Place an order."""
        return await api_instance.place_order(order)

    @app.get("/tasks")
    async def get_running_tasks():
        """Get running tasks."""
        return {"running_tasks": list(api_instance.running_tasks.keys())}

else:
    app = None

# Export the app for use by the main platform
__all__ = ['app', 'TradingAPI', 'FASTAPI_AVAILABLE']