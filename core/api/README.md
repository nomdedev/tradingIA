# TradingIA REST API Documentation

## Overview

The TradingIA REST API provides remote access to all platform functionalities including data management, backtesting, strategy execution, and live trading capabilities.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, the API does not require authentication. In production environments, consider implementing API key authentication.

## Endpoints

### Health & Status

#### GET /health
Get basic system health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime": "0d 0h 5m",
  "components": {
    "data_manager": true,
    "strategy_engine": true,
    "backtester": true,
    "analysis_engines": true,
    "settings": true,
    "reporters": true,
    "broker_manager": true
  }
}
```

#### GET /status
Get detailed system status.

### Market Data

#### GET /data
Retrieve market data for analysis.

**Parameters:**
- `symbol` (required): Trading symbol (e.g., "AAPL", "BTC/USD")
- `timeframe` (required): Timeframe ("1Min", "5Min", "15Min", "1H", "1D")
- `start_date` (optional): Start date in YYYY-MM-DD format
- `end_date` (optional): End date in YYYY-MM-DD format

**Example:**
```
GET /data?symbol=BTC/USD&timeframe=1H&start_date=2024-01-01&end_date=2024-01-31
```

**Response:**
```json
{
  "symbol": "BTC/USD",
  "timeframe": "1H",
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "data_points": 744,
  "data": [...]
}
```

### Strategies

#### GET /strategies
Get list of available trading strategies.

**Response:**
```json
{
  "strategies": [
    "momentum_macd_adx",
    "mean_reversion_ibs_bb",
    "pairs_trading_cointegration"
  ]
}
```

### Backtesting

#### POST /backtest
Run a backtest asynchronously.

**Request Body:**
```json
{
  "strategy_name": "momentum_macd_adx",
  "symbol": "BTC/USD",
  "timeframe": "1H",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "initial_balance": 10000.0,
  "parameters": {
    "fast_period": 12,
    "slow_period": 26,
    "signal_period": 9
  }
}
```

**Response:**
```json
{
  "task_id": "backtest_momentum_macd_adx_20241114_143022",
  "status": "running",
  "message": "Backtest started successfully"
}
```

#### GET /backtest/{task_id}
Get status of a backtest task.

**Response:**
```json
{
  "task_id": "backtest_momentum_macd_adx_20241114_143022",
  "status": "completed",
  "message": "Backtest completed (results not persisted in this demo)"
}
```

### Broker Management

#### GET /brokers/status
Get broker system status.

**Response:**
```json
{
  "available": true,
  "status": {
    "brokers": {
      "alpaca_paper": {
        "connected": true,
        "account": {...},
        "positions": [...]
      }
    }
  }
}
```

#### POST /brokers
Add a new broker configuration.

**Request Body:**
```json
{
  "name": "alpaca_paper",
  "broker_type": "alpaca",
  "credentials": {
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "base_url": "https://paper-api.alpaca.markets"
  },
  "enabled": true
}
```

**Response:**
```json
{
  "message": "Broker 'alpaca_paper' added successfully"
}
```

### Order Management

#### POST /orders
Place an order through a configured broker.

**Request Body:**
```json
{
  "broker_name": "alpaca_paper",
  "symbol": "AAPL",
  "side": "buy",
  "order_type": "market",
  "quantity": 10,
  "price": null
}
```

**Response:**
```json
{
  "message": "Order placed successfully",
  "order": {
    "id": "order_123",
    "status": "filled",
    "filled_qty": 10,
    "avg_fill_price": 150.25
  }
}
```

### Task Management

#### GET /tasks
Get list of currently running tasks.

**Response:**
```json
{
  "running_tasks": [
    "backtest_momentum_macd_adx_20241114_143022"
  ]
}
```

## Running the API

### Standalone Server
```bash
cd /path/to/tradingia
python core/api/server.py
```

### From Python Code
```python
from core.api import app, API_AVAILABLE

if API_AVAILABLE:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Configuration

The API can be configured using environment variables:

- `TRADINGIA_API_HOST`: Server host (default: "0.0.0.0")
- `TRADINGIA_API_PORT`: Server port (default: 8000)
- `TRADINGIA_API_DEBUG`: Enable debug mode (default: false)
- `TRADINGIA_API_CORS`: Enable CORS (default: true)
- `TRADINGIA_API_DOCS`: Enable API documentation (default: true)

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found
- `500`: Internal Server Error

Error responses include a JSON body with error details:
```json
{
  "detail": "Error description"
}
```

## Rate Limiting

Basic rate limiting is implemented (configurable via environment variables):
- Default: 100 requests per 60 seconds
- Configure with `TRADINGIA_API_RATE_LIMIT` and `TRADINGIA_API_RATE_WINDOW`

## Security Considerations

For production deployment:

1. Implement proper authentication (API keys, JWT, OAuth)
2. Use HTTPS/WSS for encrypted communication
3. Configure CORS properly for your domains
4. Implement request validation and sanitization
5. Add rate limiting and DDoS protection
6. Use environment variables for sensitive configuration
7. Implement proper logging and monitoring