"""
Alpaca API utilities for data normalization and processing.
Handles trade data normalization and API response processing.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timezone

def normalize_alpaca_trades(raw_trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize Alpaca trade data to consistent format.

    Args:
        raw_trades: Raw trade data from Alpaca API

    Returns:
        Normalized DataFrame with consistent column names
    """
    try:
        if not raw_trades:
            return pd.DataFrame(columns=[
                'timestamp', 'symbol', 'side', 'qty', 'price',
                'order_type', 'status', 'filled_qty', 'filled_avg_price'
            ])

        normalized_data = []

        for trade in raw_trades:
            try:
                # Handle different possible field names
                timestamp = trade.get('submitted_at') or trade.get('created_at') or trade.get('timestamp')
                if timestamp:
                    # Ensure timestamp is timezone-aware
                    if isinstance(timestamp, str):
                        timestamp = pd.to_datetime(timestamp, utc=True)
                    elif isinstance(timestamp, datetime):
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                        else:
                            timestamp = timestamp.astimezone(timezone.utc)

                normalized_trade = {
                    'timestamp': timestamp,
                    'symbol': trade.get('symbol', ''),
                    'side': trade.get('side', '').lower(),
                    'qty': float(trade.get('qty', 0)),
                    'price': float(trade.get('limit_price') or trade.get('stop_price') or trade.get('price', 0)),
                    'order_type': trade.get('type', '').lower(),
                    'status': trade.get('status', '').lower(),
                    'filled_qty': float(trade.get('filled_qty', 0)),
                    'filled_avg_price': float(trade.get('filled_avg_price', 0))
                }

                normalized_data.append(normalized_trade)

            except Exception as e:
                # Log error but continue processing
                print(f"Error normalizing trade: {e}")
                continue

        df = pd.DataFrame(normalized_data)

        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns and not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

        return df

    except Exception as e:
        print(f"Error in normalize_alpaca_trades: {e}")
        return pd.DataFrame()

def normalize_alpaca_positions(raw_positions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize Alpaca position data.

    Args:
        raw_positions: Raw position data from Alpaca API

    Returns:
        Normalized DataFrame
    """
    try:
        if not raw_positions:
            return pd.DataFrame(columns=[
                'symbol', 'qty', 'avg_entry_price', 'current_price',
                'market_value', 'unrealized_pl', 'unrealized_plpc'
            ])

        normalized_data = []

        for pos in raw_positions:
            try:
                normalized_pos = {
                    'symbol': pos.get('symbol', ''),
                    'qty': float(pos.get('qty', 0)),
                    'avg_entry_price': float(pos.get('avg_entry_price', 0)),
                    'current_price': float(pos.get('current_price', 0)),
                    'market_value': float(pos.get('market_value', 0)),
                    'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                    'unrealized_plpc': float(pos.get('unrealized_plpc', 0))
                }

                normalized_data.append(normalized_pos)

            except Exception as e:
                print(f"Error normalizing position: {e}")
                continue

        return pd.DataFrame(normalized_data)

    except Exception as e:
        print(f"Error in normalize_alpaca_positions: {e}")
        return pd.DataFrame()

def normalize_alpaca_account(raw_account: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Alpaca account data.

    Args:
        raw_account: Raw account data from Alpaca API

    Returns:
        Normalized account dictionary
    """
    try:
        return {
            'equity': float(raw_account.get('equity', 0)),
            'last_equity': float(raw_account.get('last_equity', 0)),
            'cash': float(raw_account.get('cash', 0)),
            'buying_power': float(raw_account.get('buying_power', 0)),
            'daytrade_count': int(raw_account.get('daytrade_count', 0)),
            'status': raw_account.get('status', ''),
            'currency': raw_account.get('currency', 'USD')
        }

    except Exception as e:
        print(f"Error in normalize_alpaca_account: {e}")
        return {}

def normalize_alpaca_bars(raw_bars: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize Alpaca bar/OHLC data.

    Args:
        raw_bars: Raw bar data from Alpaca API

    Returns:
        Normalized DataFrame with OHLC data
    """
    try:
        if not raw_bars:
            return pd.DataFrame(columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])

        normalized_data = []

        for bar in raw_bars:
            try:
                # Handle timestamp
                timestamp = bar.get('t') or bar.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp, utc=True)
                elif isinstance(timestamp, datetime):
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    else:
                        timestamp = timestamp.astimezone(timezone.utc)

                normalized_bar = {
                    'timestamp': timestamp,
                    'open': float(bar.get('o', 0)),
                    'high': float(bar.get('h', 0)),
                    'low': float(bar.get('l', 0)),
                    'close': float(bar.get('c', 0)),
                    'volume': int(bar.get('v', 0))
                }

                normalized_data.append(normalized_bar)

            except Exception as e:
                print(f"Error normalizing bar: {e}")
                continue

        df = pd.DataFrame(normalized_data)

        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns and not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

        return df

    except Exception as e:
        print(f"Error in normalize_alpaca_bars: {e}")
        return pd.DataFrame()

def calculate_trade_pnl(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate P&L for trades.

    Args:
        trades_df: DataFrame with trade data

    Returns:
        DataFrame with P&L calculations
    """
    try:
        if trades_df.empty:
            return trades_df

        df = trades_df.copy()

        # Calculate P&L for filled trades
        if 'filled_qty' in df.columns and 'filled_avg_price' in df.columns:
            # This is a simplified calculation - in practice you'd need entry/exit prices
            df['pnl'] = 0.0  # Placeholder
            df['pnl_pct'] = 0.0  # Placeholder

        return df

    except Exception as e:
        print(f"Error calculating trade P&L: {e}")
        return trades_df

def validate_alpaca_response(response: Any) -> bool:
    """
    Validate Alpaca API response.

    Args:
        response: API response to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        if response is None:
            return False

        # Check for error indicators
        if isinstance(response, dict):
            if 'error' in response or 'message' in response:
                return False

        return True

    except Exception:
        return False