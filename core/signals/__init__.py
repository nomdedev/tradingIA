"""
Core Signals Module.

ÁREA 8: TradingSignal Standard dataclass.
Todas las estrategias deben usar esta estructura estandarizada.
"""

from .trading_signal import TradingSignal, SignalDirection, SignalStrength

__all__ = ['TradingSignal', 'SignalDirection', 'SignalStrength']
