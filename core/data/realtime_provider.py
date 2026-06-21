from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List
import threading
import time
import random
import pandas as pd
from datetime import datetime

class RealTimeDataProvider(ABC):
    """Abstract base class for real-time data providers."""
    
    def __init__(self):
        self.subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self.is_running = False
        
    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to data updates."""
        self.subscribers.append(callback)
        
    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Unsubscribe from data updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            
    def notify(self, data: Dict[str, Any]):
        """Notify all subscribers with new data."""
        import logging
        logger = logging.getLogger(__name__)
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}")
                
    @abstractmethod
    def start(self, symbols: List[str]):
        """Start streaming data for symbols."""
        pass
        
    @abstractmethod
    def stop(self):
        """Stop streaming data."""
        pass

class MockRealTimeProvider(RealTimeDataProvider):
    """Simulates real-time data for testing/paper trading."""
    
    def __init__(self, interval_sec: float = 1.0):
        super().__init__()
        self.interval = interval_sec
        self.thread = None
        self.symbols = []
        self.prices = {}
        
    def start(self, symbols: List[str]):
        self.symbols = symbols
        self.is_running = True
        
        # Initialize random prices
        for sym in symbols:
            if "BTC" in sym: self.prices[sym] = 45000.0
            elif "ETH" in sym: self.prices[sym] = 2500.0
            else: self.prices[sym] = 100.0
            
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _run_loop(self):
        while self.is_running:
            for sym in self.symbols:
                # Random walk
                change = random.uniform(-0.002, 0.002)
                self.prices[sym] *= (1 + change)
                
                tick = {
                    "type": "ticker",
                    "symbol": sym,
                    "price": self.prices[sym],
                    "volume": random.randint(1, 100),
                    "timestamp": datetime.now().isoformat()
                }
                self.notify(tick)
                
            time.sleep(self.interval)
