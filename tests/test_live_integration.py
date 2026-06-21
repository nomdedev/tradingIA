"""
Tests de integración para Live Trading y Monitoring

Prueba flujos completos:
- LiveTrader conexión y desconexión
- Order submission con retry
- Production monitoring lifecycle
- Kill switch activation
"""

import pytest
import threading
import time
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime


class TestLiveTraderIntegration:
    """Tests de integración para LiveTrader"""
    
    @pytest.fixture
    def mock_alpaca_api(self):
        """Mock de Alpaca API"""
        mock_api = MagicMock()
        mock_api.get_account.return_value = MagicMock(
            equity=10000.0,
            buying_power=10000.0,
            cash=10000.0,
            status='ACTIVE'
        )
        mock_api.list_positions.return_value = []
        mock_api.list_orders.return_value = []
        return mock_api
    
    def test_live_trader_initialization(self, mock_alpaca_api):
        """Test que LiveTrader se inicializa correctamente"""
        with patch.dict('os.environ', {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            from core.execution.live_trader import LiveTrader, ALPACA_AVAILABLE
            
            if not ALPACA_AVAILABLE:
                pytest.skip("Alpaca SDK not installed")
            
            with patch('alpaca_trade_api.REST', return_value=mock_alpaca_api):
                trader = LiveTrader(
                    api_key='test_key',
                    api_secret='test_secret',
                    paper=True
                )
                
                assert trader is not None
                assert trader._paper_trading == True
                assert not trader._kill_switch_active
    
    def test_live_trader_context_manager(self, mock_alpaca_api):
        """Test que LiveTrader funciona como context manager"""
        with patch.dict('os.environ', {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret'
        }):
            from core.execution.live_trader import LiveTrader, ALPACA_AVAILABLE
            
            if not ALPACA_AVAILABLE:
                pytest.skip("Alpaca SDK not installed")
            
            with patch('alpaca_trade_api.REST', return_value=mock_alpaca_api):
                with LiveTrader(
                    api_key='test_key',
                    api_secret='test_secret',
                    paper=True
                ) as trader:
                    assert trader._running == True
                
                # After exit, should be stopped
                assert trader._running == False
    
    def test_kill_switch_activation(self, mock_alpaca_api):
        """Test que kill switch se activa correctamente"""
        from core.execution.live_trader import LiveTrader, ALPACA_AVAILABLE
        
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca SDK not installed")
        
        with patch('alpaca_trade_api.REST', return_value=mock_alpaca_api):
            trader = LiveTrader(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Activate kill switch
            trader.activate_kill_switch("Test reason")
            
            assert trader._kill_switch_active == True
            assert trader._kill_switch_reason == "Test reason"
            
            # Deactivate
            trader.deactivate_kill_switch()
            assert trader._kill_switch_active == False
    
    def test_rate_limiter(self, mock_alpaca_api):
        """Test que rate limiter funciona"""
        from core.execution.live_trader import LiveTrader, ALPACA_AVAILABLE
        
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca SDK not installed")
        
        with patch('alpaca_trade_api.REST', return_value=mock_alpaca_api):
            trader = LiveTrader(
                api_key='test_key',
                api_secret='test_secret',
                paper=True,
                rate_limit=10  # Low limit for testing
            )
            
            # Make several calls
            start_time = time.time()
            for _ in range(5):
                trader._rate_limit_check()
            
            # Should complete quickly with few calls
            elapsed = time.time() - start_time
            assert elapsed < 1.0, "Rate limiter should not block for few calls"


class TestProductionMonitoringIntegration:
    """Tests de integración para Production Monitoring"""
    
    def test_monitor_start_stop(self):
        """Test ciclo de vida del monitor"""
        from src.production_monitoring import ProductionMonitor
        
        monitor = ProductionMonitor(log_level="WARNING")
        
        # Start monitoring
        monitor.start_monitoring(interval_seconds=1)
        assert monitor.monitoring_active == True
        assert monitor.monitor_thread is not None
        assert monitor.monitor_thread.is_alive()
        
        # Let it run briefly
        time.sleep(0.5)
        
        # Stop monitoring
        monitor.stop_monitoring(timeout=2.0)
        assert monitor.monitoring_active == False
        
        # Thread should terminate
        time.sleep(0.5)
        assert not monitor.monitor_thread.is_alive()
    
    def test_monitor_thread_cleanup(self):
        """Test que threads se limpian correctamente"""
        from src.production_monitoring import ProductionMonitor
        
        monitor = ProductionMonitor(log_level="WARNING")
        
        # Start and stop multiple times
        for _ in range(3):
            monitor.start_monitoring(interval_seconds=1)
            time.sleep(0.2)
            monitor.stop_monitoring(timeout=2.0)
            time.sleep(0.2)
        
        # All threads should be cleaned up
        assert not monitor.monitoring_active
        if monitor.monitor_thread:
            assert not monitor.monitor_thread.is_alive()
    
    def test_monitor_health_check(self):
        """Test health check básico"""
        from src.production_monitoring import ProductionMonitor
        
        monitor = ProductionMonitor(log_level="WARNING")
        
        # Perform single health check
        monitor._perform_health_check()
        
        # Should have updated system health
        assert 'last_check' in monitor.system_health
        assert monitor.system_health['last_check'] is not None


class TestOrderRetryLogic:
    """Tests para lógica de retry en órdenes"""
    
    def test_retry_on_transient_error(self):
        """Test que retry funciona en errores transitorios"""
        from core.execution.live_trader import LiveTrader, ALPACA_AVAILABLE
        
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca SDK not installed")
        
        call_count = 0
        
        def mock_submit(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            return MagicMock(id='order_123', status='accepted')
        
        mock_api = MagicMock()
        mock_api.submit_order = mock_submit
        mock_api.get_account.return_value = MagicMock(
            equity=10000.0, buying_power=10000.0, status='ACTIVE'
        )
        mock_api.list_positions.return_value = []
        
        with patch('alpaca_trade_api.REST', return_value=mock_api):
            trader = LiveTrader(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            trader.api = mock_api
            
            # Should succeed after retries
            order = trader.submit_order_with_retry(
                symbol='BTCUSD',
                qty=0.01,
                side='buy'
            )
            
            assert call_count == 3, "Should have retried twice"
            assert order is not None


class TestCallbackSystem:
    """Tests para sistema de callbacks"""
    
    def test_fill_callback(self):
        """Test que callbacks de fill se ejecutan"""
        from core.execution.live_trader import LiveTrader, ALPACA_AVAILABLE
        
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca SDK not installed")
        
        callback_executed = False
        received_order = None
        
        def on_fill(order):
            nonlocal callback_executed, received_order
            callback_executed = True
            received_order = order
        
        mock_api = MagicMock()
        mock_api.get_account.return_value = MagicMock(
            equity=10000.0, status='ACTIVE'
        )
        mock_api.list_positions.return_value = []
        
        with patch('alpaca_trade_api.REST', return_value=mock_api):
            trader = LiveTrader(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            trader.on_fill(on_fill)
            
            # Simulate fill notification
            mock_order = MagicMock(id='order_123', symbol='BTCUSD')
            for callback in trader._on_fill_callbacks:
                callback(mock_order)
            
            assert callback_executed, "Fill callback should have executed"


class TestThreadSafety:
    """Tests para thread safety"""
    
    def test_concurrent_order_submission(self):
        """Test que múltiples threads pueden enviar órdenes"""
        from core.execution.live_trader import LiveTrader, ALPACA_AVAILABLE
        
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca SDK not installed")
        
        order_count = 0
        lock = threading.Lock()
        
        def mock_submit(*args, **kwargs):
            nonlocal order_count
            with lock:
                order_count += 1
            time.sleep(0.01)  # Simulate API latency
            return MagicMock(id=f'order_{order_count}', status='accepted')
        
        mock_api = MagicMock()
        mock_api.submit_order = mock_submit
        mock_api.get_account.return_value = MagicMock(
            equity=10000.0, buying_power=10000.0, status='ACTIVE'
        )
        mock_api.list_positions.return_value = []
        
        with patch('alpaca_trade_api.REST', return_value=mock_api):
            trader = LiveTrader(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            trader.api = mock_api
            
            # Submit orders from multiple threads
            threads = []
            for i in range(5):
                t = threading.Thread(
                    target=trader.submit_order_with_retry,
                    kwargs={'symbol': 'BTCUSD', 'qty': 0.01, 'side': 'buy'}
                )
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join(timeout=5.0)
            
            assert order_count == 5, "All orders should be submitted"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
