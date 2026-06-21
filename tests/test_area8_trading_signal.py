"""
Test ÁREA 8: TradingSignal Standard.

Valida que el dataclass TradingSignal tenga:
1. Campos requeridos: timestamp, symbol, direction, entry_price
2. Campos de trazabilidad: reasons, council_approved
3. Serialización/deserialización correcta
4. Helpers para creación rápida
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
from datetime import datetime


class TestArea8TradingSignal(unittest.TestCase):
    """Tests para TradingSignal dataclass."""
    
    def test_required_fields(self):
        """Test que TradingSignal tenga campos requeridos."""
        from core.signals import TradingSignal, SignalDirection
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=50000.0,
        )
        
        # Campos requeridos deben existir
        self.assertIsNotNone(signal.timestamp)
        self.assertEqual(signal.symbol, "BTC")
        self.assertEqual(signal.direction, SignalDirection.LONG)
        self.assertEqual(signal.entry_price, 50000.0)
    
    def test_optional_fields_with_defaults(self):
        """Test que campos opcionales tengan defaults sensatos."""
        from core.signals import TradingSignal, SignalDirection
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="ETH",
            direction=SignalDirection.SHORT,
            entry_price=3000.0,
        )
        
        # Defaults
        self.assertEqual(signal.confidence, 0.5)
        self.assertEqual(signal.strategy_name, "unknown")
        self.assertEqual(signal.timeframe, "1H")
        self.assertIsNone(signal.stop_loss)
        self.assertIsNone(signal.take_profit)
        self.assertEqual(signal.reasons, [])
        self.assertEqual(signal.indicators_snapshot, {})
        self.assertTrue(signal.council_approved)
    
    def test_reasons_field_exists(self):
        """Test que tenga campo reasons para trazabilidad."""
        from core.signals import TradingSignal, SignalDirection
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=50000.0,
            reasons=["RSI oversold", "MA crossover", "Volume spike"]
        )
        
        self.assertEqual(len(signal.reasons), 3)
        self.assertIn("RSI oversold", signal.reasons)
    
    def test_council_integration_fields(self):
        """Test campos de integración con Council."""
        from core.signals import TradingSignal, SignalDirection
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=50000.0,
        )
        
        # Marcar aprobación del Council
        signal.mark_council_approval(
            approved=True,
            score=0.85,
            reasons=["Regime favorable", "Risk within limits"]
        )
        
        self.assertTrue(signal.council_approved)
        self.assertEqual(signal.council_score, 0.85)
        self.assertEqual(len(signal.council_reasons), 2)
    
    def test_signal_direction_enum(self):
        """Test que SignalDirection tenga valores esperados."""
        from core.signals import SignalDirection
        
        # Valores esperados
        self.assertEqual(SignalDirection.LONG.value, "long")
        self.assertEqual(SignalDirection.SHORT.value, "short")
        self.assertEqual(SignalDirection.CLOSE_LONG.value, "close_long")
        self.assertEqual(SignalDirection.CLOSE_SHORT.value, "close_short")
        self.assertEqual(SignalDirection.HOLD.value, "hold")
    
    def test_signal_strength(self):
        """Test cálculo de strength basado en confidence."""
        from core.signals import TradingSignal, SignalDirection, SignalStrength
        
        # Weak
        weak = TradingSignal(
            timestamp=datetime.now(), symbol="BTC",
            direction=SignalDirection.LONG, entry_price=50000,
            confidence=0.2
        )
        self.assertEqual(weak.strength, SignalStrength.WEAK)
        
        # Strong
        strong = TradingSignal(
            timestamp=datetime.now(), symbol="BTC",
            direction=SignalDirection.LONG, entry_price=50000,
            confidence=0.75
        )
        self.assertEqual(strong.strength, SignalStrength.STRONG)
        
        # Very Strong
        very_strong = TradingSignal(
            timestamp=datetime.now(), symbol="BTC",
            direction=SignalDirection.LONG, entry_price=50000,
            confidence=0.9
        )
        self.assertEqual(very_strong.strength, SignalStrength.VERY_STRONG)
    
    def test_is_entry_is_exit_properties(self):
        """Test propiedades is_entry y is_exit."""
        from core.signals import TradingSignal, SignalDirection
        
        entry_signal = TradingSignal(
            timestamp=datetime.now(), symbol="BTC",
            direction=SignalDirection.LONG, entry_price=50000
        )
        self.assertTrue(entry_signal.is_entry)
        self.assertFalse(entry_signal.is_exit)
        
        exit_signal = TradingSignal(
            timestamp=datetime.now(), symbol="BTC",
            direction=SignalDirection.CLOSE_LONG, entry_price=51000
        )
        self.assertFalse(exit_signal.is_entry)
        self.assertTrue(exit_signal.is_exit)
    
    def test_risk_reward_ratio(self):
        """Test cálculo de risk/reward ratio."""
        from core.signals import TradingSignal, SignalDirection
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=50000.0,
            stop_loss=49000.0,   # Risk: $1000
            take_profit=53000.0  # Reward: $3000
        )
        
        # R:R = 3000/1000 = 3.0
        self.assertAlmostEqual(signal.risk_reward_ratio, 3.0, places=2)
    
    def test_to_dict_serialization(self):
        """Test serialización a diccionario."""
        from core.signals import TradingSignal, SignalDirection
        
        signal = TradingSignal(
            timestamp=datetime(2024, 1, 15, 14, 30),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=50000.0,
            confidence=0.8,
            strategy_name="momentum",
            reasons=["test reason"]
        )
        
        data = signal.to_dict()
        
        self.assertEqual(data['symbol'], "BTC")
        self.assertEqual(data['direction'], "long")
        self.assertEqual(data['entry_price'], 50000.0)
        self.assertEqual(data['confidence'], 0.8)
        self.assertEqual(data['strategy_name'], "momentum")
        self.assertIn('timestamp', data)
        self.assertIn('signal_id', data)
    
    def test_from_dict_deserialization(self):
        """Test deserialización desde diccionario."""
        from core.signals import TradingSignal, SignalDirection
        
        data = {
            'timestamp': '2024-01-15T14:30:00',
            'symbol': 'ETH',
            'direction': 'short',
            'entry_price': 3000.0,
            'confidence': 0.7,
            'strategy_name': 'mean_reversion'
        }
        
        signal = TradingSignal.from_dict(data)
        
        self.assertEqual(signal.symbol, "ETH")
        self.assertEqual(signal.direction, SignalDirection.SHORT)
        self.assertEqual(signal.entry_price, 3000.0)
        self.assertEqual(signal.confidence, 0.7)
    
    def test_create_long_signal_helper(self):
        """Test helper create_long_signal."""
        from core.signals.trading_signal import create_long_signal
        
        signal = create_long_signal(
            symbol="BTC",
            entry_price=50000,
            timestamp=datetime.now(),
            strategy_name="test_strategy",
            confidence=0.9,
            reasons=["reason1", "reason2"]
        )
        
        self.assertTrue(signal.is_long)
        self.assertEqual(signal.symbol, "BTC")
        self.assertEqual(signal.confidence, 0.9)
        self.assertEqual(len(signal.reasons), 2)
    
    def test_create_short_signal_helper(self):
        """Test helper create_short_signal."""
        from core.signals.trading_signal import create_short_signal
        
        signal = create_short_signal(
            symbol="ETH",
            entry_price=3000,
            timestamp=datetime.now(),
            strategy_name="test_strategy"
        )
        
        self.assertTrue(signal.is_short)
        self.assertEqual(signal.symbol, "ETH")
    
    def test_create_exit_signal_helper(self):
        """Test helper create_exit_signal."""
        from core.signals.trading_signal import create_exit_signal
        
        signal = create_exit_signal(
            symbol="BTC",
            current_price=52000,
            timestamp=datetime.now(),
            strategy_name="test_strategy",
            is_long_position=True
        )
        
        self.assertTrue(signal.is_exit)
        self.assertEqual(signal.confidence, 1.0)  # Exit siempre es 100%
    
    def test_convert_legacy_signal(self):
        """Test conversión de señal legacy."""
        from core.signals.trading_signal import convert_legacy_signal, SignalDirection
        
        # Formato 1: action = buy
        legacy1 = {'action': 'buy', 'price': 50000, 'symbol': 'BTC'}
        signal1 = convert_legacy_signal(legacy1)
        self.assertEqual(signal1.direction, SignalDirection.LONG)
        self.assertEqual(signal1.entry_price, 50000)
        
        # Formato 2: signal = -1
        legacy2 = {'signal': -1, 'entry': 3000, 'symbol': 'ETH'}
        signal2 = convert_legacy_signal(legacy2)
        self.assertEqual(signal2.direction, SignalDirection.SHORT)
        
        # Formato 3: direction = short
        legacy3 = {'direction': 'short', 'entry_price': 100}
        signal3 = convert_legacy_signal(legacy3)
        self.assertEqual(signal3.direction, SignalDirection.SHORT)
    
    def test_confidence_normalization(self):
        """Test que confidence se normalice a [0, 1]."""
        from core.signals import TradingSignal, SignalDirection
        
        # Confidence > 1 se normaliza
        signal1 = TradingSignal(
            timestamp=datetime.now(), symbol="BTC",
            direction=SignalDirection.LONG, entry_price=50000,
            confidence=1.5
        )
        self.assertEqual(signal1.confidence, 1.0)
        
        # Confidence < 0 se normaliza
        signal2 = TradingSignal(
            timestamp=datetime.now(), symbol="BTC",
            direction=SignalDirection.LONG, entry_price=50000,
            confidence=-0.5
        )
        self.assertEqual(signal2.confidence, 0.0)
    
    def test_signal_id_generation(self):
        """Test que se genere signal_id automáticamente."""
        from core.signals import TradingSignal, SignalDirection
        
        signal = TradingSignal(
            timestamp=datetime(2024, 1, 15, 14, 30, 45),
            symbol="BTC",
            direction=SignalDirection.LONG,
            entry_price=50000,
            strategy_name="my_strategy"
        )
        
        self.assertIsNotNone(signal.signal_id)
        self.assertIn("my_strategy", signal.signal_id)
        self.assertIn("BTC", signal.signal_id)
    
    def test_add_reason_method(self):
        """Test método add_reason."""
        from core.signals import TradingSignal, SignalDirection
        
        signal = TradingSignal(
            timestamp=datetime.now(), symbol="BTC",
            direction=SignalDirection.LONG, entry_price=50000
        )
        
        signal.add_reason("Reason 1")
        signal.add_reason("Reason 2")
        signal.add_reason("Reason 1")  # Duplicado, no debe agregarse
        
        self.assertEqual(len(signal.reasons), 2)
    
    def test_add_indicator_method(self):
        """Test método add_indicator."""
        from core.signals import TradingSignal, SignalDirection
        
        signal = TradingSignal(
            timestamp=datetime.now(), symbol="BTC",
            direction=SignalDirection.LONG, entry_price=50000
        )
        
        signal.add_indicator("rsi", 28.5)
        signal.add_indicator("macd", 150.3)
        
        self.assertEqual(signal.indicators_snapshot['rsi'], 28.5)
        self.assertEqual(signal.indicators_snapshot['macd'], 150.3)


def run_tests():
    """Ejecutar tests y mostrar resumen."""
    print("=" * 60)
    print("ÁREA 8: TradingSignal Standard Tests")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestArea8TradingSignal)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("RESUMEN:")
    print(f"  Tests ejecutados: {result.testsRun}")
    print(f"  Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Fallidos: {len(result.failures)}")
    print(f"  Errores: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ÁREA 8: TODOS LOS TESTS PASARON")
    else:
        print("\n❌ ÁREA 8: HAY TESTS FALLIDOS")
        for test, trace in result.failures + result.errors:
            print(f"\n  - {test}")
            print(f"    {trace.split(chr(10))[0]}")
    
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
