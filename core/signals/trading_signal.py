"""
ÁREA 8: TradingSignal Dataclass Estandarizado.

Todas las estrategias deben retornar List[TradingSignal].
Esto permite:
1. Interfaz uniforme entre estrategias y backtester
2. Trazabilidad completa de señales
3. Integración con Council para validación
4. Metadata rica para análisis post-trade
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class SignalDirection(Enum):
    """Dirección de la señal."""
    LONG = "long"
    SHORT = "short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"


class SignalStrength(Enum):
    """Fuerza de la señal (para filtrar por confidence)."""
    WEAK = "weak"           # 0.0 - 0.3
    MODERATE = "moderate"   # 0.3 - 0.6
    STRONG = "strong"       # 0.6 - 0.8
    VERY_STRONG = "very_strong"  # 0.8 - 1.0


@dataclass
class TradingSignal:
    """
    ÁREA 8: Estructura estandarizada para señales de trading.
    
    Todas las estrategias deben retornar List[TradingSignal] en su
    método generate_signals().
    
    Example:
        >>> signal = TradingSignal(
        ...     timestamp=datetime.now(),
        ...     symbol="BTC",
        ...     direction=SignalDirection.LONG,
        ...     entry_price=50000.0,
        ...     confidence=0.75,
        ...     strategy_name="momentum_strategy"
        ... )
        >>> signal.to_dict()
        {'timestamp': ..., 'symbol': 'BTC', 'direction': 'long', ...}
    """
    
    # === CAMPOS REQUERIDOS ===
    timestamp: datetime
    """Momento en que se generó la señal (debe ser <= último dato disponible)."""
    
    symbol: str
    """Símbolo del activo (ej: 'BTC', 'ETH')."""
    
    direction: SignalDirection
    """Dirección de la operación."""
    
    entry_price: float
    """Precio de entrada sugerido."""
    
    # === CAMPOS OPCIONALES (con defaults) ===
    confidence: float = 0.5
    """Confianza en la señal (0.0 - 1.0)."""
    
    strategy_name: str = "unknown"
    """Nombre de la estrategia que generó la señal."""
    
    timeframe: str = "1H"
    """Timeframe principal de la señal."""
    
    stop_loss: Optional[float] = None
    """Precio de stop loss (None = sin stop)."""
    
    take_profit: Optional[float] = None
    """Precio de take profit (None = sin TP)."""
    
    position_size_pct: Optional[float] = None
    """Tamaño de posición sugerido (0.0 - 1.0, None = usar default)."""
    
    max_hold_periods: Optional[int] = None
    """Máximo de períodos para mantener la posición."""
    
    # === TRAZABILIDAD ===
    reasons: List[str] = field(default_factory=list)
    """Razones que motivaron la señal (para logging/debug)."""
    
    indicators_snapshot: Dict[str, float] = field(default_factory=dict)
    """Snapshot de indicadores al momento de la señal."""
    
    regime: Optional[str] = None
    """Régimen de mercado detectado (bull/bear/chop)."""
    
    # === COUNCIL INTEGRATION ===
    council_approved: bool = True
    """Si el Council aprobó esta señal."""
    
    council_score: Optional[float] = None
    """Score del Council (0.0 - 1.0)."""
    
    council_reasons: List[str] = field(default_factory=list)
    """Razones del Council para aprobar/rechazar."""
    
    # === METADATA ===
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadata adicional específica de la estrategia."""
    
    signal_id: Optional[str] = None
    """ID único de la señal (se genera automáticamente si no se provee)."""
    
    def __post_init__(self):
        """Validación y normalización post-inicialización."""
        # Generar signal_id si no existe
        if self.signal_id is None:
            ts_str = self.timestamp.strftime("%Y%m%d%H%M%S") if self.timestamp else "000000"
            self.signal_id = f"{self.strategy_name}_{self.symbol}_{ts_str}"
        
        # Normalizar confidence
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Validar dirección
        if isinstance(self.direction, str):
            self.direction = SignalDirection(self.direction.lower())
        
        # Validar precios positivos
        if self.entry_price is not None and self.entry_price <= 0:
            raise ValueError(f"entry_price debe ser positivo: {self.entry_price}")
        
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError(f"stop_loss debe ser positivo: {self.stop_loss}")
        
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError(f"take_profit debe ser positivo: {self.take_profit}")
    
    @property
    def strength(self) -> SignalStrength:
        """Obtener fuerza de señal basada en confidence."""
        if self.confidence >= 0.8:
            return SignalStrength.VERY_STRONG
        elif self.confidence >= 0.6:
            return SignalStrength.STRONG
        elif self.confidence >= 0.3:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    @property
    def is_entry(self) -> bool:
        """Indica si es una señal de entrada."""
        return self.direction in (SignalDirection.LONG, SignalDirection.SHORT)
    
    @property
    def is_exit(self) -> bool:
        """Indica si es una señal de salida."""
        return self.direction in (SignalDirection.CLOSE_LONG, SignalDirection.CLOSE_SHORT)
    
    @property
    def is_long(self) -> bool:
        """Indica si es una señal de compra/long."""
        return self.direction == SignalDirection.LONG
    
    @property
    def is_short(self) -> bool:
        """Indica si es una señal de venta/short."""
        return self.direction == SignalDirection.SHORT
    
    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calcular ratio risk/reward si hay SL y TP definidos."""
        if self.stop_loss is None or self.take_profit is None:
            return None
        
        if self.is_long:
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        
        if risk <= 0:
            return None
        
        return reward / risk
    
    def add_reason(self, reason: str):
        """Agregar una razón a la lista de razones."""
        if reason and reason not in self.reasons:
            self.reasons.append(reason)
    
    def add_indicator(self, name: str, value: float):
        """Agregar un indicador al snapshot."""
        self.indicators_snapshot[name] = value
    
    def mark_council_approval(
        self,
        approved: bool,
        score: float,
        reasons: List[str] = None
    ):
        """
        Marcar resultado de evaluación del Council.
        
        Args:
            approved: Si el Council aprobó la señal
            score: Score del Council (0.0 - 1.0)
            reasons: Razones de la decisión
        """
        self.council_approved = approved
        self.council_score = max(0.0, min(1.0, score))
        if reasons:
            self.council_reasons = reasons
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización."""
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'entry_price': self.entry_price,
            'confidence': self.confidence,
            'strength': self.strength.value,
            'strategy_name': self.strategy_name,
            'timeframe': self.timeframe,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size_pct': self.position_size_pct,
            'max_hold_periods': self.max_hold_periods,
            'reasons': self.reasons,
            'indicators_snapshot': self.indicators_snapshot,
            'regime': self.regime,
            'council_approved': self.council_approved,
            'council_score': self.council_score,
            'council_reasons': self.council_reasons,
            'risk_reward_ratio': self.risk_reward_ratio,
            'is_entry': self.is_entry,
            'is_exit': self.is_exit,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Crear TradingSignal desde diccionario."""
        # Parsear timestamp
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Parsear direction
        direction = data.get('direction', 'hold')
        if isinstance(direction, str):
            direction = SignalDirection(direction)
        
        return cls(
            timestamp=timestamp,
            symbol=data.get('symbol', ''),
            direction=direction,
            entry_price=data.get('entry_price', 0.0),
            confidence=data.get('confidence', 0.5),
            strategy_name=data.get('strategy_name', 'unknown'),
            timeframe=data.get('timeframe', '1H'),
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            position_size_pct=data.get('position_size_pct'),
            max_hold_periods=data.get('max_hold_periods'),
            reasons=data.get('reasons', []),
            indicators_snapshot=data.get('indicators_snapshot', {}),
            regime=data.get('regime'),
            council_approved=data.get('council_approved', True),
            council_score=data.get('council_score'),
            council_reasons=data.get('council_reasons', []),
            metadata=data.get('metadata', {}),
            signal_id=data.get('signal_id'),
        )
    
    def __str__(self) -> str:
        """Representación string legible."""
        return (
            f"TradingSignal({self.direction.value.upper()} {self.symbol} "
            f"@ {self.entry_price:.2f}, conf={self.confidence:.0%}, "
            f"strategy={self.strategy_name})"
        )
    
    def __repr__(self) -> str:
        """Representación para debug."""
        return (
            f"TradingSignal(signal_id='{self.signal_id}', "
            f"symbol='{self.symbol}', direction={self.direction}, "
            f"entry_price={self.entry_price}, confidence={self.confidence})"
        )


# ============================================================================
# Helper Functions para migración de estrategias
# ============================================================================

def create_long_signal(
    symbol: str,
    entry_price: float,
    timestamp: datetime,
    strategy_name: str,
    confidence: float = 0.5,
    stop_loss: float = None,
    take_profit: float = None,
    reasons: List[str] = None,
    **kwargs
) -> TradingSignal:
    """
    Helper para crear señal LONG rápidamente.
    
    Example:
        >>> signal = create_long_signal(
        ...     symbol="BTC",
        ...     entry_price=50000,
        ...     timestamp=datetime.now(),
        ...     strategy_name="momentum",
        ...     confidence=0.8,
        ...     reasons=["RSI oversold", "MA crossover"]
        ... )
    """
    return TradingSignal(
        timestamp=timestamp,
        symbol=symbol,
        direction=SignalDirection.LONG,
        entry_price=entry_price,
        confidence=confidence,
        strategy_name=strategy_name,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasons=reasons or [],
        **kwargs
    )


def create_short_signal(
    symbol: str,
    entry_price: float,
    timestamp: datetime,
    strategy_name: str,
    confidence: float = 0.5,
    stop_loss: float = None,
    take_profit: float = None,
    reasons: List[str] = None,
    **kwargs
) -> TradingSignal:
    """Helper para crear señal SHORT rápidamente."""
    return TradingSignal(
        timestamp=timestamp,
        symbol=symbol,
        direction=SignalDirection.SHORT,
        entry_price=entry_price,
        confidence=confidence,
        strategy_name=strategy_name,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasons=reasons or [],
        **kwargs
    )


def create_exit_signal(
    symbol: str,
    current_price: float,
    timestamp: datetime,
    strategy_name: str,
    is_long_position: bool = True,
    reasons: List[str] = None,
    **kwargs
) -> TradingSignal:
    """Helper para crear señal de cierre."""
    direction = SignalDirection.CLOSE_LONG if is_long_position else SignalDirection.CLOSE_SHORT
    
    return TradingSignal(
        timestamp=timestamp,
        symbol=symbol,
        direction=direction,
        entry_price=current_price,
        confidence=1.0,  # Exit signals son decisiones finales
        strategy_name=strategy_name,
        reasons=reasons or [],
        **kwargs
    )


def convert_legacy_signal(
    legacy_signal: Dict[str, Any],
    strategy_name: str = "legacy"
) -> TradingSignal:
    """
    Convertir señal legacy (dict) al formato TradingSignal.
    
    Soporta formatos comunes:
    - {'action': 'buy', 'price': 50000}
    - {'signal': 1, 'entry': 50000}  (1=buy, -1=sell, 0=hold)
    - {'direction': 'long', 'price': 50000}
    
    Args:
        legacy_signal: Dict con formato antiguo
        strategy_name: Nombre de estrategia
        
    Returns:
        TradingSignal estandarizado
    """
    # Detectar dirección
    direction = SignalDirection.HOLD
    
    # Formato 1: action = buy/sell
    if 'action' in legacy_signal:
        action = legacy_signal['action'].lower()
        if action in ('buy', 'long'):
            direction = SignalDirection.LONG
        elif action in ('sell', 'short'):
            direction = SignalDirection.SHORT
        elif action in ('close', 'exit'):
            direction = SignalDirection.CLOSE_LONG
    
    # Formato 2: signal = 1/-1/0
    elif 'signal' in legacy_signal:
        sig = legacy_signal['signal']
        if sig == 1:
            direction = SignalDirection.LONG
        elif sig == -1:
            direction = SignalDirection.SHORT
    
    # Formato 3: direction = long/short
    elif 'direction' in legacy_signal:
        dir_str = legacy_signal['direction'].lower()
        if dir_str in ('long', 'buy'):
            direction = SignalDirection.LONG
        elif dir_str in ('short', 'sell'):
            direction = SignalDirection.SHORT
    
    # Detectar precio
    price = (
        legacy_signal.get('price') or 
        legacy_signal.get('entry') or 
        legacy_signal.get('entry_price') or 
        0.0
    )
    
    # Detectar timestamp
    timestamp = legacy_signal.get('timestamp') or legacy_signal.get('time') or datetime.now()
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except (ValueError, TypeError):
            # AUDITORÍA FIX: Capturar excepciones específicas
            timestamp = datetime.now()
    
    return TradingSignal(
        timestamp=timestamp,
        symbol=legacy_signal.get('symbol', 'UNKNOWN'),
        direction=direction,
        entry_price=price,
        confidence=legacy_signal.get('confidence', 0.5),
        strategy_name=strategy_name,
        stop_loss=legacy_signal.get('stop_loss') or legacy_signal.get('sl'),
        take_profit=legacy_signal.get('take_profit') or legacy_signal.get('tp'),
        metadata={'legacy_format': True, 'original': legacy_signal}
    )
