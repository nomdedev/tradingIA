"""
Risk Manager Component.
Actúa como el guardián final antes de la ejecución.
Mantiene estado de riesgo (drawdown diario, exposición total).
Implementa Kill Switch.

ÁREA 6 IMPROVEMENTS:
- Total drawdown desde high water mark (no solo diario)
- Tracking de pérdidas consecutivas
- Correlación de posiciones para ajustar exposición
- VaR/CVaR básico
"""

import logging
from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime, date
from collections import deque
import numpy as np

# AUDITORÍA FIX: Importar constantes globales
try:
    from core.constants import (
        DEFAULT_MAX_DAILY_DRAWDOWN,
        DEFAULT_MAX_TOTAL_DRAWDOWN,
        DEFAULT_MAX_CONSECUTIVE_LOSSES,
        DEFAULT_VAR_CONFIDENCE,
        DEFAULT_CVAR_CONFIDENCE,
        DEFAULT_MAX_VAR_PCT,
    )
except ImportError:
    DEFAULT_MAX_DAILY_DRAWDOWN = 0.05
    DEFAULT_MAX_TOTAL_DRAWDOWN = 0.15
    DEFAULT_MAX_CONSECUTIVE_LOSSES = 5
    DEFAULT_VAR_CONFIDENCE = 0.95
    DEFAULT_CVAR_CONFIDENCE = 0.95
    DEFAULT_MAX_VAR_PCT = 0.02

logger = logging.getLogger(__name__)


class RiskManager:
    """
    ÁREA 6 FIX: Risk Manager Mejorado.
    
    Mejoras implementadas:
    1. Total drawdown desde high water mark
    2. Consecutive losses tracking
    3. Correlation-adjusted exposure
    4. VaR/CVaR básico
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # AUDITORÍA FIX: Usar constantes globales en lugar de magic numbers
        # Límites de drawdown
        self.max_daily_drawdown = self.config.get("max_daily_drawdown", DEFAULT_MAX_DAILY_DRAWDOWN)
        self.max_total_drawdown = self.config.get("max_total_drawdown", DEFAULT_MAX_TOTAL_DRAWDOWN)
        self.max_exposure = self.config.get("max_exposure", 1.0)  # 100% leverage
        self.kill_switch_file = self.config.get("kill_switch_file", "kill_switch.json")
        
        # ÁREA 6: Límites adicionales
        self.max_consecutive_losses = self.config.get("max_consecutive_losses", DEFAULT_MAX_CONSECUTIVE_LOSSES)
        self.max_correlation_exposure = self.config.get("max_correlation_exposure", 0.7)
        self.var_confidence = self.config.get("var_confidence", DEFAULT_VAR_CONFIDENCE)
        self.cvar_confidence = self.config.get("cvar_confidence", DEFAULT_CVAR_CONFIDENCE)
        self.max_var_pct = self.config.get("max_var_pct", DEFAULT_MAX_VAR_PCT)  # AUDITORÍA FIX: configurable
        
        # Estado básico
        self.daily_start_equity = 0.0
        self.current_equity = 0.0
        self.current_date = None
        self.is_halted = False
        
        # ÁREA 6: High Water Mark tracking
        self.high_water_mark = 0.0
        self.initial_equity = 0.0
        
        # ÁREA 6: Consecutive losses tracking
        self.consecutive_losses = 0
        self.trade_results: deque = deque(maxlen=100)  # Últimos 100 trades
        
        # ÁREA 6: Returns history for VaR
        self.returns_history: deque = deque(maxlen=252)  # ~1 año de datos diarios
        
        # ÁREA 6: Open positions for correlation
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        
        # AUDITORÍA FIX: Auto-inicializar si initial_equity está en config
        if "initial_equity" in self.config:
            self.initialize(self.config["initial_equity"])
        
    def initialize(self, initial_equity: float):
        """
        ÁREA 6: Inicializar con equity inicial para tracking completo.
        
        Args:
            initial_equity: Capital inicial del backtest/trading
        """
        if initial_equity <= 0:
            raise ValueError("Initial equity must be positive")
            
        self.initial_equity = initial_equity
        self.high_water_mark = initial_equity
        self.current_equity = initial_equity
        self.daily_start_equity = initial_equity
        self.consecutive_losses = 0
        self.trade_results.clear()
        self.returns_history.clear()
        logger.info(f"RiskManager initialized with equity: ${initial_equity:,.2f}")

    def update_state(self, equity: float, current_date: date):
        """Actualiza el estado del Risk Manager con el equity actual."""
        # Track daily reset
        if self.current_date != current_date:
            # Guardar return del día anterior
            if self.daily_start_equity > 0:
                daily_return = (self.current_equity - self.daily_start_equity) / self.daily_start_equity
                self.returns_history.append(daily_return)
            
            self.daily_start_equity = equity
            self.current_date = current_date

        self.current_equity = equity
        
        # ÁREA 6: Actualizar high water mark
        if equity > self.high_water_mark:
            self.high_water_mark = equity

        # Check Kill Switch file
        self._check_kill_switch_file()
    
    def record_trade_result(self, pnl: float, symbol: str = None):
        """
        ÁREA 6: Registrar resultado de trade para tracking de rachas.
        
        Args:
            pnl: Profit/Loss del trade
            symbol: Símbolo del trade (opcional)
        """
        self.trade_results.append({
            'pnl': pnl,
            'symbol': symbol,
            'timestamp': datetime.now(),
            'is_win': pnl > 0
        })
        
        # Actualizar contador de pérdidas consecutivas
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                logger.warning(
                    f"⚠️ {self.consecutive_losses} pérdidas consecutivas "
                    f"(límite: {self.max_consecutive_losses})"
                )
        else:
            self.consecutive_losses = 0
    
    def get_total_drawdown(self) -> float:
        """
        ÁREA 6: Calcular drawdown total desde high water mark.
        
        Returns:
            Total drawdown como decimal (ej: 0.10 = 10%)
        """
        if self.high_water_mark <= 0:
            return 0.0
        return (self.high_water_mark - self.current_equity) / self.high_water_mark
    
    def get_daily_drawdown(self) -> float:
        """
        Calcular drawdown del día actual.
        
        Returns:
            Daily drawdown como decimal
        """
        if self.daily_start_equity <= 0:
            return 0.0
        return (self.daily_start_equity - self.current_equity) / self.daily_start_equity
    
    def calculate_var(self, position_value: float) -> float:
        """
        ÁREA 6: Calcular Value at Risk histórico.
        
        Args:
            position_value: Valor de la posición en USD
            
        Returns:
            VaR en USD (pérdida máxima esperada al nivel de confianza)
        """
        if len(self.returns_history) < 20:
            # No hay suficientes datos, usar estimación conservadora
            return position_value * 0.03  # 3% VaR default
        
        returns = np.array(self.returns_history)
        var_percentile = 100 * (1 - self.var_confidence)
        var_return = np.percentile(returns, var_percentile)
        
        return abs(var_return * position_value)
    
    def calculate_cvar(self, position_value: float) -> float:
        """
        ÁREA 6: Calcular Conditional VaR (Expected Shortfall).
        
        Args:
            position_value: Valor de la posición en USD
            
        Returns:
            CVaR en USD (pérdida esperada dado que excedemos VaR)
        """
        if len(self.returns_history) < 20:
            return position_value * 0.05  # 5% CVaR default
        
        returns = np.array(self.returns_history)
        var_percentile = 100 * (1 - self.cvar_confidence)
        var_threshold = np.percentile(returns, var_percentile)
        
        # CVaR es el promedio de retornos por debajo del VaR
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) == 0:
            return self.calculate_var(position_value)
        
        cvar_return = np.mean(tail_returns)
        return abs(cvar_return * position_value)
    
    def update_correlation_matrix(self, correlations: np.ndarray):
        """
        ÁREA 6: Actualizar matriz de correlación entre assets.
        
        Args:
            correlations: Matriz de correlación NxN
        """
        self.correlation_matrix = correlations
    
    def calculate_correlated_risk(
        self,
        new_position: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        ÁREA 6: Calcular riesgo ajustado por correlación.
        
        Args:
            new_position: {'symbol': str, 'value': float, 'direction': str}
            
        Returns:
            Dict con riesgo correlacionado y recomendación
        """
        if not self.open_positions:
            return {
                'correlated_exposure': 0.0,
                'marginal_risk': 1.0,
                'recommendation': 'No open positions - full size OK'
            }
        
        # Si no tenemos matriz de correlación, asumir correlación media
        avg_correlation = 0.5
        
        total_exposure = sum(p.get('value', 0) for p in self.open_positions.values())
        new_value = new_position.get('value', 0)
        
        # Riesgo correlacionado simplificado
        # Con correlación perfecta (1.0), el riesgo se suma linealmente
        # Con correlación 0, el riesgo se suma como sqrt(sum of squares)
        
        if self.correlation_matrix is not None:
            # Usar matriz real si está disponible
            # Por simplicidad, usamos promedio de correlaciones
            avg_correlation = np.mean(np.abs(self.correlation_matrix))
        
        # Modelo simplificado de riesgo correlacionado
        combined_exposure = total_exposure + new_value
        correlated_exposure = combined_exposure * (0.5 + 0.5 * avg_correlation)
        
        # Factor de reducción si hay alta correlación
        if avg_correlation > 0.7:
            marginal_risk = 0.5  # Reducir tamaño 50%
            recommendation = "⚠️ Alta correlación - reducir tamaño 50%"
        elif avg_correlation > 0.5:
            marginal_risk = 0.75
            recommendation = "ℹ️ Correlación moderada - reducir tamaño 25%"
        else:
            marginal_risk = 1.0
            recommendation = "✅ Baja correlación - tamaño completo OK"
        
        return {
            'correlated_exposure': correlated_exposure,
            'avg_correlation': avg_correlation,
            'marginal_risk': marginal_risk,
            'recommendation': recommendation,
            'total_open_value': total_exposure,
        }
    
    def add_position(self, symbol: str, position_data: Dict[str, Any]):
        """ÁREA 6: Registrar nueva posición abierta."""
        self.open_positions[symbol] = {
            **position_data,
            'opened_at': datetime.now()
        }
    
    def remove_position(self, symbol: str):
        """ÁREA 6: Eliminar posición cerrada."""
        if symbol in self.open_positions:
            del self.open_positions[symbol]

    def _check_kill_switch_file(self):
        """Verifica si existe un archivo de kill switch activado externamente."""
        # AUDITORÍA FIX: Manejar race condition si archivo se elimina entre exists y open
        try:
            if os.path.exists(self.kill_switch_file):
                with open(self.kill_switch_file, "r") as f:
                    data = json.load(f)
                    if data.get("active", False):
                        self.is_halted = True
                        logger.critical(f"KILL SWITCH ACTIVATED via file: {data.get('reason')}")
        except FileNotFoundError:
            # Archivo eliminado entre exists() y open() - no es error
            pass
        except json.JSONDecodeError as e:
            logger.warning(f"Kill switch file corrupted: {e}")
        except Exception as e:
            logger.warning(f"Error checking kill switch file: {e}")

    def check_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifica si una orden puede ser ejecutada.
        
        ÁREA 6: Ahora incluye verificación de:
        - Total drawdown (no solo diario)
        - Consecutive losses
        - VaR limit
        
        Retorna {'allowed': bool, 'reason': str, 'risk_metrics': dict}
        """
        if self.is_halted:
            return {
                "allowed": False, 
                "reason": "System Halted (Kill Switch)",
                "risk_metrics": self._get_risk_metrics()
            }

        # 1. ÁREA 6: Check Total Drawdown (desde high water mark)
        total_dd = self.get_total_drawdown()
        if total_dd > self.max_total_drawdown:
            self.is_halted = True
            return {
                "allowed": False, 
                "reason": f"Max Total Drawdown exceeded: {total_dd:.2%} > {self.max_total_drawdown:.2%}",
                "risk_metrics": self._get_risk_metrics()
            }

        # 2. Check Daily Drawdown
        daily_dd = self.get_daily_drawdown()
        if daily_dd > self.max_daily_drawdown:
            self.is_halted = True
            return {
                "allowed": False, 
                "reason": f"Max Daily Drawdown exceeded: {daily_dd:.2%}",
                "risk_metrics": self._get_risk_metrics()
            }
        
        # 3. ÁREA 6: Check Consecutive Losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return {
                "allowed": False,
                "reason": f"Max consecutive losses reached: {self.consecutive_losses}",
                "risk_metrics": self._get_risk_metrics()
            }
        
        # 4. ÁREA 6: Check VaR limit (si hay position_value en request)
        position_value = order_request.get('position_value', 0)
        if position_value > 0:
            var = self.calculate_var(position_value)
            # AUDITORÍA FIX: Usar constante configurable en lugar de hardcoded
            max_var = self.current_equity * self.max_var_pct
            if var > max_var:
                return {
                    "allowed": False,
                    "reason": f"VaR too high: ${var:,.0f} > ${max_var:,.0f}",
                    "risk_metrics": self._get_risk_metrics()
                }

        return {
            "allowed": True, 
            "reason": "Risk Checks Passed",
            "risk_metrics": self._get_risk_metrics()
        }
    
    def _get_risk_metrics(self) -> Dict[str, Any]:
        """ÁREA 6: Obtener métricas de riesgo actuales."""
        return {
            'current_equity': self.current_equity,
            'high_water_mark': self.high_water_mark,
            'total_drawdown': self.get_total_drawdown(),
            'daily_drawdown': self.get_daily_drawdown(),
            'consecutive_losses': self.consecutive_losses,
            'is_halted': self.is_halted,
            'open_positions_count': len(self.open_positions),
            'var_95': self.calculate_var(self.current_equity * 0.1) if self.current_equity > 0 else 0,
        }
    
    def get_position_size_adjustment(self) -> float:
        """
        ÁREA 6: Obtener factor de ajuste de tamaño basado en riesgo actual.
        
        Returns:
            Factor multiplicador (0.0 - 1.0) para ajustar tamaño de posición
        """
        adjustment = 1.0
        
        # Reducir por drawdown
        total_dd = self.get_total_drawdown()
        if total_dd > 0.05:  # > 5% drawdown
            adjustment *= (1.0 - total_dd)  # Reducir proporcionalmente
        
        # Reducir por pérdidas consecutivas
        if self.consecutive_losses >= 3:
            adjustment *= 0.7  # -30%
        elif self.consecutive_losses >= 2:
            adjustment *= 0.85  # -15%
        
        # Mínimo 20% del tamaño
        return max(0.2, min(1.0, adjustment))

    def activate_kill_switch(self, reason: str):
        self.is_halted = True
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        # Persistir estado
        with open(self.kill_switch_file, "w") as f:
            json.dump({"active": True, "reason": reason, "timestamp": str(datetime.now())}, f)

    def reset_kill_switch(self):
        self.is_halted = False
        if os.path.exists(self.kill_switch_file):
            os.remove(self.kill_switch_file)
        logger.info("Kill Switch Reset")
    
    def reset(self):
        """ÁREA 6: Reset completo del estado."""
        self.is_halted = False
        self.consecutive_losses = 0
        self.trade_results.clear()
        self.open_positions.clear()
        self.returns_history.clear()
        self.high_water_mark = self.initial_equity
        self.current_equity = self.initial_equity
        self.daily_start_equity = self.initial_equity
        logger.info("RiskManager reset complete")
