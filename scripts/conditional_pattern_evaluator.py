"""
Conditional Pattern Evaluator
Sistema para evaluar patrones condicionales complejos

Ejemplos de patrones a evaluar:
1. "Precio toca EMA22 + Squeeze Momentum pendiente negativa + Precio debajo POC" → ¿Movimiento contrario?
2. "Volumen alto + Movimiento importante" → ¿Qué parámetros se repiten?
3. "IFVG + Confirmación multi-timeframe + Volumen" → ¿Probabilidad de éxito?

Este sistema permite:
- Definir condiciones complejas
- Evaluar probabilidad de éxito
- Identificar parámetros óptimos
- Generar reglas de trading automáticas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib
from dataclasses import dataclass
from enum import Enum


class ConditionType(Enum):
    """Tipos de condiciones evaluables"""
    PRICE_NEAR_EMA = "price_near_ema"
    SQUEEZE_MOMENTUM_SLOPE = "squeeze_momentum_slope"
    PRICE_VS_POC = "price_vs_poc"
    VOLUME_HIGH = "volume_high"
    PRICE_MOVEMENT_LARGE = "price_movement_large"
    IFVG_PRESENT = "ifvg_present"
    MULTI_TF_ALIGNED = "multi_tf_aligned"
    EMA_CROSS = "ema_cross"
    ATR_EXPANSION = "atr_expansion"
    VOLATILITY_SPIKE = "volatility_spike"


@dataclass
class Condition:
    """Definición de una condición"""
    type: ConditionType
    params: Dict
    weight: float = 1.0  # Peso de la condición en el patrón
    
    def __repr__(self):
        return f"{self.type.value}({self.params})"


@dataclass
class Pattern:
    """Patrón compuesto por múltiples condiciones"""
    name: str
    conditions: List[Condition]
    require_all: bool = True  # True=AND, False=OR
    
    def __repr__(self):
        operator = "AND" if self.require_all else "OR"
        cond_str = f" {operator} ".join([str(c) for c in self.conditions])
        return f"{self.name}: {cond_str}"


@dataclass
class PatternResult:
    """Resultado de evaluación de un patrón"""
    pattern: Pattern
    occurrences: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    best_params: Dict
    trades: List[Dict]
    
    def __repr__(self):
        return (f"Pattern: {self.pattern.name}\n"
                f"  Occurrences: {self.occurrences}\n"
                f"  Win Rate: {self.win_rate:.2%}\n"
                f"  Expectancy: {self.expectancy:.4f}\n"
                f"  Profit Factor: {self.profit_factor:.2f}")


class ConditionalPatternEvaluator:
    """
    Evalúa patrones condicionales complejos en datos de trading
    
    Permite definir condiciones como:
    - "Precio está a X% de EMA"
    - "Squeeze momentum tiene pendiente negativa"
    - "Volumen es Y veces mayor que promedio"
    
    Y evaluar su probabilidad de éxito en predicción de movimientos
    """
    
    def __init__(self, df: pd.DataFrame, forward_bars: int = 10, profit_threshold: float = 0.01):
        """
        Args:
            df: DataFrame con OHLCV y indicadores
            forward_bars: Barras hacia adelante para evaluar resultado
            profit_threshold: Umbral de profit para considerar trade ganador (1% default)
        """
        self.df = df.copy()
        self.forward_bars = forward_bars
        self.profit_threshold = profit_threshold
        
        # Calcular indicadores necesarios
        self._calculate_indicators()
        
    def _calculate_indicators(self):
        """Calcular indicadores técnicos necesarios"""
        close = self.df['close'].values.astype(float)
        high = self.df['high'].values.astype(float)
        low = self.df['low'].values.astype(float)
        volume = self.df['volume'].values.astype(float)
        
        # EMAs
        for period in [5, 9, 21, 22, 34, 50, 100, 200]:
            self.df[f'ema{period}'] = talib.EMA(close, timeperiod=period)
        
        # ATR
        self.df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Volume MA
        self.df['volume_ma'] = talib.SMA(volume, timeperiod=20)
        
        # Squeeze Momentum (simplified)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        kc_middle = talib.EMA(close, timeperiod=20)
        kc_upper = kc_middle + (self.df['atr'] * 1.5)
        kc_lower = kc_middle - (self.df['atr'] * 1.5)
        
        self.df['squeeze_on'] = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        
        # Momentum
        self.df['momentum'] = close - np.roll(close, 10)
        self.df['momentum_slope'] = self.df['momentum'].diff(5)
        
        # Price change
        self.df['price_change_pct'] = self.df['close'].pct_change()
        
        # Forward returns (para evaluar resultados)
        self.df['forward_return'] = self.df['close'].shift(-self.forward_bars) / self.df['close'] - 1
        self.df['forward_max'] = self.df['high'].rolling(window=self.forward_bars).max().shift(-self.forward_bars)
        self.df['forward_min'] = self.df['low'].rolling(window=self.forward_bars).min().shift(-self.forward_bars)
        
    def _evaluate_condition(self, condition: Condition, idx: int) -> bool:
        """
        Evaluar si una condición se cumple en un índice dado
        
        Args:
            condition: Condición a evaluar
            idx: Índice en el DataFrame
            
        Returns:
            True si la condición se cumple
        """
        if idx < 0 or idx >= len(self.df):
            return False
            
        row = self.df.iloc[idx]
        
        if condition.type == ConditionType.PRICE_NEAR_EMA:
            # Precio cerca de EMA
            ema_period = condition.params.get('period', 22)
            tolerance_pct = condition.params.get('tolerance_pct', 0.5)  # 0.5% default
            
            ema_col = f'ema{ema_period}'
            if ema_col not in self.df.columns or pd.isna(row[ema_col]):
                return False
                
            distance_pct = abs(row['close'] - row[ema_col]) / row[ema_col] * 100
            return distance_pct <= tolerance_pct
            
        elif condition.type == ConditionType.SQUEEZE_MOMENTUM_SLOPE:
            # Pendiente del momentum del squeeze
            direction = condition.params.get('direction', 'negative')  # 'positive' or 'negative'
            
            if pd.isna(row['momentum_slope']):
                return False
                
            if direction == 'negative':
                return row['momentum_slope'] < 0
            else:
                return row['momentum_slope'] > 0
                
        elif condition.type == ConditionType.PRICE_VS_POC:
            # Precio relativo al POC
            position = condition.params.get('position', 'below')  # 'above' or 'below'
            
            # Calcular POC (simplificado - usar price actual como proxy)
            # En implementación real, usar Volume Profile calculado
            if 'poc' not in self.df.columns:
                # Si no hay POC, usar SMA como proxy
                if 'ema50' in self.df.columns and not pd.isna(row['ema50']):
                    poc = row['ema50']
                else:
                    return False
            else:
                poc = row['poc']
                
            if position == 'below':
                return row['close'] < poc
            else:
                return row['close'] > poc
                
        elif condition.type == ConditionType.VOLUME_HIGH:
            # Volumen alto relativo a promedio
            multiplier = condition.params.get('multiplier', 1.5)
            
            if pd.isna(row['volume_ma']) or row['volume_ma'] == 0:
                return False
                
            return row['volume'] > row['volume_ma'] * multiplier
            
        elif condition.type == ConditionType.PRICE_MOVEMENT_LARGE:
            # Movimiento de precio importante
            threshold_pct = condition.params.get('threshold_pct', 2.0)  # 2% default
            
            if pd.isna(row['price_change_pct']):
                return False
                
            return abs(row['price_change_pct'] * 100) >= threshold_pct
            
        elif condition.type == ConditionType.IFVG_PRESENT:
            # IFVG presente
            direction = condition.params.get('direction', 'any')  # 'bullish', 'bearish', 'any'
            
            # Detectar IFVG (simplificado)
            if idx < 2:
                return False
                
            curr_low = row['low']
            curr_high = row['high']
            prev2_high = self.df.iloc[idx-2]['high']
            prev2_low = self.df.iloc[idx-2]['low']
            
            bullish_ifvg = curr_low > prev2_high
            bearish_ifvg = curr_high < prev2_low
            
            if direction == 'bullish':
                return bullish_ifvg
            elif direction == 'bearish':
                return bearish_ifvg
            else:
                return bullish_ifvg or bearish_ifvg
                
        elif condition.type == ConditionType.EMA_CROSS:
            # Cruce de EMAs
            fast_period = condition.params.get('fast', 9)
            slow_period = condition.params.get('slow', 21)
            direction = condition.params.get('direction', 'bullish')
            
            fast_col = f'ema{fast_period}'
            slow_col = f'ema{slow_period}'
            
            if idx < 1 or fast_col not in self.df.columns or slow_col not in self.df.columns:
                return False
                
            curr_fast = row[fast_col]
            curr_slow = row[slow_col]
            prev_fast = self.df.iloc[idx-1][fast_col]
            prev_slow = self.df.iloc[idx-1][slow_col]
            
            if pd.isna(curr_fast) or pd.isna(curr_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
                return False
                
            if direction == 'bullish':
                return prev_fast <= prev_slow and curr_fast > curr_slow
            else:
                return prev_fast >= prev_slow and curr_fast < curr_slow
                
        elif condition.type == ConditionType.ATR_EXPANSION:
            # Expansión de ATR
            if idx < 1 or pd.isna(row['atr']):
                return False
                
            prev_atr = self.df.iloc[idx-1]['atr']
            if pd.isna(prev_atr) or prev_atr == 0:
                return False
                
            atr_change_pct = (row['atr'] - prev_atr) / prev_atr * 100
            threshold = condition.params.get('threshold_pct', 10)
            
            return atr_change_pct >= threshold
            
        return False
    
    def _evaluate_pattern(self, pattern: Pattern) -> PatternResult:
        """
        Evaluar un patrón completo en todo el dataset
        
        Returns:
            PatternResult con estadísticas del patrón
        """
        pattern_matches = []
        
        # Evaluar cada barra
        for idx in range(100, len(self.df) - self.forward_bars):  # Skip primeras 100 para indicadores
            # Evaluar todas las condiciones
            conditions_met = []
            
            for condition in pattern.conditions:
                is_met = self._evaluate_condition(condition, idx)
                conditions_met.append(is_met)
            
            # Determinar si el patrón se cumple
            if pattern.require_all:
                pattern_met = all(conditions_met)
            else:
                pattern_met = any(conditions_met)
            
            if pattern_met:
                # Calcular resultado forward
                forward_return = self.df.iloc[idx]['forward_return']
                
                if pd.notna(forward_return):
                    trade = {
                        'idx': idx,
                        'timestamp': self.df.index[idx],
                        'entry_price': self.df.iloc[idx]['close'],
                        'forward_return': forward_return,
                        'win': forward_return > self.profit_threshold,
                        'conditions_met': dict(zip([c.type.value for c in pattern.conditions], conditions_met))
                    }
                    pattern_matches.append(trade)
        
        # Calcular estadísticas
        if len(pattern_matches) == 0:
            return PatternResult(
                pattern=pattern,
                occurrences=0,
                win_rate=0,
                avg_profit=0,
                avg_loss=0,
                profit_factor=0,
                expectancy=0,
                best_params={},
                trades=[]
            )
        
        wins = [t for t in pattern_matches if t['win']]
        losses = [t for t in pattern_matches if not t['win']]
        
        win_rate = len(wins) / len(pattern_matches)
        avg_profit = np.mean([t['forward_return'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['forward_return'] for t in losses])) if losses else 0
        
        gross_profit = sum([t['forward_return'] for t in wins])
        gross_loss = abs(sum([t['forward_return'] for t in losses]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        expectancy = (avg_profit * win_rate) - (avg_loss * (1 - win_rate))
        
        return PatternResult(
            pattern=pattern,
            occurrences=len(pattern_matches),
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            best_params=self._find_best_params(pattern, pattern_matches),
            trades=pattern_matches
        )
    
    def _find_best_params(self, pattern: Pattern, trades: List[Dict]) -> Dict:
        """Encontrar los mejores parámetros dentro de los trades exitosos"""
        if not trades:
            return {}
            
        # Analizar trades ganadores para encontrar parámetros comunes
        winning_trades = [t for t in trades if t['win']]
        
        if not winning_trades:
            return {}
        
        # Extraer características comunes
        best_params = {
            'avg_forward_return': np.mean([t['forward_return'] for t in winning_trades]),
            'median_forward_return': np.median([t['forward_return'] for t in winning_trades]),
            'min_forward_return': min([t['forward_return'] for t in winning_trades]),
            'max_forward_return': max([t['forward_return'] for t in winning_trades]),
        }
        
        return best_params
    
    def evaluate_patterns(self, patterns: List[Pattern]) -> List[PatternResult]:
        """
        Evaluar múltiples patrones
        
        Args:
            patterns: Lista de patrones a evaluar
            
        Returns:
            Lista de PatternResult ordenada por expectancy
        """
        results = []
        
        for pattern in patterns:
            result = self._evaluate_pattern(pattern)
            results.append(result)
        
        # Ordenar por expectancy (descendente)
        results.sort(key=lambda x: x.expectancy, reverse=True)
        
        return results
    
    def auto_discover_patterns(self, max_patterns: int = 20) -> List[PatternResult]:
        """
        Descubrir automáticamente los mejores patrones
        
        Esta función prueba combinaciones comunes de condiciones
        y retorna las más exitosas
        """
        print("🔍 Auto-discovering patterns...")
        
        # Definir patrones comunes a probar
        patterns_to_test = []
        
        # 1. Patrón: Precio toca EMA + Squeeze pendiente negativa + Debajo POC
        patterns_to_test.append(Pattern(
            name="EMA_Touch_SqueezeNeg_BelowPOC",
            conditions=[
                Condition(ConditionType.PRICE_NEAR_EMA, {'period': 22, 'tolerance_pct': 0.5}),
                Condition(ConditionType.SQUEEZE_MOMENTUM_SLOPE, {'direction': 'negative'}),
                Condition(ConditionType.PRICE_VS_POC, {'position': 'below'})
            ],
            require_all=True
        ))
        
        # 2. Patrón: Volumen alto + Movimiento grande
        patterns_to_test.append(Pattern(
            name="HighVolume_LargeMove",
            conditions=[
                Condition(ConditionType.VOLUME_HIGH, {'multiplier': 2.0}),
                Condition(ConditionType.PRICE_MOVEMENT_LARGE, {'threshold_pct': 2.0})
            ],
            require_all=True
        ))
        
        # 3. Patrón: IFVG + Volumen alto
        patterns_to_test.append(Pattern(
            name="IFVG_HighVolume",
            conditions=[
                Condition(ConditionType.IFVG_PRESENT, {'direction': 'any'}),
                Condition(ConditionType.VOLUME_HIGH, {'multiplier': 1.5})
            ],
            require_all=True
        ))
        
        # 4. Patrón: EMA Cross + ATR Expansion
        patterns_to_test.append(Pattern(
            name="EMACross_ATRExpansion",
            conditions=[
                Condition(ConditionType.EMA_CROSS, {'fast': 9, 'slow': 21, 'direction': 'bullish'}),
                Condition(ConditionType.ATR_EXPANSION, {'threshold_pct': 10})
            ],
            require_all=True
        ))
        
        # 5. Variaciones de EMA touch con diferentes parámetros
        for ema_period in [9, 21, 22, 34, 50]:
            for tolerance in [0.3, 0.5, 1.0]:
                patterns_to_test.append(Pattern(
                    name=f"EMA{ema_period}_Touch_{tolerance}pct",
                    conditions=[
                        Condition(ConditionType.PRICE_NEAR_EMA, 
                                {'period': ema_period, 'tolerance_pct': tolerance})
                    ],
                    require_all=True
                ))
        
        # 6. Combinaciones de volumen alto con diferentes multipliers
        for multiplier in [1.5, 2.0, 2.5, 3.0]:
            patterns_to_test.append(Pattern(
                name=f"HighVolume_{multiplier}x",
                conditions=[
                    Condition(ConditionType.VOLUME_HIGH, {'multiplier': multiplier})
                ],
                require_all=True
            ))
        
        print(f"📊 Testing {len(patterns_to_test)} pattern combinations...")
        
        # Evaluar todos los patrones
        results = self.evaluate_patterns(patterns_to_test)
        
        # Filtrar solo patrones con suficientes ocurrencias y expectancy positivo
        valid_results = [r for r in results if r.occurrences >= 10 and r.expectancy > 0]
        
        print(f"✅ Found {len(valid_results)} valid patterns with positive expectancy")
        
        return valid_results[:max_patterns]
    
    def generate_report(self, results: List[PatternResult], filename: str = "conditional_patterns_report.md"):
        """Generar reporte de patrones encontrados"""
        report = []
        report.append("# Conditional Patterns Evaluation Report\n")
        report.append(f"**Dataset:** {len(self.df)} bars\n")
        report.append(f"**Forward bars:** {self.forward_bars}\n")
        report.append(f"**Profit threshold:** {self.profit_threshold:.2%}\n")
        report.append(f"**Total patterns evaluated:** {len(results)}\n\n")
        
        report.append("---\n\n")
        report.append("## 🏆 Top Patterns by Expectancy\n\n")
        
        for i, result in enumerate(results[:20], 1):
            report.append(f"### {i}. {result.pattern.name}\n\n")
            report.append(f"**Pattern Definition:**\n")
            report.append(f"```\n{result.pattern}\n```\n\n")
            
            report.append(f"**Performance Metrics:**\n")
            report.append(f"- **Occurrences:** {result.occurrences}\n")
            report.append(f"- **Win Rate:** {result.win_rate:.2%}\n")
            report.append(f"- **Expectancy:** {result.expectancy:.4f}\n")
            report.append(f"- **Profit Factor:** {result.profit_factor:.2f}\n")
            report.append(f"- **Avg Profit:** {result.avg_profit:.4f}\n")
            report.append(f"- **Avg Loss:** {result.avg_loss:.4f}\n\n")
            
            if result.best_params:
                report.append(f"**Best Parameters:**\n")
                for key, value in result.best_params.items():
                    if isinstance(value, float):
                        report.append(f"- {key}: {value:.4f}\n")
                    else:
                        report.append(f"- {key}: {value}\n")
                report.append("\n")
            
            report.append("---\n\n")
        
        # Guardar reporte
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"📄 Report saved to: {filename}")
        
        return filename


# === EJEMPLO DE USO ===
if __name__ == "__main__":
    # Cargar datos
    df = pd.read_csv('data/btc_15Min.csv', parse_dates=['timestamp'], index_col='timestamp')
    
    # Crear evaluador
    evaluator = ConditionalPatternEvaluator(
        df=df,
        forward_bars=10,  # Evaluar resultado a 10 barras (2.5 horas en 15min)
        profit_threshold=0.01  # 1% profit para considerar ganador
    )
    
    # Auto-descubrir patrones
    results = evaluator.auto_discover_patterns(max_patterns=20)
    
    # Generar reporte
    evaluator.generate_report(results, filename="reports/conditional_patterns_report.md")
    
    # Imprimir top 5
    print("\n🏆 TOP 5 PATTERNS:\n")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result.pattern.name}")
        print(f"   Win Rate: {result.win_rate:.2%} | Expectancy: {result.expectancy:.4f}")
        print(f"   Occurrences: {result.occurrences}")
        print()
