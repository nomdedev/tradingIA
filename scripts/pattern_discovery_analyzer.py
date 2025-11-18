"""
Pattern Discovery Analyzer
Identifica patrones específicos que predicen movimientos de precio exitosos

Este módulo analiza:
1. Proximidad a EMAs (22, 55, 200) y probabilidad de rebote/ruptura
2. Influencia del POC y volumen de compras/ventas
3. Impacto de IFVG (Imbalance/Fair Value Gaps)
4. Efectividad del Squeeze Momentum
5. Confirmación de tendencias multi-timeframe (15m, 1h en operaciones de 5m)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import talib
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PatternResult:
    """Resultado de un patrón detectado"""
    condition: str
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    occurrences: int
    confidence: float


class PatternDiscoveryAnalyzer:
    """
    Analiza patrones específicos y su tasa de éxito
    """
    
    def __init__(self, min_occurrences: int = 10):
        self.min_occurrences = min_occurrences
        self.patterns = []
        
    def analyze_ema_proximity_patterns(self, df: pd.DataFrame, 
                                      ema_periods: List[int] = [22, 55, 200],
                                      proximity_thresholds: List[float] = [0.5, 1.0, 2.0, 3.0],
                                      forward_periods: int = 20) -> List[PatternResult]:
        """
        Analiza qué tan efectivo es operar cuando el precio está cerca de una EMA
        
        Args:
            df: DataFrame con OHLCV
            ema_periods: Períodos de EMAs a analizar
            proximity_thresholds: Distancias porcentuales a analizar (0.5% = 0.5)
            forward_periods: Períodos hacia adelante para medir éxito
        """
        results = []
        
        # Calcular EMAs
        for period in ema_periods:
            df[f'ema_{period}'] = talib.EMA(np.asarray(df['close']), timeperiod=period)
            df[f'dist_ema_{period}'] = ((df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']) * 100
        
        # Calcular retorno futuro
        df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        
        # Analizar cada combinación EMA + threshold
        for period in ema_periods:
            for threshold in proximity_thresholds:
                # Patrón: Precio cerca de EMA (dentro del threshold)
                condition = abs(df[f'dist_ema_{period}']) <= threshold
                
                # Subir: precio por debajo de EMA y sube
                bounce_up = condition & (df[f'dist_ema_{period}'] < 0) & (df['future_return'] > 0)
                bounce_up_wins = bounce_up.sum()
                bounce_up_total = (condition & (df[f'dist_ema_{period}'] < 0)).sum()
                
                if bounce_up_total >= self.min_occurrences:
                    avg_profit = df.loc[bounce_up, 'future_return'].mean() * 100
                    avg_loss = df.loc[(condition & (df[f'dist_ema_{period}'] < 0) & (df['future_return'] <= 0)), 'future_return'].mean() * 100
                    
                    results.append(PatternResult(
                        condition=f"Precio dentro {threshold}% DEBAJO EMA{period} → REBOTE ALCISTA",
                        win_rate=bounce_up_wins / bounce_up_total if bounce_up_total > 0 else 0,
                        avg_profit=avg_profit if not pd.isna(avg_profit) else 0,
                        avg_loss=avg_loss if not pd.isna(avg_loss) else 0,
                        profit_factor=abs(avg_profit / avg_loss) if (not pd.isna(avg_loss) and avg_loss != 0) else 0,
                        occurrences=bounce_up_total,
                        confidence=min(bounce_up_total / 50, 1.0)  # Confianza basada en muestras
                    ))
                
                # Bajar: precio por encima de EMA y baja
                bounce_down = condition & (df[f'dist_ema_{period}'] > 0) & (df['future_return'] < 0)
                bounce_down_wins = bounce_down.sum()
                bounce_down_total = (condition & (df[f'dist_ema_{period}'] > 0)).sum()
                
                if bounce_down_total >= self.min_occurrences:
                    avg_profit = abs(df.loc[bounce_down, 'future_return'].mean() * 100)
                    avg_loss = abs(df.loc[(condition & (df[f'dist_ema_{period}'] > 0) & (df['future_return'] >= 0)), 'future_return'].mean() * 100)
                    
                    results.append(PatternResult(
                        condition=f"Precio dentro {threshold}% ENCIMA EMA{period} → RECHAZO BAJISTA",
                        win_rate=bounce_down_wins / bounce_down_total if bounce_down_total > 0 else 0,
                        avg_profit=avg_profit if not pd.isna(avg_profit) else 0,
                        avg_loss=avg_loss if not pd.isna(avg_loss) else 0,
                        profit_factor=abs(avg_profit / avg_loss) if (not pd.isna(avg_loss) and avg_loss != 0) else 0,
                        occurrences=bounce_down_total,
                        confidence=min(bounce_down_total / 50, 1.0)
                    ))
        
        return results
    
    def analyze_volume_poc_patterns(self, df: pd.DataFrame,
                                   forward_periods: int = 20) -> List[PatternResult]:
        """
        Analiza patrones relacionados con volumen y POC (Point of Control)
        
        POC = Precio con mayor volumen en un período
        """
        results = []
        
        # Calcular volumen promedio
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calcular desbalance compra/venta (aproximación con volumen y precio)
        df['price_change'] = df['close'].pct_change()
        df['volume_buy'] = np.where(df['price_change'] > 0, df['volume'], 0)
        df['volume_sell'] = np.where(df['price_change'] < 0, df['volume'], 0)
        
        df['volume_buy_ma'] = df['volume_buy'].rolling(20).mean()
        df['volume_sell_ma'] = df['volume_sell'].rolling(20).mean()
        df['buy_sell_ratio'] = df['volume_buy_ma'] / (df['volume_sell_ma'] + 1e-10)
        
        # Calcular POC (precio modal en ventana deslizante)
        def calculate_poc(window_df):
            if len(window_df) < 5:
                return window_df['close'].iloc[-1]
            price_bins = pd.cut(window_df['close'], bins=10)
            volume_by_price = window_df.groupby(price_bins)['volume'].sum()
            poc_bin = volume_by_price.idxmax()
            return poc_bin.mid if not pd.isna(poc_bin) else window_df['close'].iloc[-1]
        
        df['poc'] = df['close'].rolling(50).apply(lambda x: calculate_poc(df.loc[x.index]), raw=False)
        df['dist_poc'] = ((df['close'] - df['poc']) / df['poc']) * 100
        
        # Calcular retorno futuro
        df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        
        # Patrón 1: Volumen alto + cerca de POC
        high_volume = df['volume_ratio'] > 1.5
        near_poc = abs(df['dist_poc']) < 1.0
        
        pattern_1 = high_volume & near_poc
        pattern_1_wins = (pattern_1 & (df['future_return'] > 0)).sum()
        pattern_1_total = pattern_1.sum()
        
        if pattern_1_total >= self.min_occurrences:
            avg_profit = df.loc[pattern_1 & (df['future_return'] > 0), 'future_return'].mean() * 100
            avg_loss = df.loc[pattern_1 & (df['future_return'] <= 0), 'future_return'].mean() * 100
            
            results.append(PatternResult(
                condition="Volumen Alto (>1.5x) + Precio cerca POC (±1%)",
                win_rate=pattern_1_wins / pattern_1_total,
                avg_profit=avg_profit if not pd.isna(avg_profit) else 0,
                avg_loss=avg_loss if not pd.isna(avg_loss) else 0,
                profit_factor=abs(avg_profit / avg_loss) if (not pd.isna(avg_loss) and avg_loss != 0) else 0,
                occurrences=pattern_1_total,
                confidence=min(pattern_1_total / 50, 1.0)
            ))
        
        # Patrón 2: Desbalance compra/venta fuerte
        strong_buy = df['buy_sell_ratio'] > 2.0
        strong_sell = df['buy_sell_ratio'] < 0.5
        
        for condition, name, expected_dir in [(strong_buy, "Desbalance COMPRA >2x", 1), 
                                               (strong_sell, "Desbalance VENTA >2x", -1)]:
            wins = (condition & (np.sign(df['future_return']) == expected_dir)).sum()
            total = condition.sum()
            
            if total >= self.min_occurrences:
                avg_profit = abs(df.loc[condition & (np.sign(df['future_return']) == expected_dir), 'future_return'].mean() * 100)
                avg_loss = abs(df.loc[condition & (np.sign(df['future_return']) != expected_dir), 'future_return'].mean() * 100)
                
                results.append(PatternResult(
                    condition=name,
                    win_rate=wins / total,
                    avg_profit=avg_profit if not pd.isna(avg_profit) else 0,
                    avg_loss=avg_loss if not pd.isna(avg_loss) else 0,
                    profit_factor=abs(avg_profit / avg_loss) if (not pd.isna(avg_loss) and avg_loss != 0) else 0,
                    occurrences=total,
                    confidence=min(total / 50, 1.0)
                ))
        
        return results
    
    def analyze_ifvg_patterns(self, df: pd.DataFrame,
                             forward_periods: int = 20) -> List[PatternResult]:
        """
        Analiza patrones de IFVG (Imbalance/Fair Value Gaps)
        
        IFVG = Gap entre vela actual y vela hace 2 períodos
        """
        results = []
        
        # Detectar IFVG alcistas (gap hacia arriba)
        df['ifvg_bullish'] = (df['low'] > df['high'].shift(2))
        
        # Detectar IFVG bajistas (gap hacia abajo)
        df['ifvg_bearish'] = (df['high'] < df['low'].shift(2))
        
        # Tamaño del gap
        df['ifvg_bull_size'] = np.where(df['ifvg_bullish'], 
                                        (df['low'] - df['high'].shift(2)) / df['close'] * 100, 
                                        0)
        df['ifvg_bear_size'] = np.where(df['ifvg_bearish'], 
                                        (df['low'].shift(2) - df['high']) / df['close'] * 100, 
                                        0)
        
        # Retorno futuro
        df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        
        # Patrón 1: IFVG alcista se llena (precio vuelve al gap)
        ifvg_bull_filled = df['ifvg_bullish'] & (df['low'].shift(-5) <= df['high'].shift(2))
        ifvg_bull_total = df['ifvg_bullish'].sum()
        
        if ifvg_bull_total >= self.min_occurrences:
            wins = (df['ifvg_bullish'] & (df['future_return'] > 0)).sum()
            avg_profit = df.loc[df['ifvg_bullish'] & (df['future_return'] > 0), 'future_return'].mean() * 100
            avg_loss = df.loc[df['ifvg_bullish'] & (df['future_return'] <= 0), 'future_return'].mean() * 100
            
            results.append(PatternResult(
                condition="IFVG Alcista (gap hacia arriba)",
                win_rate=wins / ifvg_bull_total,
                avg_profit=avg_profit if not pd.isna(avg_profit) else 0,
                avg_loss=avg_loss if not pd.isna(avg_loss) else 0,
                profit_factor=abs(avg_profit / avg_loss) if (not pd.isna(avg_loss) and avg_loss != 0) else 0,
                occurrences=ifvg_bull_total,
                confidence=min(ifvg_bull_total / 30, 1.0)
            ))
        
        # Patrón 2: IFVG bajista
        ifvg_bear_total = df['ifvg_bearish'].sum()
        
        if ifvg_bear_total >= self.min_occurrences:
            wins = (df['ifvg_bearish'] & (df['future_return'] < 0)).sum()
            avg_profit = abs(df.loc[df['ifvg_bearish'] & (df['future_return'] < 0), 'future_return'].mean() * 100)
            avg_loss = abs(df.loc[df['ifvg_bearish'] & (df['future_return'] >= 0), 'future_return'].mean() * 100)
            
            results.append(PatternResult(
                condition="IFVG Bajista (gap hacia abajo)",
                win_rate=wins / ifvg_bear_total,
                avg_profit=avg_profit if not pd.isna(avg_profit) else 0,
                avg_loss=avg_loss if not pd.isna(avg_loss) else 0,
                profit_factor=abs(avg_profit / avg_loss) if (not pd.isna(avg_loss) and avg_loss != 0) else 0,
                occurrences=ifvg_bear_total,
                confidence=min(ifvg_bear_total / 30, 1.0)
            ))
        
        return results
    
    def analyze_squeeze_momentum_patterns(self, df: pd.DataFrame,
                                         forward_periods: int = 20) -> List[PatternResult]:
        """
        Analiza efectividad del Squeeze Momentum en diferentes condiciones
        """
        results = []
        
        # Calcular Squeeze Momentum (simplificado)
        bb_length = 20
        kc_length = 20
        
        df['bb_basis'] = df['close'].rolling(bb_length).mean()
        df['bb_std'] = df['close'].rolling(bb_length).std()
        df['bb_upper'] = df['bb_basis'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_basis'] - (2 * df['bb_std'])
        
        df['kc_basis'] = df['close'].rolling(kc_length).mean()
        df['tr'] = np.maximum(df['high'] - df['low'],
                             np.maximum(abs(df['high'] - df['close'].shift(1)),
                                      abs(df['low'] - df['close'].shift(1))))
        df['tr_ma'] = df['tr'].rolling(bb_length).mean()
        df['kc_upper'] = df['kc_basis'] + (1.5 * df['tr_ma'])
        df['kc_lower'] = df['kc_basis'] - (1.5 * df['tr_ma'])
        
        # Squeeze ON/OFF
        df['squeeze_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
        df['squeeze_off'] = (df['bb_lower'] < df['kc_lower']) & (df['bb_upper'] > df['kc_upper'])
        
        # Momentum (simplificado)
        df['momentum'] = df['close'] - df['close'].rolling(10).mean()
        
        # Retorno futuro
        df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        
        # Patrón 1: Squeeze OFF + Momentum positivo
        squeeze_fire_long = df['squeeze_off'] & (df['momentum'] > 0)
        squeeze_fire_long_total = squeeze_fire_long.sum()
        
        if squeeze_fire_long_total >= self.min_occurrences:
            wins = (squeeze_fire_long & (df['future_return'] > 0)).sum()
            avg_profit = df.loc[squeeze_fire_long & (df['future_return'] > 0), 'future_return'].mean() * 100
            avg_loss = df.loc[squeeze_fire_long & (df['future_return'] <= 0), 'future_return'].mean() * 100
            
            results.append(PatternResult(
                condition="Squeeze OFF + Momentum POSITIVO → LARGO",
                win_rate=wins / squeeze_fire_long_total,
                avg_profit=avg_profit if not pd.isna(avg_profit) else 0,
                avg_loss=avg_loss if not pd.isna(avg_loss) else 0,
                profit_factor=abs(avg_profit / avg_loss) if (not pd.isna(avg_loss) and avg_loss != 0) else 0,
                occurrences=squeeze_fire_long_total,
                confidence=min(squeeze_fire_long_total / 30, 1.0)
            ))
        
        # Patrón 2: Squeeze OFF + Momentum negativo
        squeeze_fire_short = df['squeeze_off'] & (df['momentum'] < 0)
        squeeze_fire_short_total = squeeze_fire_short.sum()
        
        if squeeze_fire_short_total >= self.min_occurrences:
            wins = (squeeze_fire_short & (df['future_return'] < 0)).sum()
            avg_profit = abs(df.loc[squeeze_fire_short & (df['future_return'] < 0), 'future_return'].mean() * 100)
            avg_loss = abs(df.loc[squeeze_fire_short & (df['future_return'] >= 0), 'future_return'].mean() * 100)
            
            results.append(PatternResult(
                condition="Squeeze OFF + Momentum NEGATIVO → CORTO",
                win_rate=wins / squeeze_fire_short_total,
                avg_profit=avg_profit if not pd.isna(avg_profit) else 0,
                avg_loss=avg_loss if not pd.isna(avg_loss) else 0,
                profit_factor=abs(avg_profit / avg_loss) if (not pd.isna(avg_loss) and avg_loss != 0) else 0,
                occurrences=squeeze_fire_short_total,
                confidence=min(squeeze_fire_short_total / 30, 1.0)
            ))
        
        return results
    
    def analyze_multitimeframe_patterns(self, df_5min: pd.DataFrame,
                                       forward_periods: int = 20) -> List[PatternResult]:
        """
        Analiza cómo las tendencias en 15m y 1h afectan operaciones en 5m
        """
        results = []
        
        # Crear timeframes superiores
        df_15min = df_5min.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        df_1h = df_5min.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Calcular tendencias
        for tf_df, name in [(df_15min, '15min'), (df_1h, '1h')]:
            tf_df[f'ema_20_{name}'] = talib.EMA(np.asarray(tf_df['close']), timeperiod=20)
            tf_df[f'ema_50_{name}'] = talib.EMA(np.asarray(tf_df['close']), timeperiod=50)
            tf_df[f'trend_{name}'] = np.where(tf_df[f'ema_20_{name}'] > tf_df[f'ema_50_{name}'], 1,
                                             np.where(tf_df[f'ema_20_{name}'] < tf_df[f'ema_50_{name}'], -1, 0))
        
        # Alinear con 5min
        df_5min['trend_15min'] = df_15min['trend_15min'].reindex(df_5min.index, method='ffill')
        df_5min['trend_1h'] = df_1h['trend_1h'].reindex(df_5min.index, method='ffill')
        
        # Retorno futuro
        df_5min['future_return'] = df_5min['close'].shift(-forward_periods) / df_5min['close'] - 1
        
        # Patrón 1: Todas las tendencias alineadas alcistas
        all_bullish = (df_5min['trend_15min'] == 1) & (df_5min['trend_1h'] == 1)
        all_bullish_total = all_bullish.sum()
        
        if all_bullish_total >= self.min_occurrences:
            wins = (all_bullish & (df_5min['future_return'] > 0)).sum()
            avg_profit = df_5min.loc[all_bullish & (df_5min['future_return'] > 0), 'future_return'].mean() * 100
            avg_loss = df_5min.loc[all_bullish & (df_5min['future_return'] <= 0), 'future_return'].mean() * 100
            
            results.append(PatternResult(
                condition="15min ALCISTA + 1h ALCISTA → Operar LARGO en 5min",
                win_rate=wins / all_bullish_total,
                avg_profit=avg_profit if not pd.isna(avg_profit) else 0,
                avg_loss=avg_loss if not pd.isna(avg_loss) else 0,
                profit_factor=abs(avg_profit / avg_loss) if (not pd.isna(avg_loss) and avg_loss != 0) else 0,
                occurrences=all_bullish_total,
                confidence=min(all_bullish_total / 50, 1.0)
            ))
        
        # Patrón 2: Todas las tendencias alineadas bajistas
        all_bearish = (df_5min['trend_15min'] == -1) & (df_5min['trend_1h'] == -1)
        all_bearish_total = all_bearish.sum()
        
        if all_bearish_total >= self.min_occurrences:
            wins = (all_bearish & (df_5min['future_return'] < 0)).sum()
            avg_profit = abs(df_5min.loc[all_bearish & (df_5min['future_return'] < 0), 'future_return'].mean() * 100)
            avg_loss = abs(df_5min.loc[all_bearish & (df_5min['future_return'] >= 0), 'future_return'].mean() * 100)
            
            results.append(PatternResult(
                condition="15min BAJISTA + 1h BAJISTA → Operar CORTO en 5min",
                win_rate=wins / all_bearish_total,
                avg_profit=avg_profit if not pd.isna(avg_profit) else 0,
                avg_loss=avg_loss if not pd.isna(avg_loss) else 0,
                profit_factor=abs(avg_profit / avg_loss) if (not pd.isna(avg_loss) and avg_loss != 0) else 0,
                occurrences=all_bearish_total,
                confidence=min(all_bearish_total / 50, 1.0)
            ))
        
        # Patrón 3: Divergencia (operar contra tendencia puede ser arriesgado)
        divergent = (df_5min['trend_15min'] != df_5min['trend_1h'])
        divergent_total = divergent.sum()
        
        if divergent_total >= self.min_occurrences:
            wins_either = (divergent & (abs(df_5min['future_return']) > 0.01)).sum()
            
            results.append(PatternResult(
                condition="15min vs 1h DIVERGENTES → EVITAR operar (sin confirmación)",
                win_rate=wins_either / divergent_total if divergent_total > 0 else 0,
                avg_profit=0,  # No hay dirección clara
                avg_loss=0,
                profit_factor=0,
                occurrences=divergent_total,
                confidence=min(divergent_total / 50, 1.0)
            ))
        
        return results
    
    def run_full_analysis(self, df: pd.DataFrame) -> Dict[str, List[PatternResult]]:
        """
        Ejecuta todos los análisis de patrones
        """
        print("Ejecutando análisis completo de patrones predictivos...")
        print("=" * 70)
        
        all_results = {}
        
        # 1. Análisis de EMAs
        print("\n1. Analizando proximidad a EMAs (22, 55, 200)...")
        ema_results = self.analyze_ema_proximity_patterns(df.copy())
        all_results['ema_proximity'] = ema_results
        print(f"   Encontrados {len(ema_results)} patrones relacionados con EMAs")
        
        # 2. Análisis de Volumen y POC
        print("\n2. Analizando volumen y POC...")
        volume_results = self.analyze_volume_poc_patterns(df.copy())
        all_results['volume_poc'] = volume_results
        print(f"   Encontrados {len(volume_results)} patrones de volumen/POC")
        
        # 3. Análisis de IFVG
        print("\n3. Analizando IFVG (Imbalance/Fair Value Gaps)...")
        ifvg_results = self.analyze_ifvg_patterns(df.copy())
        all_results['ifvg'] = ifvg_results
        print(f"   Encontrados {len(ifvg_results)} patrones IFVG")
        
        # 4. Análisis de Squeeze Momentum
        print("\n4. Analizando Squeeze Momentum...")
        squeeze_results = self.analyze_squeeze_momentum_patterns(df.copy())
        all_results['squeeze_momentum'] = squeeze_results
        print(f"   Encontrados {len(squeeze_results)} patrones Squeeze")
        
        # 5. Análisis Multi-timeframe
        print("\n5. Analizando confirmación multi-timeframe (15m, 1h en 5m)...")
        mtf_results = self.analyze_multitimeframe_patterns(df.copy())
        all_results['multitimeframe'] = mtf_results
        print(f"   Encontrados {len(mtf_results)} patrones multi-timeframe")
        
        return all_results
    
    def generate_pattern_report(self, all_results: Dict[str, List[PatternResult]],
                               min_profit_factor: float = 1.5,
                               min_win_rate: float = 0.55) -> str:
        """
        Genera reporte con los mejores patrones encontrados
        """
        report = []
        report.append("# 📊 REPORTE DE PATRONES PREDICTIVOS")
        report.append(f"Generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("## 🎯 Objetivo")
        report.append("Identificar qué condiciones/patrones tienen mayor probabilidad de éxito")
        report.append(f"Criterios: Win Rate > {min_win_rate*100:.0f}% y Profit Factor > {min_profit_factor}")
        report.append("")
        
        # Recolectar TODOS los patrones
        all_patterns = []
        for category, patterns in all_results.items():
            for pattern in patterns:
                all_patterns.append((category, pattern))
        
        # Filtrar mejores patrones
        best_patterns = [
            (cat, pat) for cat, pat in all_patterns 
            if pat.win_rate >= min_win_rate and pat.profit_factor >= min_profit_factor
        ]
        
        # Ordenar por profit factor
        best_patterns.sort(key=lambda x: x[1].profit_factor, reverse=True)
        
        report.append("## 🏆 TOP PATRONES MÁS RENTABLES")
        report.append("")
        report.append("| # | Patrón | Win Rate | Profit Factor | Avg Profit | Casos | Confianza |")
        report.append("|---|--------|----------|---------------|------------|-------|-----------|")
        
        for i, (category, pattern) in enumerate(best_patterns[:15], 1):
            confidence_stars = "⭐" * int(pattern.confidence * 5)
            report.append(
                f"| {i} | {pattern.condition[:60]} | "
                f"{pattern.win_rate:.1%} | {pattern.profit_factor:.2f}x | "
                f"{pattern.avg_profit:.2f}% | {pattern.occurrences} | {confidence_stars} |"
            )
        
        if not best_patterns:
            report.append("| - | No se encontraron patrones que cumplan los criterios | - | - | - | - | - |")
        
        report.append("")
        
        # Análisis por categoría
        report.append("## 📈 ANÁLISIS POR CATEGORÍA")
        report.append("")
        
        categories = {
            'ema_proximity': '🎯 Proximidad a EMAs',
            'volume_poc': '📊 Volumen y POC',
            'ifvg': '⚡ IFVG (Fair Value Gaps)',
            'squeeze_momentum': '💥 Squeeze Momentum',
            'multitimeframe': '🔄 Multi-Timeframe'
        }
        
        for cat_key, cat_name in categories.items():
            if cat_key in all_results:
                report.append(f"### {cat_name}")
                report.append("")
                
                patterns = all_results[cat_key]
                if patterns:
                    # Top 3 de esta categoría
                    sorted_patterns = sorted(patterns, key=lambda x: x.profit_factor, reverse=True)[:3]
                    
                    for i, pat in enumerate(sorted_patterns, 1):
                        report.append(f"**{i}. {pat.condition}**")
                        report.append(f"   - Win Rate: {pat.win_rate:.1%}")
                        report.append(f"   - Profit Factor: {pat.profit_factor:.2f}x")
                        report.append(f"   - Avg Profit/Loss: {pat.avg_profit:.2f}% / {pat.avg_loss:.2f}%")
                        report.append(f"   - Ocurrencias: {pat.occurrences}")
                        report.append(f"   - Confianza: {pat.confidence:.0%}")
                        report.append("")
                else:
                    report.append("   No se encontraron patrones suficientes")
                    report.append("")
        
        # Recomendaciones
        report.append("## 💡 RECOMENDACIONES")
        report.append("")
        
        if best_patterns:
            top_3 = best_patterns[:3]
            report.append("### Patrones a Priorizar:")
            for i, (cat, pat) in enumerate(top_3, 1):
                report.append(f"{i}. **{pat.condition}**")
                report.append(f"   → Usar cuando este patrón se detecte (PF: {pat.profit_factor:.2f}x)")
            report.append("")
        
        report.append("### Cómo Usar Este Reporte:")
        report.append("1. **Enfócate en los patrones con mayor Profit Factor** (>2.0 es excelente)")
        report.append("2. **Valida la confianza**: Más ⭐ = más datos = más confiable")
        report.append("3. **Combina patrones**: Busca cuando varios patrones se alinean")
        report.append("4. **Evita patrones con baja confianza** (<50%) hasta tener más datos")
        report.append("")
        
        # Mapa de decisiones
        report.append("## 🗺️ MAPA DE DECISIONES")
        report.append("")
        report.append("```")
        report.append("ENTRADA LARGO:")
        long_patterns = [p for _, p in best_patterns if 'LARGO' in p.condition or 'ALCISTA' in p.condition or 'REBOTE ALCISTA' in p.condition]
        if long_patterns:
            for pat in long_patterns[:3]:
                report.append(f"  ✓ {pat.condition[:70]}")
        else:
            report.append("  ✗ No se encontraron patrones alcistas sólidos")
        
        report.append("")
        report.append("ENTRADA CORTO:")
        short_patterns = [p for _, p in best_patterns if 'CORTO' in p.condition or 'BAJISTA' in p.condition or 'RECHAZO BAJISTA' in p.condition]
        if short_patterns:
            for pat in short_patterns[:3]:
                report.append(f"  ✓ {pat.condition[:70]}")
        else:
            report.append("  ✗ No se encontraron patrones bajistas sólidos")
        
        report.append("")
        report.append("EVITAR OPERAR:")
        avoid_patterns = [p for _, p in best_patterns if 'EVITAR' in p.condition or 'DIVERGENTE' in p.condition]
        if avoid_patterns:
            for pat in avoid_patterns[:3]:
                report.append(f"  ✗ {pat.condition[:70]}")
        else:
            report.append("  • Cuando ningún patrón de alta confianza está presente")
        report.append("```")
        report.append("")
        
        return "\n".join(report)


def run_pattern_discovery(df: pd.DataFrame, 
                         min_occurrences: int = 10,
                         output_file: str = None):
    """
    Ejecuta análisis completo de descubrimiento de patrones
    """
    analyzer = PatternDiscoveryAnalyzer(min_occurrences=min_occurrences)
    
    # Ejecutar análisis
    all_results = analyzer.run_full_analysis(df)
    
    # Generar reporte
    report = analyzer.generate_pattern_report(all_results)
    
    # Guardar resultados
    if output_file is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"pattern_discovery_report_{timestamp}.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)
    print(f"\nReporte guardado en: {output_file}")
    
    return all_results, report


if __name__ == "__main__":
    print("Pattern Discovery Analyzer")
    print("Ejecuta run_pattern_discovery(df) con datos de mercado 5min")
