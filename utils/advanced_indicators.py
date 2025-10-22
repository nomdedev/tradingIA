import pandas as pd
import numpy as np

def calculate_squeeze_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el indicador Squeeze Momentum
    Combina Bollinger Bands y Keltner Channels para detectar compresión de volatilidad
    """
    df = df.copy()

    # Calcular squeeze: BB inferior > KC superior O BB superior < KC inferior
    df['squeeze_on'] = ((df['BB_lower'] > df['KC_upper']) | (df['BB_upper'] < df['KC_lower'])).astype(int)

    # Calcular momentum durante squeeze
    df['squeeze_momentum'] = 0.0

    # Encontrar periodos de squeeze
    squeeze_periods = df[df['squeeze_on'] == 1]

    for idx in squeeze_periods.index:
        # Calcular momentum como cambio porcentual durante squeeze
        if idx > 0:
            momentum = (df.loc[idx, 'Close'] - df.loc[idx-1, 'Close']) / df.loc[idx-1, 'Close'] * 100
            df.loc[idx, 'squeeze_momentum'] = momentum

    # Calcular delta del momentum (aceleración)
    df['squeeze_momentum_delta'] = df['squeeze_momentum'].diff()

    return df

def detect_ifvg(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Detecta Implied Fair Value Gaps (IFVG)
    Gaps que aparecen después de movimientos fuertes y actúan como zonas de soporte/resistencia
    """
    df = df.copy()

    df['ifvg_bullish_nearby'] = False
    df['ifvg_bearish_nearby'] = False

    for i in range(len(df)):
        current_price = df.iloc[i]['Close']

        # Buscar gaps bullish (precio actual cerca del gap)
        bullish_gaps = []
        bearish_gaps = []

        # Revisar velas anteriores en la ventana lookback
        start_idx = max(0, i - lookback)
        for j in range(start_idx, i):
            high = df.iloc[j]['High']
            low = df.iloc[j]['Low']
            close = df.iloc[j]['Close']
            open_price = df.iloc[j]['Open']

            # Detectar gap bullish (precio abre por encima del máximo anterior)
            if j > 0:
                prev_high = df.iloc[j-1]['High']
                if open_price > prev_high:
                    gap_level = (prev_high + open_price) / 2  # Nivel medio del gap
                    bullish_gaps.append(gap_level)

            # Detectar gap bearish (precio abre por debajo del mínimo anterior)
            if j > 0:
                prev_low = df.iloc[j-1]['Low']
                if open_price < prev_low:
                    gap_level = (prev_low + open_price) / 2  # Nivel medio del gap
                    bearish_gaps.append(gap_level)

        # Verificar si el precio actual está cerca de algún gap
        tolerance = current_price * 0.005  # 0.5% de tolerancia

        for gap in bullish_gaps:
            if abs(current_price - gap) <= tolerance:
                df.iloc[i, df.columns.get_loc('ifvg_bullish_nearby')] = True
                break

        for gap in bearish_gaps:
            if abs(current_price - gap) <= tolerance:
                df.iloc[i, df.columns.get_loc('ifvg_bearish_nearby')] = True
                break

    return df

def calculate_institutional_emas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula EMAs institucionales (22, 55, 200)
    Usadas por instituciones para identificar tendencias a largo plazo
    """
    df = df.copy()

    # Calcular EMAs
    df['EMA_22'] = df['Close'].ewm(span=22, adjust=False).mean()
    df['EMA_55'] = df['Close'].ewm(span=55, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Calcular pendientes (slope) para detectar dirección de tendencia
    df['EMA_22_slope'] = df['EMA_22'].diff()
    df['EMA_55_slope'] = df['EMA_55'].diff()
    df['EMA_200_slope'] = df['EMA_200'].diff()

    # Determinar alineación EMA
    # 1 = bullish (EMA22 > EMA55 > EMA200)
    # -1 = bearish (EMA22 < EMA55 < EMA200)
    # 0 = neutral/mixed
    df['ema_alignment'] = 0
    bullish_mask = (df['EMA_22'] > df['EMA_55']) & (df['EMA_55'] > df['EMA_200'])
    bearish_mask = (df['EMA_22'] < df['EMA_55']) & (df['EMA_55'] < df['EMA_200'])

    df.loc[bullish_mask, 'ema_alignment'] = 1
    df.loc[bearish_mask, 'ema_alignment'] = -1

    # Filtros de precio vs EMA
    df['price_above_ema200'] = df['Close'] > df['EMA_200']
    df['price_above_ema55'] = df['Close'] > df['EMA_55']
    df['price_above_ema22'] = df['Close'] > df['EMA_22']

    return df

def add_all_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade todos los indicadores avanzados al DataFrame
    """
    print("🧮 Calculando indicadores avanzados...")

    # Squeeze Momentum
    df = calculate_squeeze_momentum(df)
    squeeze_count = df['squeeze_on'].sum()
    print(f"✓ Squeeze Momentum calculado: {squeeze_count} periodos de squeeze detectados")

    # IFVG Detection
    df = detect_ifvg(df)
    bullish_ifvg = df['ifvg_bullish_nearby'].sum()
    bearish_ifvg = df['ifvg_bearish_nearby'].sum()
    print(f"✓ IFVG detectados: {bullish_ifvg} bullish, {bearish_ifvg} bearish")

    # EMAs Institucionales
    df = calculate_institutional_emas(df)
    bullish_emas = (df['ema_alignment'] == 1).sum()
    bearish_emas = (df['ema_alignment'] == -1).sum()
    neutral_emas = (df['ema_alignment'] == 0).sum()
    print(f"✓ EMAs institucionales calculadas: {bullish_emas} bullish, {bearish_emas} bearish, {neutral_emas} neutral")

    # Resumen final
    new_columns = [
        'squeeze_on', 'squeeze_momentum', 'squeeze_momentum_delta',
        'ifvg_bullish_nearby', 'ifvg_bearish_nearby',
        'EMA_22', 'EMA_55', 'EMA_200', 'EMA_22_slope', 'EMA_55_slope', 'EMA_200_slope',
        'ema_alignment', 'price_above_ema200', 'price_above_ema55', 'price_above_ema22'
    ]

    print(f"✓ Todos los indicadores avanzados agregados. Columnas finales: {len(df.columns)}")
    print(f"   Nuevas columnas: {', '.join(new_columns)}")

    return df