# 🧪 ESTRATEGIA DE TESTING - 8 Áreas Críticas

**Creado:** 12 de Enero 2026  
**Versión:** 1.0  
**Objetivo:** Asegurar que cada fix se valida correctamente

---

## 📋 RESUMEN EJECUTIVO

Cada una de las 8 áreas críticas requiere una estrategia de testing específica:
- ✅ Unit Tests: Validar función individual
- ✅ Integration Tests: Validar interacción con otras funciones
- ✅ Regression Tests: Verificar que no rompemos nada
- ✅ Comparison Tests: Antes vs Después de implementación

**Total de tests a crear:** ~40 tests nuevos  
**Cobertura target:** >90% de código crítico

---

## 🚨 ÁREA 1: Look-Ahead Bias

### Problema
Volume Profile y otros indicadores calculan valores futuros con datos que no están disponibles en el momento actual.

### Estrategia de Testing

#### 1.1 Unit Test: No Look-Ahead in Volume Profile
```python
def test_volume_profile_advanced_no_look_ahead():
    """
    Validar que volume_profile_advanced() solo usa datos pasados
    """
    # Arrange
    df = create_synthetic_ohlc(periods=100)
    lookback = 20
    
    # Act
    vpc = volume_profile_advanced(df, lookback)
    
    # Assert
    for i in range(lookback, len(df)):
        # En tiempo i, el VPC debe usar solo datos [i-lookback:i]
        expected_price = df.iloc[i-lookback:i]['close'].mean()
        assert vpc.iloc[i] == expected_price  # Aproximadamente
        assert not any(df.iloc[i:i+5]['close'] in vpc.iloc[i])  # No hay futuros
```

**Status:** ⬜ Por crear  
**ETA:** 13 Enero  
**Archivo:** `tests/test_indicators_no_look_ahead.py`

---

#### 1.2 Unit Test: Filtering Signals No Future Data
```python
def test_generate_filtered_signals_no_future():
    """
    Validar que generate_filtered_signals() no incluye datos futuros
    """
    df = create_synthetic_ohlc(periods=100)
    
    # Ejecutar en punto i
    signals_at_i = generate_filtered_signals(df, lookback_point=50)
    
    # La señal en posición 50 no debe incluir datos de posiciones 51+
    assert signals_at_i['ema_fast'] == df.iloc[:50]['close'].ewm(span=5).mean().iloc[-1]
    assert signals_at_i['ema_slow'] == df.iloc[:50]['close'].ewm(span=20).mean().iloc[-1]
    # NOT using df[51:] data
```

**Status:** ⬜ Por crear  
**ETA:** 14 Enero  
**Archivo:** `tests/test_signals_no_look_ahead.py`

---

#### 1.3 Integration Test: Backtester Respects Data Alignment
```python
def test_backtester_data_alignment():
    """
    Validar que el backtester nunca usa datos futuros en decisiones
    """
    backtest = SimpleBacktest()
    df = load_historical_data('BTC', '2023-01-01', '2023-12-31')
    
    # Simular decisiones históricos
    for date in df.index:
        # En fecha X, solo deberían existir datos hasta fecha X
        available_data = df[:date]  # Inclusive
        
        # Generar señal con solo datos disponibles
        signal = strategy.generate_signal(available_data)
        
        # Verificar: signal no usa información de df[date+1:]
        assert not _uses_future_data(signal, available_data)
```

**Status:** ⬜ Por crear  
**ETA:** 14 Enero  
**Archivo:** `tests/test_backtest_data_alignment.py`

---

#### 1.4 Comparison Test: Before vs After Sharpe Ratio
```python
def test_look_ahead_bias_sharpe_impact():
    """
    Validar que arreglar look-ahead bias reduce Sharpe ratio inflado
    """
    df = load_historical_data('BTC', '2023-01-01', '2023-12-31')
    
    # Backtest CON bug (versión anterior)
    backtest_old = SimpleBacktest(use_future_data=True)  # Bug simulado
    result_old = backtest_old.run(df, strategy=VP_IFVG_Strategy())
    sharpe_old = result_old['sharpe_ratio']  # Inflado ej: 2.5
    
    # Backtest SIN bug (versión fija)
    backtest_new = SimpleBacktest(use_future_data=False)  # Fix
    result_new = backtest_new.run(df, strategy=VP_IFVG_Strategy())
    sharpe_new = result_new['sharpe_ratio']  # Realista ej: 1.2
    
    # Assert: Sharpe debería caer 30-40%
    assert sharpe_old > sharpe_new * 1.3
    assert sharpe_new > 0.8  # Aún rentable pero más realista
    
    # Documentar
    print(f"Sharpe inflation: {(sharpe_old/sharpe_new - 1)*100:.1f}%")
```

**Status:** ⬜ Por crear  
**ETA:** 14 Enero  
**Archivo:** `tests/test_look_ahead_impact.py`

---

#### 1.5 Regression Test: All Other Signals Still Work
```python
def test_no_regression_other_indicators():
    """
    Después de arreglar look-ahead bias, otros indicadores funcionan igual
    """
    df = create_synthetic_ohlc(periods=100)
    
    # EMA debe ser idéntica
    ema_old = df['close'].ewm(span=20).mean()
    ema_new = calculate_ema(df)
    assert (ema_old - ema_new).abs().max() < 0.001
    
    # ATR sin cambios
    atr_old = calculate_atr(df, period=14)
    atr_new = calculate_atr_fixed(df, period=14)
    pd.testing.assert_series_equal(atr_old, atr_new)
```

**Status:** ⬜ Por crear  
**ETA:** 14 Enero  
**Archivo:** `tests/test_no_regression_area1.py`

---

## 🚨 ÁREA 2: Walk-Forward Analysis

### Problema
WFA no optimiza parámetros en cada período, simplemente valida con los mismos parámetros.

### Estrategia de Testing

#### 2.1 Unit Test: Parameters Change Between Periods
```python
def test_wfa_parameters_change_each_period():
    """
    Validar que WFA optimiza parámetros diferentes para cada período
    """
    backtest = AdvancedBacktest()
    
    # Configurar WFA con 5 períodos
    config = WFAConfig(
        n_periods=5,
        param_ranges={'ema_fast': [5, 10, 15], 'ema_slow': [20, 30, 40]}
    )
    
    results = backtest.run_walk_forward(df, config)
    
    # Verificar: Parámetros diferentes entre períodos
    period_params = results['period_parameters']
    assert period_params[0] != period_params[1]  # P1 != P2
    assert period_params[1] != period_params[2]  # P2 != P3
    
    # Verificar: Parámetros están dentro de rangos
    for params in period_params.values():
        assert 5 <= params['ema_fast'] <= 15
        assert 20 <= params['ema_slow'] <= 40
```

**Status:** ⬜ Por crear  
**ETA:** 21 Enero  
**Archivo:** `tests/test_wfa_parameters.py`

---

#### 2.2 Unit Test: Degradation Calculation
```python
def test_wfa_degradation_calculation():
    """
    Validar que degradación OOS/IS se calcula correctamente
    """
    results = {
        'period_0': {'sharpe_is': 2.0, 'sharpe_oos': 1.2},  # Degrada
        'period_1': {'sharpe_is': 1.8, 'sharpe_oos': 1.5},  # Degrada
        'period_2': {'sharpe_is': 1.9, 'sharpe_oos': 1.4},  # Degrada
    }
    
    # Calcular degradación
    degradation = []
    for period, metrics in results.items():
        deg = 1 - (metrics['sharpe_oos'] / metrics['sharpe_is'])
        degradation.append(deg)
    
    # Assert
    expected = [0.4, 0.167, 0.263]  # (1 - OOS/IS)
    assert np.allclose(degradation, expected)
    
    # Stability score (1 - avg degradation)
    stability_score = 1 - np.mean(degradation)
    assert 0 < stability_score < 1
    assert stability_score > 0.6  # Considerado estable
```

**Status:** ⬜ Por crear  
**ETA:** 22 Enero  
**Archivo:** `tests/test_wfa_degradation.py`

---

#### 2.3 Integration Test: Optimization Improves OOS Results
```python
def test_wfa_optimization_improves_oos():
    """
    Validar que optimización produce mejores resultados OOS
    """
    # Sin optimización (random params)
    results_no_opt = run_wfa_without_optimization(df)
    
    # Con optimización (Bayesian)
    results_with_opt = run_wfa_with_optimization(df)
    
    # OOS performance debe mejorar
    assert results_with_opt['avg_sharpe_oos'] > results_no_opt['avg_sharpe_oos']
```

**Status:** ⬜ Por crear  
**ETA:** 22 Enero  
**Archivo:** `tests/test_wfa_optimization_impact.py`

---

## 🚨 ÁREA 3: Kelly Criterion Dinámico

### Problema
Kelly se calcula con lookback fijo (50 trades) sin considerar régimen de mercado.

### Estrategia de Testing

#### 3.1 Unit Test: Regime Detection Works
```python
def test_kelly_regime_detection():
    """
    Validar que se detecta correctamente bull/bear/sideways
    """
    # Bull market: SMA100 > SMA200, trending up
    df_bull = create_trending_data(trend='up', length=500)
    regime_bull = detect_market_regime(df_bull)
    assert regime_bull in ['bull', 'strong_bull']
    
    # Bear market
    df_bear = create_trending_data(trend='down', length=500)
    regime_bear = detect_market_regime(df_bear)
    assert regime_bear in ['bear', 'strong_bear']
    
    # Sideways
    df_sideways = create_ranging_data(length=500)
    regime_sideways = detect_market_regime(df_sideways)
    assert regime_sideways == 'sideways'
```

**Status:** ⬜ Por crear  
**ETA:** 25 Enero  
**Archivo:** `tests/test_regime_detection.py`

---

#### 3.2 Unit Test: Regime-Adjusted Kelly
```python
def test_kelly_regime_adjustment():
    """
    Validar que Kelly se ajusta correctamente por régimen
    """
    # Parámetros base: win_rate=55%, W/L=1.5
    base_kelly = calculate_kelly_fraction(0.55, 1.5)  # ~7.5%
    
    # Aplicar factores por régimen
    kelly_bull = base_kelly * KELLY_FACTORS['bull']       # 1.0x
    kelly_sideways = base_kelly * KELLY_FACTORS['sideways'] # 0.7x
    kelly_bear = base_kelly * KELLY_FACTORS['bear']       # 0.5x
    
    # Validar ordenamiento
    assert kelly_bull > kelly_sideways > kelly_bear
    assert 0 < kelly_bear < kelly_bull
```

**Status:** ⬜ Por crear  
**ETA:** 25 Enero  
**Archivo:** `tests/test_kelly_regime_adjustment.py`

---

#### 3.3 Unit Test: Serial Correlation Penalty
```python
def test_kelly_serial_correlation_penalty():
    """
    Validar que Kelly se reduce con rachas de ganancias
    """
    # Sin correlación serial: 3 trades ganadores consecutivos
    trade_sequence = [
        {'result': 1, 'pnl': 100},
        {'result': 1, 'pnl': 150},
        {'result': 1, 'pnl': 80},
    ]
    
    # Kelly normal
    kelly_normal = 7.5
    
    # Con serial correlation penalty
    kelly_adjusted = calculate_kelly_with_serial_correlation(
        kelly_base=kelly_normal,
        consecutive_wins=3,
        penalty_per_consecutive=0.015
    )
    # 7.5% - (3 * 1.5%) = 3%
    
    assert kelly_adjusted < kelly_normal
    assert kelly_adjusted == pytest.approx(3.0, rel=0.1)
```

**Status:** ⬜ Por crear  
**ETA:** 26 Enero  
**Archivo:** `tests/test_kelly_serial_correlation.py`

---

## 🚨 ÁREA 4: Council Integration

### Problema
Council existe pero nunca se consulta en decisiones de trading durante backtesting.

### Estrategia de Testing

#### 4.1 Unit Test: Council Decision Approval
```python
def test_council_decision_approval():
    """
    Validar que Council aprueba trades cuando condiciones son favorables
    """
    council = Council()
    council.register_standard_experts()
    
    # Contexto favorable: Sin drawdown, tendencia alcista
    context = {
        'signal': {'action': 'BUY', 'strength': 0.8},
        'current_equity': 10000,
        'peak_equity': 10000,
        'current_dd': 0.0,  # Sin drawdown
        'market_regime': 'bull',
        'volatility': 0.02,
    }
    
    decision = council.decide(context)
    
    # Assert
    assert decision['approve'] == True
    assert decision['score'] > 0.6  # Consensus positivo
```

**Status:** ⬜ Por crear  
**ETA:** 16 Enero  
**Archivo:** `tests/test_council_approval.py`

---

#### 4.2 Unit Test: Council Veto
```python
def test_council_veto_high_drawdown():
    """
    Validar que Council rechaza trades en alto drawdown
    """
    council = Council()
    council.register_standard_experts()
    
    # Contexto desfavorable: Alto drawdown
    context = {
        'signal': {'action': 'BUY', 'strength': 0.9},  # Incluso señal fuerte
        'current_equity': 8000,
        'peak_equity': 10000,
        'current_dd': 0.20,  # 20% drawdown
        'market_regime': 'bear',
        'volatility': 0.05,
    }
    
    decision = council.decide(context)
    
    # Assert
    assert decision['approve'] == False
    assert 'Risk Warden' in decision['veto_reasons']
```

**Status:** ⬜ Por crear  
**ETA:** 16 Enero  
**Archivo:** `tests/test_council_veto.py`

---

#### 4.3 Integration Test: Backtester Consults Council
```python
def test_backtester_consults_council_integration():
    """
    Validar que backtester consulta Council antes de cada trade
    """
    backtest = SimpleBacktest(use_council=True)
    
    # Mock Council para verificar que es consultado
    with patch.object(council, 'decide') as mock_decide:
        mock_decide.return_value = {'approve': False}
        
        result = backtest.run(df, strategy=test_strategy)
        
        # Assert
        assert mock_decide.called
        assert mock_decide.call_count >= 5  # Al menos 5 veces
        
        # Verificar que trades fueron rechazados
        assert result['total_trades'] < expected_trades_without_council
```

**Status:** ⬜ Por crear  
**ETA:** 17 Enero  
**Archivo:** `tests/test_council_integration_backtester.py`

---

#### 4.4 Comparison Test: Impact on P&L
```python
def test_council_impact_on_pnl():
    """
    Validar que Council reduce trades pero mejora P&L
    """
    # Backtest SIN Council
    result_no_council = run_backtest(df, use_council=False)
    
    # Backtest CON Council
    result_with_council = run_backtest(df, use_council=True)
    
    # Aserciones
    assert result_with_council['total_trades'] < result_no_council['total_trades']
    assert result_with_council['win_rate'] > result_no_council['win_rate']
    # P&L total podría ser similar o mejor gracias a mejor win rate
    assert result_with_council['sharpe_ratio'] > result_no_council['sharpe_ratio']
```

**Status:** ⬜ Por crear  
**ETA:** 17 Enero  
**Archivo:** `tests/test_council_pnl_impact.py`

---

## 🚨 ÁREA 7: Data Validation Pipeline

### Problema
DataValidator existe pero nunca se llama automáticamente. Datos corruptos pueden pasar.

### Estrategia de Testing

#### 7.1 Unit Test: OHLC Validation
```python
def test_validation_ohlc_relationships():
    """
    Validar que se detectan OHLC inválidos
    """
    validator = DataValidator()
    
    # Caso 1: High < Low (inválido)
    invalid_df = pd.DataFrame({
        'open': [100, 101],
        'high': [102, 101],  # High < Low en row 1
        'low': [103, 100],
        'close': [101, 101]
    })
    
    with pytest.raises(DataValidationError):
        validator.validate_ohlc_relationships(invalid_df)
    
    # Caso 2: Close > High (inválido)
    invalid_df2 = pd.DataFrame({
        'open': [100],
        'high': [102],
        'low': [99],
        'close': [105]  # Cierre > High
    })
    
    with pytest.raises(DataValidationError):
        validator.validate_ohlc_relationships(invalid_df2)
```

**Status:** ⬜ Por crear  
**ETA:** 18 Enero  
**Archivo:** `tests/test_data_validator_ohlc.py`

---

#### 7.2 Unit Test: Time Gap Detection
```python
def test_validation_time_gaps():
    """
    Validar que se detectan gaps en tiempo
    """
    validator = DataValidator()
    
    # Datos con gap (falta una vela)
    dates = pd.date_range('2024-01-01', periods=10, freq='H')
    dates_with_gap = dates.delete(5)  # Quita hora 5
    
    df_with_gap = pd.DataFrame({
        'open': np.random.rand(9),
        'high': np.random.rand(9),
        'low': np.random.rand(9),
        'close': np.random.rand(9),
    }, index=dates_with_gap)
    
    warnings = validator.detect_time_gaps(df_with_gap, expected_freq='H')
    
    # Assert
    assert len(warnings) > 0
    assert 'missing' in warnings[0].lower()
```

**Status:** ⬜ Por crear  
**ETA:** 18 Enero  
**Archivo:** `tests/test_data_validator_gaps.py`

---

#### 7.3 Unit Test: Look-Ahead Bias Detection
```python
def test_validation_look_ahead_bias_detection():
    """
    Validar que se detecta uso de datos futuros
    """
    validator = DataValidator()
    
    # Dataset que usa datos futuros
    df = create_synthetic_ohlc(periods=100)
    
    # Función con look-ahead bias
    def indicator_with_bias(data):
        return data['close'].rolling(window=5, center=True).mean()
    
    # Validar
    has_bias = validator.detect_look_ahead_bias(df, indicator_with_bias)
    
    # Assert
    assert has_bias == True
    
    # Función sin bias
    def indicator_no_bias(data):
        return data['close'].rolling(window=5).mean()
    
    has_bias = validator.detect_look_ahead_bias(df, indicator_no_bias)
    assert has_bias == False
```

**Status:** ⬜ Por crear  
**ETA:** 18 Enero  
**Archivo:** `tests/test_data_validator_bias.py`

---

#### 7.4 Integration Test: Validation Pipeline Mandatory
```python
def test_data_loading_validation_mandatory():
    """
    Validar que validación es OBLIGATORIA en carga de datos
    """
    # Crear datos corruptos
    corrupted_data = pd.DataFrame({
        'open': [100, 101],
        'high': [102, 101],  # High < Low
        'low': [103, 100],
        'close': [101, 101]
    })
    
    # Intentar cargar sin validación debería fallar
    with pytest.raises(DataValidationError):
        data_manager = DataManager()
        data_manager.load_data(corrupted_data, validate=True, skip_on_error=False)
    
    # Con validate=False debería pasar (pero log warning)
    with pytest.warns(UserWarning):
        data = data_manager.load_data(corrupted_data, validate=False)
        # Datos se cargan pero con warning
```

**Status:** ⬜ Por crear  
**ETA:** 19 Enero  
**Archivo:** `tests/test_data_loading_mandatory_validation.py`

---

#### 7.5 Unit Test: Auto-Fix Capabilities
```python
def test_data_auto_fix_duplicates():
    """
    Validar que se pueden auto-arreglar duplicados
    """
    validator = DataValidator()
    
    # DataFrame con duplicados
    df_duplicates = pd.DataFrame({
        'open': [100, 100, 101],
        'high': [102, 102, 103],
        'low': [99, 99, 100],
        'close': [101, 101, 102],
    }, index=pd.date_range('2024-01-01', periods=3, freq='H'))
    
    # Auto-fix
    df_fixed = validator.auto_fix_duplicates(df_duplicates)
    
    # Assert: Duplicado removido
    assert len(df_fixed) == 2
    assert not df_fixed.index.duplicated().any()
```

**Status:** ⬜ Por crear  
**ETA:** 19 Enero  
**Archivo:** `tests/test_data_auto_fix.py`

---

## 🚨 ÁREA 5: Market Impact Crypto

### Problema
Usa modelo equity Almgren-Chriss, no apto para crypto 24/7.

### Estrategia de Testing

#### 5.1 Unit Test: Crypto Hourly Liquidity
```python
def test_market_impact_hourly_liquidity():
    """
    Validar que impact varía por hora del día
    """
    model = MarketImpactModelCrypto()
    
    # Peak hours (13-15 UTC): Liquidez máxima (1.0)
    impact_peak = model.calculate_impact(
        order_size=100000,
        market_cap=1000000000,
        hour_utc=14
    )
    
    # Low hours (3-5 UTC): Liquidez mínima (0.15)
    impact_low = model.calculate_impact(
        order_size=100000,
        market_cap=1000000000,
        hour_utc=4
    )
    
    # Assert
    assert impact_low > impact_peak * 2  # Mucho más impact
    assert impact_peak < 0.01  # < 0.1% en peak
    assert impact_low < 0.05   # < 0.5% en low
```

**Status:** ⬜ Por crear  
**ETA:** 28 Enero  
**Archivo:** `tests/test_market_impact_hourly.py`

---

#### 5.2 Unit Test: Buy/Sell Asymmetry
```python
def test_market_impact_buy_sell_asymmetry():
    """
    Validar que venta tiene más impact que compra
    """
    model = MarketImpactModelCrypto()
    
    order_size = 100000
    market_cap = 1000000000
    
    impact_buy = model.calculate_impact(
        order_size=order_size,
        side='BUY',
        market_cap=market_cap
    )
    
    impact_sell = model.calculate_impact(
        order_size=order_size,
        side='SELL',
        market_cap=market_cap
    )
    
    # Assert: Venta 30% más slippage
    assert impact_sell > impact_buy
    assert impact_sell / impact_buy == pytest.approx(1.3, rel=0.1)
```

**Status:** ⬜ Por crear  
**ETA:** 28 Enero  
**Archivo:** `tests/test_market_impact_asymmetry.py`

---

## 🚨 ÁREA 6: Risk Manager

### Problema
Solo verifica daily DD, no total DD. Sin correlación de posiciones.

### Estrategia de Testing

#### 6.1 Unit Test: Total Drawdown Tracking
```python
def test_risk_manager_total_drawdown():
    """
    Validar que se trackea máximo drawdown desde pico
    """
    risk_mgr = RiskManager(max_total_dd=0.20)
    
    # Simulación: Peak en día 1, luego declive
    equity_history = [10000, 9800, 9500, 9200, 9000]
    
    for i, equity in enumerate(equity_history):
        risk_mgr.update_state(current_equity=equity)
        
        if i == 4:  # En día 5
            total_dd = risk_mgr.get_total_drawdown()
            expected_dd = (10000 - 9000) / 10000
            assert total_dd == pytest.approx(expected_dd)
```

**Status:** ⬜ Por crear  
**ETA:** 31 Enero  
**Archivo:** `tests/test_risk_manager_total_dd.py`

---

#### 6.2 Unit Test: Correlated Risk Calculation
```python
def test_risk_manager_correlated_risk():
    """
    Validar que correlación de posiciones se calcula correctamente
    """
    risk_mgr = RiskManager()
    
    # 3 posiciones con correlación alta (0.85)
    positions = {
        'BTC': {'size': 10000, 'volatility': 0.04},
        'ETH': {'size': 5000, 'volatility': 0.05},
        'SOL': {'size': 3000, 'volatility': 0.06},
    }
    
    correlation_matrix = np.array([
        [1.0, 0.85, 0.80],
        [0.85, 1.0, 0.75],
        [0.80, 0.75, 1.0],
    ])
    
    correlated_risk = risk_mgr.calculate_correlated_risk(
        positions, correlation_matrix
    )
    
    # Risk correlacionado > risk simple (suma)
    simple_risk = sum(p['size'] * p['volatility'] for p in positions.values())
    assert correlated_risk > simple_risk  # No es suma simple
```

**Status:** ⬜ Por crear  
**ETA:** 31 Enero  
**Archivo:** `tests/test_risk_manager_correlation.py`

---

## 🚨 ÁREA 8: Signal Standardization

### Problema
3 formatos diferentes para señales (FVGData, Series boolean, BaseStrategy).

### Estrategia de Testing

#### 8.1 Unit Test: Signal Format Consistency
```python
def test_trading_signal_format():
    """
    Validar que todas las estrategias retornan TradingSignal
    """
    signal = TradingSignal(
        timestamp=pd.Timestamp.now(),
        action='BUY',
        strength=0.8,
        price=45000.0,
        strategy_name='VP_IFVG_EMA',
        metadata={'reason': 'FVG filled'}
    )
    
    # Assert: Campos requeridos presentes
    assert signal.action in ['BUY', 'SELL', 'HOLD']
    assert 0 <= signal.strength <= 1.0
    assert signal.timestamp is not None
    assert signal.strategy_name is not None
```

**Status:** ⬜ Por crear  
**ETA:** 7 Febrero  
**Archivo:** `tests/test_trading_signal_format.py`

---

#### 8.2 Integration Test: All Strategies Conform
```python
def test_all_strategies_return_trading_signal():
    """
    Validar que TODAS las estrategias usan TradingSignal
    """
    strategies = [
        VP_IFVG_EMA_Strategy(),
        BaseStrategy(),
        SimpleMovingAverage(),
        # ... todas las estrategias
    ]
    
    df = create_synthetic_ohlc(periods=50)
    
    for strategy in strategies:
        signals = strategy.generate_signals(df)
        
        # Validar formato
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert all(hasattr(signal, field) for field in REQUIRED_FIELDS)
```

**Status:** ⬜ Por crear  
**ETA:** 7 Febrero  
**Archivo:** `tests/test_all_strategies_conform.py`

---

---

## 📊 RESUMEN DE TESTING

| Área | Tests Unit | Tests Integration | Tests Regression | Total | ETA |
|------|-----------|-------------------|-----------------|-------|-----|
| 1 | 3 | 1 | 1 | 5 | 14 Ene |
| 2 | 3 | 1 | 0 | 4 | 22 Ene |
| 3 | 3 | 0 | 0 | 3 | 26 Ene |
| 4 | 2 | 2 | 1 | 5 | 17 Ene |
| 5 | 2 | 0 | 0 | 2 | 28 Ene |
| 6 | 2 | 0 | 0 | 2 | 1 Feb |
| 7 | 5 | 1 | 0 | 6 | 19 Ene |
| 8 | 1 | 1 | 0 | 2 | 7 Feb |
| **TOTAL** | **21** | **6** | **2** | **29** | **7 Feb** |

---

## 🎯 COBERTURA TARGET

```
Unit Tests:        >90% cobertura de funciones críticas
Integration Tests: Todos los puntos de integración
Regression Tests:  Verificar que sistemas 3rd party siguen funcionando
Comparison Tests:  Before vs After para cada área

TOTAL TARGET: 29+ tests, >90% cobertura crítica
```

---

**Último actualizado:** 12 de Enero 2026
