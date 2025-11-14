"""
Demo del Sistema de Configuración Avanzado
===========================================

Demuestra todas las capacidades del nuevo sistema:
1. Gestión de estrategias y presets
2. Configuración de múltiples APIs
3. Integración con agentes IA
4. Sistema de seguridad y encriptación
"""

import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from advanced_config_manager import AdvancedConfigManager
from ai_agent_integrator import AIAgentIntegrator
import json

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")

def demo_api_management():
    """Demostrar gestión de APIs"""
    print_section("1. GESTIÓN DE APIs")
    
    config = AdvancedConfigManager()
    
    # Listar APIs disponibles
    print("📡 APIs Disponibles:")
    apis = config.list_available_apis()
    for api in apis:
        status = "✅ Activa" if api['enabled'] else "⚪ Inactiva"
        default = " (Por Defecto)" if api['is_default'] else ""
        creds = " 🔑" if api['has_credentials'] else " ⚠️ Sin credenciales"
        print(f"  {status} {api['name']}{default}{creds}")
        print(f"     URL: {api['base_url']}")
    
    # Mostrar API por defecto
    default_api = config.get_default_api()
    print(f"\n🎯 API Por Defecto: {default_api}")
    
    # Ejemplo de configuración de credenciales
    print("\n💡 Configuración de Credenciales:")
    print("   config.set_api_credentials('alpaca', 'API_KEY', 'SECRET_KEY')")
    print("   config.set_api_credentials('binance', 'API_KEY', 'SECRET_KEY')")
    print("   config.set_default_api('binance')")

def demo_strategy_presets():
    """Demostrar gestión de presets de estrategias"""
    print_section("2. GESTIÓN DE PRESETS DE ESTRATEGIAS")
    
    config = AdvancedConfigManager()
    
    # Guardar un preset de ejemplo
    strategy = "MACD_ADX"
    preset_name = "Aggressive_Momentum"
    params = {
        "macd_fast": 8,
        "macd_slow": 21,
        "adx_period": 10
    }
    
    print(f"💾 Guardando preset '{preset_name}' para estrategia '{strategy}'...")
    success = config.save_strategy_preset(strategy, preset_name, params)
    print(f"   {'✅ Guardado exitoso' if success else '❌ Error al guardar'}")
    
    # Listar presets disponibles
    print(f"\n📋 Presets disponibles para '{strategy}':")
    presets = config.list_strategy_presets(strategy)
    for preset in presets:
        print(f"   - {preset}")
    
    # Cargar un preset
    if presets:
        print(f"\n📖 Cargando preset '{presets[0]}'...")
        loaded = config.load_strategy_preset(strategy, presets[0])
        if loaded:
            print(f"   Parámetros: {json.dumps(loaded['params'], indent=6)}")
    
    print("\n💡 Casos de Uso:")
    print("   - Guardar configuraciones optimizadas")
    print("   - Comparar diferentes configuraciones")
    print("   - Compartir configuraciones entre usuarios")
    print("   - Volver rápidamente a configuraciones anteriores")

def demo_ai_integration():
    """Demostrar integración con agentes IA"""
    print_section("3. INTEGRACIÓN CON AGENTES IA")
    
    config = AdvancedConfigManager()
    
    # Listar agentes disponibles
    print("🤖 Agentes IA Disponibles:")
    agents = config.list_available_agents()
    for agent in agents:
        status = "✅ Configurado" if agent['has_credentials'] else "⚪ Sin configurar"
        enabled = " (Activo)" if agent['enabled'] else ""
        print(f"  {status} {agent['name']}{enabled}")
        print(f"     ID: {agent['id']}")
        print(f"     Capacidades: {', '.join(agent['capabilities'])}")
    
    # Agente activo
    active = config.get_active_agent()
    print(f"\n🎯 Agente Activo: {active if active else 'Ninguno'}")
    
    # Opciones de análisis
    print("\n⚙️ Opciones de Análisis:")
    options = config.get_analysis_options()
    for key, value in options.items():
        status = "✅" if value else "⚪"
        print(f"  {status} {key.replace('_', ' ').title()}")
    
    # Triggers de análisis
    print("\n🔔 Triggers de Análisis Automático:")
    ai_config = config.get_ai_config()
    triggers = ai_config.get('analysis_triggers', {})
    for trigger, enabled in triggers.items():
        status = "✅" if enabled else "⚪"
        print(f"  {status} {trigger.replace('_', ' ').title()}")
    
    print("\n💡 Funcionalidades:")
    print("   - Análisis automático de resultados de backtesting")
    print("   - Validación matemática de estrategias")
    print("   - Sugerencias de optimización de parámetros")
    print("   - Comparación de múltiples estrategias")
    print("   - Revisión de código de estrategias")

def demo_ai_analysis():
    """Demostrar análisis con IA"""
    print_section("4. ANÁLISIS CON AGENTE IA (Demo)")
    
    config = AdvancedConfigManager()
    ai_agent = AIAgentIntegrator(config)
    
    # Verificar si está habilitado
    if not ai_agent.is_enabled():
        print("⚠️  Análisis IA deshabilitado en configuración")
        print("\n💡 Para habilitar:")
        print("   1. Configurar API key de un agente (Claude, ChatGPT, etc.)")
        print("   2. Activar el agente en la configuración")
        print("   3. Habilitar vscode_integration en app_config.json")
    else:
        print("✅ Análisis IA habilitado")
        active_agent = config.get_active_agent()
        print(f"   Agente activo: {active_agent}")
    
    # Ejemplo de análisis de backtest
    print("\n📊 Ejemplo de Análisis de Backtest:")
    print("""
   strategy = "MACD_ADX"
   params = {"macd_fast": 12, "macd_slow": 26, "adx_period": 14}
   metrics = {
       "total_return": 0.25,
       "sharpe_ratio": 1.8,
       "max_drawdown": 0.15,
       "win_rate": 0.58
   }
   
   result = ai_agent.analyze_backtest_results(
       strategy, params, metrics, trades
   )
   """)
    
    print("\n💬 El agente IA proporcionará:")
    print("   ✓ Evaluación del rendimiento ajustado por riesgo")
    print("   ✓ Análisis del win rate y profit factor")
    print("   ✓ Evaluación del drawdown y recuperación")
    print("   ✓ Sugerencias específicas de mejora")
    print("   ✓ Recomendaciones de parámetros")
    print("   ✓ Análisis de distribución de trades")

def demo_security_features():
    """Demostrar características de seguridad"""
    print_section("5. CARACTERÍSTICAS DE SEGURIDAD")
    
    config = AdvancedConfigManager()
    security = config.config.get('security_settings', {})
    
    print("🔒 Configuración de Seguridad:")
    print(f"   Encriptar credenciales: {'✅ Sí' if security.get('encrypt_credentials') else '❌ No'}")
    print(f"   Auto-logout: {security.get('auto_logout_minutes', 0)} minutos")
    print(f"   Confirmar trades: {'✅ Sí' if security.get('require_confirmation_trades') else '❌ No'}")
    print(f"   Límite posición: {security.get('max_position_size_pct', 0)}%")
    print(f"   Límite pérdida diaria: {security.get('daily_loss_limit_pct', 0)}%")
    
    print("\n🔐 Protección de Credenciales:")
    print("   - API keys encriptadas con Fernet (AES)")
    print("   - Clave de encriptación en archivo seguro (.encryption_key)")
    print("   - Variables de entorno como fuente alternativa")
    print("   - Sin credenciales en código fuente")

def demo_testing_config():
    """Demostrar configuración de testing"""
    print_section("6. CONFIGURACIÓN DE TESTING")
    
    config = AdvancedConfigManager()
    testing = config.get_testing_settings()
    
    print("🧪 Configuración de Testing:")
    for key, value in testing.items():
        status = "✅" if value else "⚪"
        if isinstance(value, bool):
            print(f"  {status} {key.replace('_', ' ').title()}")
        else:
            print(f"  📁 {key.replace('_', ' ').title()}: {value}")
    
    print("\n💡 Capacidades:")
    print("   - Tests unitarios automáticos")
    print("   - Tests de integración")
    print("   - Mock de llamadas a API")
    print("   - Coverage tracking")
    print("   - Validación de datos de prueba")

def demo_performance_settings():
    """Demostrar configuración de rendimiento"""
    print_section("7. OPTIMIZACIÓN DE RENDIMIENTO")
    
    config = AdvancedConfigManager()
    perf = config.get_performance_settings()
    
    print("⚡ Configuración de Rendimiento:")
    for key, value in perf.items():
        if isinstance(value, bool):
            status = "✅" if value else "⚪"
            print(f"  {status} {key.replace('_', ' ').title()}")
        else:
            print(f"  📊 {key.replace('_', ' ').title()}: {value}")
    
    print("\n💡 Optimizaciones:")
    print("   - Cache de indicadores técnicos")
    print("   - Procesamiento paralelo (multiprocessing)")
    print("   - Optimización de DataFrames")
    print("   - Límites de memoria configurables")

def demo_full_workflow():
    """Demostrar flujo de trabajo completo"""
    print_section("8. FLUJO DE TRABAJO COMPLETO")
    
    print("📋 Flujo Típico de Uso:")
    print("\n1️⃣  CONFIGURACIÓN INICIAL")
    print("   - Configurar credenciales de API (Alpaca, Binance, etc.)")
    print("   - Seleccionar API por defecto")
    print("   - Configurar agente IA (opcional)")
    
    print("\n2️⃣  DESARROLLO DE ESTRATEGIA")
    print("   - Cargar datos históricos")
    print("   - Seleccionar estrategia")
    print("   - Ajustar parámetros con sliders")
    print("   - Guardar preset personalizado")
    
    print("\n3️⃣  BACKTESTING")
    print("   - Ejecutar backtest (simple/walk-forward/monte-carlo)")
    print("   - Visualizar resultados")
    print("   - Análisis automático con IA (si está habilitado)")
    
    print("\n4️⃣  OPTIMIZACIÓN")
    print("   - Revisar sugerencias del agente IA")
    print("   - Ajustar parámetros según análisis")
    print("   - Guardar nuevos presets")
    print("   - Comparar múltiples configuraciones")
    
    print("\n5️⃣  COMPARACIÓN")
    print("   - Cargar múltiples presets")
    print("   - Ejecutar backtests en paralelo")
    print("   - Análisis comparativo con IA")
    print("   - Seleccionar mejor configuración")
    
    print("\n6️⃣  EXPORTACIÓN")
    print("   - Exportar a PDF/CSV/JSON")
    print("   - Generar Pine Script")
    print("   - Guardar gráficos")
    print("   - Documentar decisiones")

def demo_validation():
    """Validar configuración"""
    print_section("9. VALIDACIÓN DE CONFIGURACIÓN")
    
    config = AdvancedConfigManager()
    
    print("🔍 Validando configuración...")
    is_valid, errors = config.validate_config()
    
    if is_valid:
        print("✅ Configuración válida")
    else:
        print("❌ Errores encontrados:")
        for error in errors:
            print(f"   - {error}")
    
    print("\n📊 Resumen:")
    print(f"   APIs configuradas: {len(config.list_available_apis())}")
    print(f"   Agentes IA disponibles: {len(config.list_available_agents())}")
    default_api = config.get_default_api()
    print(f"   API por defecto: {default_api if default_api else 'No configurada'}")
    active_agent = config.get_active_agent()
    print(f"   Agente IA activo: {active_agent if active_agent else 'Ninguno'}")

def main():
    """Ejecutar demostración completa"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     BTC Trading Platform - Sistema de Configuración         ║
    ║              Avanzado - Demostración Completa               ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        demo_api_management()
        demo_strategy_presets()
        demo_ai_integration()
        demo_ai_analysis()
        demo_security_features()
        demo_testing_config()
        demo_performance_settings()
        demo_full_workflow()
        demo_validation()
        
        print_section("✨ DEMOSTRACIÓN COMPLETADA")
        print("""
✅ El sistema incluye:
   • Gestión de múltiples APIs (Alpaca, Binance, Coinbase, Polygon)
   • Sistema de presets para estrategias
   • Integración con agentes IA (Copilot, Claude, ChatGPT, Custom)
   • Análisis automático con IA
   • Seguridad y encriptación de credenciales
   • Configuración completa de testing
   • Optimización de rendimiento
   • Validación de configuración

📖 Para usar en tu código:
   
   from advanced_config_manager import AdvancedConfigManager
   from ai_agent_integrator import AIAgentIntegrator
   
   # Inicializar
   config = AdvancedConfigManager()
   ai_agent = AIAgentIntegrator(config)
   
   # Configurar API
   config.set_api_credentials('alpaca', 'key', 'secret')
   
   # Guardar preset
   config.save_strategy_preset('MACD_ADX', 'my_preset', params)
   
   # Analizar con IA
   result = ai_agent.analyze_backtest_results(...)

🚀 ¡Todo listo para usar!
        """)
        
    except Exception as e:
        print(f"\n❌ Error en demostración: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
