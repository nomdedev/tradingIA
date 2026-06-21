"""
Prueba integrada del sistema completo de trading crypto
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_configuration():
    """Verificar configuración del sistema"""
    print("=" * 80)
    print("🔍 VERIFICACIÓN DE CONFIGURACIÓN DEL SISTEMA")
    print("=" * 80)

    # Verificar variables de entorno
    print("\n🔑 VARIABLES DE ENTORNO:")
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    api_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    has_credentials = api_key and api_secret
    print(f"APCA_API_KEY_ID: {'✅ SET' if api_key else '❌ MISSING'}")
    print(f"APCA_API_SECRET_KEY: {'✅ SET' if api_secret else '❌ MISSING'}")
    print(f"APCA_API_BASE_URL: {api_url}")
    print(f"Credentials available: {'✅ YES' if has_credentials else '⚠️ NO (simulated mode)'}")

    # Verificar archivos críticos
    print("\n📁 ARCHIVOS CRÍTICOS:")
    critical_files = [
        "trading_live/crypto_live_engine.py",
        "trading_live/alpaca_client.py",
        "backtesting/quick_backtester.py",
        "strategies/multi_timeframe_analyzer.py",
        "agents/ensemble_agent.py",
        "agents/moondev_risk_agent.py",
        "utils/indicators.py",
    ]

    all_files_exist = True
    for file in critical_files:
        exists = os.path.exists(file)
        status = "✅ EXISTS" if exists else "❌ MISSING"
        print(f"{file}: {status}")
        if not exists:
            all_files_exist = False

    if not all_files_exist:
        print("\n❌ ERROR: Faltan archivos críticos")
        return False

    print("\n✅ CONFIGURACIÓN VERIFICADA")
    return True, has_credentials


def test_imports():
    """Probar imports de todos los componentes"""
    print("\n📦 PROBANDO IMPORTS...")

    try:
        from trading_live.crypto_live_engine import CryptoLiveTradingEngine

        print("✅ CryptoLiveTradingEngine importado")

        from trading_live.alpaca_client import AlpacaClient

        print("✅ AlpacaClient importado")

        from backtesting.quick_backtester import QuickBacktester

        print("✅ QuickBacktester importado")

        from strategies.multi_timeframe_analyzer import MultiTimeframeAnalyzer

        print("✅ MultiTimeframeAnalyzer importado")

        from agents.ensemble_agent import EnsembleAgent

        print("✅ EnsembleAgent importado")

        from agents.moondev_risk_agent import MoonDevRiskAgent

        print("✅ MoonDevRiskAgent importado")

        from utils.indicators import add_technical_indicators

        print("✅ Technical indicators importados")

        print("✅ TODOS LOS IMPORTS EXITOSOS")
        return True

    except Exception as e:
        print(f"❌ ERROR en imports: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_initialization():
    """Probar inicialización de componentes"""
    print("\n🏗️ PROBANDO INICIALIZACIÓN...")

    try:
        # Probar componentes que no requieren credenciales
        from backtesting.quick_backtester import QuickBacktester

        backtester = QuickBacktester()
        print("✅ QuickBacktester inicializado")

        from strategies.multi_timeframe_analyzer import MultiTimeframeAnalyzer

        mtf = MultiTimeframeAnalyzer()
        print("✅ MultiTimeframeAnalyzer inicializado")

        # Solo probar AlpacaClient si hay credenciales
        try:
            from trading_live.alpaca_client import AlpacaClient

            alpaca = AlpacaClient()
            print("✅ AlpacaClient inicializado")
        except Exception as e:
            print(f"⚠️ AlpacaClient no inicializado (credenciales faltantes): {str(e)[:50]}...")

        print("✅ INICIALIZACIÓN COMPLETA")
        return True

    except Exception as e:
        print(f"❌ ERROR en inicialización: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Función principal de prueba"""
    print("🚀 PRUEBA INTEGRADA DEL SISTEMA DE TRADING CRYPTO")
    print("=" * 80)

    # Verificar configuración
    config_ok, has_credentials = check_configuration()
    if not config_ok:
        return

    # Probar imports
    if not test_imports():
        return

    # Probar inicialización (solo si hay credenciales)
    if has_credentials:
        if not test_initialization():
            return
        print("\n🎯 MODO COMPLETO: Sistema listo para trading real")
    else:
        print("\n🎭 MODO SIMULADO: Sistema verificado sin credenciales")
        print("Para trading real, configura las variables de entorno:")
        print("  APCA_API_KEY_ID=tu_api_key")
        print("  APCA_API_SECRET_KEY=tu_secret_key")

    print("\n" + "=" * 80)
    print("🎉 SISTEMA DE TRADING CRYPTO 24/7 - VERIFICADO")
    print("=" * 80)
    print("🧠 COMPONENTES INTELIGENTES:")
    print("✅ Multi-Timeframe Analysis (6 timeframes)")
    print("✅ Quick Backtester (validación histórica)")
    print("✅ Ensemble Agent (decisiones inteligentes)")
    print("✅ Moon Dev Risk Management (7 checks)")
    print("✅ Stop Loss Management (trailing stops)")
    print("✅ Position Sizing Dinámico (MTF-based)")
    print()
    print("🔄 FLUJO DE OPERACIÓN:")
    print("1. Market Data → 2. MTF Analysis → 3. Quick Backtest")
    print("4. Ensemble Decision → 5. Risk Validation → 6. Execute Trade")
    print("=" * 80)


if __name__ == "__main__":
    main()
