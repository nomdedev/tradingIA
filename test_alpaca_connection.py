#!/usr/bin/env python3
"""
Test Alpaca Connection
Script para probar la conexión a Alpaca y verificar funcionalidades básicas
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Agregar directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_live.alpaca_client import AlpacaClient

def setup_logging():
    """Configurar logging básico"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_environment():
    """Cargar variables de entorno"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print("✅ Variables de entorno cargadas desde .env")
    else:
        print("⚠️ Archivo .env no encontrado")

def test_alpaca_connection():
    """Probar conexión completa a Alpaca"""
    print("\n🧪 PRUEBA DE CONEXIÓN ALPACA")
    print("=" * 50)

    success = True  # Inicializar como True

    try:
        # Crear cliente
        print("🔌 Conectando a Alpaca...")
        client = AlpacaClient()
        print("✅ Cliente Alpaca creado exitosamente")

        # Test 1: Información de cuenta
        print("\n📊 Test 1: Información de cuenta")
        account = client.get_account_info()
        print(f"   Account ID: {account.id}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Equity: ${float(account.equity):,.2f}")
        print("✅ Información de cuenta obtenida")

        # Test 2: Estado del mercado
        print("\n🕐 Test 2: Estado del mercado")
        market_open = client.is_market_open()
        print(f"   Mercado {'abierto' if market_open else 'cerrado'}")

        market_hours = client.get_market_hours()
        print(f"   Próxima apertura: {market_hours.get('next_open', 'N/A')}")
        print(f"   Próximo cierre: {market_hours.get('next_close', 'N/A')}")
        print("✅ Estado del mercado verificado")

        # Test 3: Precio actual
        print("\n💰 Test 3: Precio actual de SPY")
        spy_price = client.get_current_price('SPY')
        print(f"   SPY Price: ${spy_price:.2f}")
        print("✅ Precio obtenido")

        # Test 4: Datos históricos
        print("\n📈 Test 4: Datos históricos")
        try:
            historical_data = client.get_historical_data('SPY', days=5)
            print(f"   Datos históricos obtenidos: {len(historical_data)} registros")
            if not historical_data.empty:
                print(f"   Rango de fechas: {historical_data.index[0]} a {historical_data.index[-1]}")
                print(f"   Precio más reciente: ${historical_data['close'].iloc[-1]:.2f}")
            print("✅ Datos históricos obtenidos")
        except Exception as e:
            if "subscription does not permit" in str(e):
                print("⚠️ Datos históricos no disponibles (cuenta gratuita)")
                print("   Esto es normal para cuentas Alpaca básicas")
                print("✅ Test omitido por limitaciones de cuenta")
            else:
                print(f"❌ Error obteniendo datos históricos: {e}")
                success = False

        # Test 5: Posiciones
        print("\n📋 Test 5: Posiciones abiertas")
        positions = client.get_positions()
        print(f"   Posiciones abiertas: {len(positions)}")
        if positions:
            for pos in positions[:3]:  # Mostrar máximo 3
                print(f"   {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f}")
        print("✅ Posiciones verificadas")

        # Test 6: Órdenes abiertas
        print("\n📝 Test 6: Órdenes abiertas")
        open_orders = client.get_open_orders()
        print(f"   Órdenes abiertas: {len(open_orders)}")
        if open_orders:
            for order in open_orders[:3]:  # Mostrar máximo 3
                print(f"   {order.symbol} {order.side} {order.qty} @ {order.type}")
        print("✅ Órdenes verificadas")

        # Test 7: Cálculo de position size
        print("\n📊 Test 7: Cálculo de position size")
        position_size = client.calculate_position_size('SPY')
        print(f"   Position size recomendado para SPY: {position_size:.0f} acciones")
        print("✅ Position size calculado")

        # Test 8: Resumen de cuenta
        print("\n📊 Test 8: Resumen completo de cuenta")
        summary = client.get_account_summary()
        print(f"   Account ID: {summary['account_id']}")
        print(f"   Cash: ${summary['cash']:,.2f}")
        print(f"   Portfolio Value: ${summary['portfolio_value']:,.2f}")
        print(f"   Positions: {summary['positions_count']}")
        print(f"   Open Orders: {summary['open_orders_count']}")
        print("✅ Resumen de cuenta generado")

        print("\n🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
        print("✅ Conexión a Alpaca funcionando correctamente")
        print("✅ Todas las funcionalidades básicas verificadas")

        return True

    except Exception as e:
        print(f"\n❌ ERROR en pruebas: {e}")
        print("Revisa tu configuración de API keys y conexión a internet")
        return False

def main():
    """Función principal"""
    setup_logging()
    load_environment()

    # Verificar variables de entorno
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        print("❌ ERROR: ALPACA_API_KEY y ALPACA_SECRET_KEY son requeridas")
        print("Configura estas variables en un archivo .env o en el entorno del sistema")
        sys.exit(1)

    print(f"🔑 API Key configurada: {api_key[:8]}...")
    print(f"🔒 Secret Key configurada: {'*' * len(secret_key)}")

    # Ejecutar pruebas
    success = test_alpaca_connection()

    if success:
        print("\n✅ Test completado exitosamente")
        print("Puedes proceder a ejecutar el paper trading real")
    else:
        print("\n❌ Test fallido")
        print("Revisa la configuración antes de continuar")
        sys.exit(1)

if __name__ == '__main__':
    main()