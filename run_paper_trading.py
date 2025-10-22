#!/usr/bin/env python3
"""
Paper Trading Runner
Script principal para ejecutar trading en vivo con Alpaca paper trading
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Agregar directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_live.alpaca_client import AlpacaClient
from trading_live.live_engine import LiveTradingEngine

def setup_logging(log_level: str = 'INFO'):
    """Configurar logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/paper_trading.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reducir logging de bibliotecas externas
    logging.getLogger('alpaca_trade_api').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def load_environment():
    """Cargar variables de entorno"""
    # Cargar .env si existe
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logging.info("✅ Variables de entorno cargadas desde .env")
    else:
        logging.warning("⚠️ Archivo .env no encontrado, usando variables de entorno del sistema")

def validate_environment():
    """Validar que todas las variables necesarias estén configuradas"""
    required_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY'
    ]

    optional_vars = [
        'ALPACA_BASE_URL',
        'GROQ_API_KEY',
        'ANTHROPIC_API_KEY',
        'XAI_API_KEY',
        'DEEPSEEK_API_KEY'
    ]

    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)

    if missing_required:
        logging.error(f"❌ Variables requeridas faltantes: {', '.join(missing_required)}")
        logging.error("Configure estas variables en un archivo .env o en el entorno del sistema")
        return False

    # Loggear estado de variables opcionales
    for var in optional_vars:
        if os.getenv(var):
            logging.info(f"✅ {var} configurada")
        else:
            logging.info(f"⚠️ {var} no configurada (opcional)")

    return True

def create_default_config():
    """Crear configuración por defecto"""
    return {
        'symbols': ['SPY'],  # Símbolos a tradear
        'check_interval': 60,  # Segundos entre checks
        'max_positions': 5,    # Máximo número de posiciones
        'risk_per_trade': 0.02,  # 2% riesgo por trade
        'max_drawdown': 0.05,   # 5% max drawdown
        'min_trade_amount': 100, # Monto mínimo por trade
        'agents': {
            'rl': {'enabled': True, 'weight': 0.5},
            'ga': {'enabled': True, 'weight': 0.5},
            'llm': {'enabled': False, 'weight': 0.0}  # Deshabilitado por ahora
        }
    }

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Paper Trading con Alpaca')
    parser.add_argument('--config', type=str, help='Archivo de configuración JSON')
    parser.add_argument('--symbols', nargs='+', default=['SPY'], help='Símbolos a tradear')
    parser.add_argument('--interval', type=int, default=60, help='Intervalo entre checks (segundos)')
    parser.add_argument('--risk', type=float, default=0.02, help='Riesgo por trade (0.02 = 2%)')
    parser.add_argument('--max-drawdown', type=float, default=0.05, help='Max drawdown (0.05 = 5%)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--dry-run', action='store_true', help='Modo simulación (no ejecutar trades)')
    parser.add_argument('--test-connection', action='store_true', help='Solo probar conexión y salir')

    args = parser.parse_args()

    # Configurar logging
    setup_logging(args.log_level)

    logging.info("🚀 Iniciando Paper Trading System")
    logging.info(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Cargar y validar entorno
    load_environment()
    if not validate_environment():
        sys.exit(1)

    try:
        # Crear cliente Alpaca
        alpaca_client = AlpacaClient()
        logging.info("✅ Alpaca client inicializado")

        # Si solo queremos probar conexión
        if args.test_connection:
            logging.info("🧪 Probando conexión a Alpaca...")

            # Obtener información de cuenta
            account = alpaca_client.get_account_info()
            logging.info(f"💰 Cash: ${account.cash}")
            logging.info(f"📊 Portfolio Value: ${account.portfolio_value}")
            logging.info(f"💳 Buying Power: ${account.buying_power}")

            # Verificar estado del mercado
            market_open = alpaca_client.is_market_open()
            logging.info(f"🕐 Mercado {'abierto' if market_open else 'cerrado'}")

            # Obtener precio de SPY
            try:
                spy_price = alpaca_client.get_current_price('SPY')
                logging.info(f"💰 SPY Price: ${spy_price:.2f}")
            except Exception as e:
                logging.warning(f"⚠️ No se pudo obtener precio de SPY: {e}")

            logging.info("✅ Prueba de conexión exitosa")
            return

        # Crear configuración
        config = create_default_config()
        config['symbols'] = args.symbols
        config['check_interval'] = args.interval
        config['risk_per_trade'] = args.risk
        config['max_drawdown'] = args.max_drawdown

        if args.dry_run:
            logging.info("🎭 MODO SIMULACIÓN ACTIVADO - No se ejecutarán trades reales")
            config['dry_run'] = True

        # Crear motor de trading
        engine = LiveTradingEngine(alpaca_client, config)
        logging.info("✅ Live Trading Engine inicializado")

        # Mostrar configuración
        logging.info("⚙️ Configuración:")
        logging.info(f"   Símbolos: {config['symbols']}")
        logging.info(f"   Intervalo: {config['check_interval']}s")
        logging.info(f"   Riesgo por trade: {config['risk_per_trade']:.1%}")
        logging.info(f"   Max drawdown: {config['max_drawdown']:.1%}")
        logging.info(f"   Agentes activos: {[k for k, v in config['agents'].items() if v['enabled']]}")

        # Iniciar trading
        logging.info("🎯 Iniciando trading en vivo...")
        logging.info("Presiona Ctrl+C para detener")

        engine.start_trading()

    except KeyboardInterrupt:
        logging.info("🛑 Interrupción detectada, finalizando...")
    except Exception as e:
        logging.error(f"❌ Error fatal: {e}")
        sys.exit(1)
    finally:
        logging.info("👋 Paper Trading finalizado")

if __name__ == '__main__':
    main()