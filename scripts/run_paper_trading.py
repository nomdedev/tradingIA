#!/usr/bin/env python3
"""
Paper Trading Runner
Ejecuta el sistema de trading en modo simulado (paper trading)
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import json

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.indicators import TechnicalIndicators
from strategies.ifvg_strategy import IFVGStrategy
from src.backtest_engine import BacktestEngine

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PaperTradingRunner:
    """Ejecutor de paper trading"""

    def __init__(self, config_path='config/app_config.json'):
        self.config = self.load_config(config_path)
        self.data_fetcher = DataFetcher()
        self.indicators = TechnicalIndicators()
        self.strategy = None
        self.backtest_engine = BacktestEngine()

    def load_config(self, config_path):
        """Cargar configuración"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                "symbol": "BTCUSD",
                "timeframe": "1H",
                "initial_capital": 10000,
                "risk_per_trade": 0.02,
                "max_open_trades": 3,
                "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "end_date": datetime.now().strftime("%Y-%m-%d")
            }

    def initialize_strategy(self):
        """Inicializar estrategia"""
        try:
            self.strategy = IFVGStrategy(
                atr_period=self.config.get('atr_period', 200),
                atr_multiplier=self.config.get('atr_multiplier', 0.25),
                risk_reward_ratio=self.config.get('risk_reward_ratio', 2.0)
            )
            logger.info("Strategy initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize strategy: {e}")
            raise

    def run_paper_trading(self):
        """Ejecutar paper trading"""
        logger.info("Starting paper trading session")
        logger.info(f"Config: {self.config}")

        try:
            # Cargar datos
            logger.info(f"Fetching data for {self.config['symbol']} ({self.config['timeframe']})")
            data = self.data_fetcher.fetch_crypto_data(
                symbol=self.config['symbol'],
                timeframe=self.config['timeframe'],
                start_date=self.config['start_date'],
                end_date=self.config['end_date']
            )

            if data.empty:
                logger.error("No data available for paper trading")
                return

            logger.info(f"Data loaded: {len(data)} bars")

            # Calcular indicadores
            logger.info("Calculating indicators...")
            data_with_indicators = self.indicators.calculate_all_indicators(data)

            # Inicializar estrategia
            self.initialize_strategy()

            # Ejecutar backtest (paper trading simulado)
            logger.info("Running paper trading simulation...")
            results = self.backtest_engine.run_backtest(
                strategy=self.strategy,
                data=data_with_indicators,
                initial_capital=self.config['initial_capital'],
                risk_per_trade=self.config['risk_per_trade'],
                max_open_trades=self.config['max_open_trades']
            )

            # Mostrar resultados
            self.display_results(results)

            # Guardar resultados
            self.save_results(results)

        except Exception as e:
            logger.error(f"Paper trading failed: {e}")
            raise

    def display_results(self, results):
        """Mostrar resultados"""
        print("\n" + "="*50)
        print("PAPER TRADING RESULTS")
        print("="*50)
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(".1f")
        print(".1f")
        print(".2f")
        print("="*50)

    def save_results(self, results):
        """Guardar resultados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/paper_trading_{timestamp}.json"

        results['timestamp'] = datetime.now().isoformat()
        results['config'] = self.config

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {filename}")

def main():
    """Función principal"""
    print("Paper Trading Runner")
    print("===================")

    runner = PaperTradingRunner()
    runner.run_paper_trading()

if __name__ == "__main__":
    main()