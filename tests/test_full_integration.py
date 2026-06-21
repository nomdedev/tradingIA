#!/usr/bin/env python3
"""
🔬 TEST COMPLETO DE INTEGRACIÓN - TRADING IA PLATFORM
======================================================

Este script realiza pruebas completas de integración de la plataforma Trading IA,
incluyendo:
- Verificación de imports
- Inicialización de componentes
- Pruebas de backtesting con realistic execution
- Tests de UI (sin mostrar ventana)
- Validación end-to-end

Fecha: 16 de Noviembre, 2025
"""

import sys
import os
import traceback
import time
from datetime import datetime

# Configurar logging
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Agregar directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IntegrationTester:
    """Clase para realizar pruebas completas de integración"""

    def __init__(self):
        self.results = {"imports": {}, "initialization": {}, "backtesting": {}, "ui": {}, "end_to_end": {}}
        self.errors = []
        self.warnings = []

    def log(self, message, level="info"):
        """Log con timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if level == "error":
            logger.error(f"[{timestamp}] ❌ {message}")
        elif level == "warning":
            logger.warning(f"[{timestamp}] ⚠️  {message}")
        elif level == "success":
            logger.info(f"[{timestamp}] ✅ {message}")
        else:
            logger.info(f"[{timestamp}] ℹ️  {message}")

    def add_error(self, test_name, error):
        """Agregar error a la lista"""
        self.errors.append({"test": test_name, "error": str(error), "traceback": traceback.format_exc()})
        self.log(f"ERROR in {test_name}: {error}", "error")

    def add_warning(self, test_name, warning):
        """Agregar warning a la lista"""
        self.warnings.append({"test": test_name, "warning": warning})
        self.log(f"WARNING in {test_name}: {warning}", "warning")

    def test_imports(self):
        """Test 1: Verificar todos los imports críticos"""
        self.log("=== TEST 1: IMPORTS VERIFICATION ===")

        critical_imports = [
            # Core components
            ("PySide6.QtWidgets", "QApplication"),
            ("PySide6.QtCore", "Qt"),
            ("pandas", None),
            ("numpy", None),
            ("plotly", None),
            # Platform components
            ("core.backend_core", "DataManager"),
            ("core.execution.backtester_core", "BacktesterCore"),
            ("src.analysis_engines", "AnalysisEngines"),
            # Realistic execution components
            ("src.execution.market_impact", "MarketImpactModel"),
            ("src.execution.order_manager", "OrderManager"),
            ("src.execution.latency_model", "LatencyModel"),
            # GUI components
            ("src.gui.platform_gui_tab3_improved", "Tab3BacktestRunner"),
            ("src.config.user_config", "UserConfigManager"),
        ]

        for module_name, class_name in critical_imports:
            try:
                module = __import__(module_name, fromlist=[class_name] if class_name else [])
                if class_name:
                    getattr(module, class_name)
                    self.results["imports"][f"{module_name}.{class_name}"] = True
                    self.log(f"Import OK: {module_name}.{class_name}", "success")
                else:
                    # For modules without specific class (pandas, numpy, plotly)
                    self.results["imports"][module_name] = True
                    self.log(f"Import OK: {module_name}", "success")
            except Exception as e:
                key = f"{module_name}.{class_name}" if class_name else module_name
                self.results["imports"][key] = False
                self.add_error("imports", f"Failed to import {key}: {e}")

    def test_initialization(self):
        """Test 2: Inicializar componentes principales"""
        self.log("=== TEST 2: COMPONENT INITIALIZATION ===")

        try:
            # Test BacktesterCore con realistic execution
            from core.execution.backtester_core import BacktesterCore

            # Inicializar sin realistic execution
            backtester_simple = BacktesterCore(initial_capital=10000, commission=0.001, slippage_pct=0.001)
            self.results["initialization"]["backtester_simple"] = True
            self.log("BacktesterCore (simple) initialized", "success")

            # Inicializar con realistic execution
            backtester_realistic = BacktesterCore(
                initial_capital=10000,
                commission=0.001,
                slippage_pct=0.001,
                enable_realistic_execution=True,
                latency_profile="retail_average",
            )
            self.results["initialization"]["backtester_realistic"] = True
            self.log("BacktesterCore (realistic) initialized", "success")

            # Verificar que los componentes estén inicializados
            if hasattr(backtester_realistic, "market_impact_model"):
                self.results["initialization"]["realistic_components"] = True
                self.log("Realistic execution components initialized", "success")
            else:
                self.add_warning("initialization", "Realistic components not properly initialized")

        except Exception as e:
            self.add_error("initialization", f"Component initialization failed: {e}")

    def test_backtesting(self):
        """Test 3: Pruebas de backtesting completas"""
        self.log("=== TEST 3: BACKTESTING TESTS ===")

        try:
            from core.execution.backtester_core import BacktesterCore
            import pandas as pd
            import numpy as np

            # Crear datos sintéticos simples
            dates = pd.date_range("2025-01-01", periods=50, freq="D")
            np.random.seed(42)

            data = {
                "close": 100 + np.cumsum(np.random.standard_normal(50) * 0.5),
                "volume": np.random.randint(1000, 10000, 50),
                "high": 100 + np.cumsum(np.random.standard_normal(50) * 0.5) + 1,
                "low": 100 + np.cumsum(np.random.standard_normal(50) * 0.5) - 1,
                "open": 100 + np.cumsum(np.random.standard_normal(50) * 0.5) + 0.1,
            }
            df = pd.DataFrame(data, index=dates)

            # Crear datos multi-tf simples
            df_multi_tf = {"5min": df}

            # Estrategia simple mock
            class MockStrategy:
                def get_parameters(self):
                    return {}

                def generate_signals(self, df_multi_tf):
                    # Señales simples: comprar si precio > promedio móvil
                    df = df_multi_tf["5min"]
                    df["sma"] = df["close"].rolling(10).mean()
                    entries = df["close"] > df["sma"]
                    exits = df["close"] < df["sma"]
                    return {"entries": entries, "exits": exits}

            # Test backtesting simple
            backtester_simple = BacktesterCore(initial_capital=10000, commission=0.001, slippage_pct=0.001)

            strategy = MockStrategy()
            signals = strategy.generate_signals(df_multi_tf)
            results_simple = backtester_simple.run_simple_backtest(df_multi_tf, MockStrategy, {})

            if results_simple and "total_return_pct" in results_simple:
                self.results["backtesting"]["simple"] = True
                self.log(f"Simple backtest OK: {results_simple['total_return_pct']:.2f}% return", "success")
            else:
                self.add_error("backtesting", "Simple backtest failed")

            # Test backtesting realistic
            backtester_realistic = BacktesterCore(
                initial_capital=10000,
                commission=0.001,
                slippage_pct=0.001,
                enable_realistic_execution=True,
                latency_profile="retail_average",
            )

            results_realistic = backtester_realistic.run_simple_backtest(df_multi_tf, MockStrategy, {})

            if results_realistic and "total_return_pct" in results_realistic:
                self.results["backtesting"]["realistic"] = True
                self.log(f"Realistic backtest OK: {results_realistic['total_return_pct']:.2f}% return", "success")

                # Verificar que tenga costos de ejecución
                if "execution_costs" in results_realistic:
                    costs = results_realistic["execution_costs"]
                    self.log(f"Execution costs: ${costs.get('total_execution_cost', 0):.2f}", "success")
                else:
                    self.add_warning("backtesting", "Realistic backtest missing execution_costs")
            else:
                self.add_error("backtesting", "Realistic backtest failed")

        except Exception as e:
            self.add_error("backtesting", f"Backtesting test failed: {e}")

    def test_ui_initialization(self):
        """Test 4: Inicialización de UI (solo imports, no instanciación)"""
        self.log("=== TEST 4: UI INITIALIZATION TEST ===")

        # Configurar QApplication para modo headless
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        try:
            from PySide6.QtWidgets import QApplication

            # Crear aplicación
            app = QApplication.instance()
            if app is None:
                app = QApplication([])

            # Test importar y verificar que las clases existen (sin instanciar)
            from src.gui.platform_gui_tab3_improved import Tab3BacktestRunner

            # Verificar que la clase tenga los métodos esperados
            if hasattr(Tab3BacktestRunner, "__init__"):
                self.results["ui"]["tab3_import"] = True
                self.log("Tab3 class imported successfully", "success")
            else:
                self.add_error("ui", "Tab3 class missing __init__ method")

            # Verificar imports de otros tabs
            try:
                from src.gui.platform_gui_tab1_improved import Tab1DataManagement

                self.results["ui"]["tab1_import"] = True
                self.log("Tab1 class imported successfully", "success")
            except Exception as e:
                self.add_warning("ui", f"Tab1 import failed: {e}")

            try:
                from src.gui.platform_gui_tab2_improved import Tab2StrategyConfig

                self.results["ui"]["tab2_import"] = True
                self.log("Tab2 class imported successfully", "success")
            except Exception as e:
                self.add_warning("ui", f"Tab2 import failed: {e}")

            # Limpiar aplicación
            if app:
                app.quit()

        except Exception as e:
            self.add_error("ui", f"UI initialization failed: {e}")

    def test_data_loading(self):
        """Test 5: Carga de datos"""
        self.log("=== TEST 5: DATA LOADING TEST ===")

        try:
            # Verificar que existan archivos de datos
            data_files = ["data/btc_1H.csv", "data/btc_5Min.csv", "data/btc_15Min.csv"]

            for file_path in data_files:
                if os.path.exists(file_path):
                    self.results["end_to_end"][f"data_{os.path.basename(file_path)}"] = True
                    self.log(f"Data file exists: {file_path}", "success")
                else:
                    self.add_warning("data_loading", f"Data file missing: {file_path}")

            # Test cargar datos con pandas
            import pandas as pd

            if os.path.exists("data/btc_1H.csv"):
                df = pd.read_csv("data/btc_1H.csv")
                if len(df) > 0:
                    self.results["end_to_end"]["data_loading"] = True
                    self.log(f"Data loaded successfully: {len(df)} rows", "success")
                else:
                    self.add_warning("data_loading", "Data file is empty")

        except Exception as e:
            self.add_error("data_loading", f"Data loading failed: {e}")

    def run_all_tests(self):
        """Ejecutar todos los tests"""
        self.log("🚀 INICIANDO TESTS COMPLETOS DE INTEGRACIÓN")
        self.log("=" * 60)

        start_time = time.time()

        # Ejecutar tests
        self.test_imports()
        self.test_initialization()
        self.test_backtesting()
        self.test_ui_initialization()
        self.test_data_loading()

        end_time = time.time()
        duration = end_time - start_time

        # Generar reporte final
        self.generate_report(duration)

    def generate_report(self, duration):
        """Generar reporte final de tests"""
        self.log("=" * 60)
        self.log("📊 REPORTE FINAL DE INTEGRACIÓN")
        self.log("=" * 60)

        total_tests = 0
        passed_tests = 0

        # Contar resultados
        for category, tests in self.results.items():
            for test_name, result in tests.items():
                total_tests += 1
                if result:
                    passed_tests += 1

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        self.log(f"⏱️  Duración total: {duration:.2f} segundos")
        self.log(f"📈 Tests totales: {total_tests}")
        self.log(f"✅ Tests pasados: {passed_tests}")
        self.log(f"❌ Tests fallidos: {len(self.errors)}")
        self.log(f"⚠️  Warnings: {len(self.warnings)}")
        self.log(f"📊 Tasa de éxito: {success_rate:.1f}%")

        # Mostrar errores
        if self.errors:
            self.log("❌ ERRORES ENCONTRADOS:")
            for error in self.errors:
                self.log(f"  • {error['test']}: {error['error']}", "error")

        # Mostrar warnings
        if self.warnings:
            self.log("⚠️  WARNINGS:")
            for warning in self.warnings:
                self.log(f"  • {warning['test']}: {warning['warning']}", "warning")

        # Evaluación final
        if success_rate >= 95:
            self.log("🎉 RESULTADO: EXCELENTE - SISTEMA LISTO PARA PRODUCCIÓN", "success")
        elif success_rate >= 85:
            self.log("✅ RESULTADO: BUENO - SISTEMA FUNCIONAL CON ALGUNOS WARNINGS", "success")
        elif success_rate >= 70:
            self.log("⚠️  RESULTADO: ACEPTABLE - REQUIERE ATENCIÓN A ERRORES", "warning")
        else:
            self.log("❌ RESULTADO: CRÍTICO - REQUIERE CORRECCIONES URGENTES", "error")

        self.log("=" * 60)


def main():
    """Función principal"""
    print("🔬 TRADING IA - TEST COMPLETO DE INTEGRACIÓN")
    print("=" * 60)

    tester = IntegrationTester()
    tester.run_all_tests()

    # Salir con código apropiado
    if tester.errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
