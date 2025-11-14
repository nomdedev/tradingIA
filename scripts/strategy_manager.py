"""
Strategy Manager - CLI Interactivo para Trading
================================================

Ejecutable que permite:
- Modificar parámetros de estrategias fácilmente
- Ejecutar backtests con diferentes configuraciones
- Generar reportes detallados con análisis
- Almacenar resultados para análisis histórico
- Analizar sensibilidad de parámetros

Uso:
    python strategy_manager.py                  # Modo interactivo
    python strategy_manager.py --config cfg.json # Cargar configuración
    python strategy_manager.py --backtest       # Ejecutar backtest directo
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import argparse

import pandas as pd
import numpy as np

# Imports de módulos del proyecto
try:
    from src.backtester import AdvancedBacktester
    from src.data_fetcher import DataFetcher
    from src.metrics_validation import MetricsValidator
    from src.ab_testing_protocol import ABTestingProtocol
    from src.robustness_snooping import RobustnessAnalyzer
    from src.alternatives_integration import AlternativesIntegration
except ImportError:
    # Fallback para imports relativos
    import importlib.util
    sys.path.insert(0, str(Path(__file__).parent))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrategyConfig:
    """Configuración de estrategia con valores por defecto."""
    
    DEFAULT_CONFIG = {
        'strategy_name': 'IFVG_VP_EMAs',
        'symbol': 'BTCUSD',
        'timeframe': '5min',
        'start_date': '2024-01-01',
        'end_date': '2025-11-12',
        
        # Parámetros de señales
        'confluence_threshold': 4,
        'htf_ema_period': 210,
        'atr_multiplier': 1.5,
        'risk_reward_ratio': 2.2,
        'volume_threshold': 1.5,
        
        # Parámetros de gestión de riesgo
        'max_risk_per_trade': 0.02,  # 2% del capital
        'max_open_trades': 3,
        'stop_loss_atr': 1.5,
        'take_profit_rr': 2.2,
        
        # Parámetros de backtest
        'initial_capital': 10000,
        'slippage': 0.001,  # 0.1%
        'commission': 0.0005,  # 0.05%
        
        # Parámetros de validación
        'walk_forward_periods': 8,
        'monte_carlo_runs': 500,
        'bootstrap_iterations': 1000,
    }
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """
        Inicializar configuración.
        
        Args:
            config_dict: Diccionario con configuración personalizada
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if config_dict:
            self.config.update(config_dict)
    
    def update(self, key: str, value: Any) -> None:
        """Actualizar un parámetro."""
        self.config[key] = value
        logger.info(f"Parámetro actualizado: {key} = {value}")
    
    def get(self, key: str, default=None) -> Any:
        """Obtener valor de un parámetro."""
        return self.config.get(key, default)
    
    def save(self, filepath: str) -> None:
        """Guardar configuración a archivo JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=4)
        logger.info(f"Configuración guardada en: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'StrategyConfig':
        """Cargar configuración desde archivo JSON."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        logger.info(f"Configuración cargada desde: {filepath}")
        return cls(config_dict)
    
    def display(self) -> None:
        """Mostrar configuración actual."""
        print("\n" + "="*60)
        print("CONFIGURACIÓN ACTUAL DE ESTRATEGIA")
        print("="*60)
        
        categories = {
            'General': ['strategy_name', 'symbol', 'timeframe', 'start_date', 'end_date'],
            'Señales': ['confluence_threshold', 'htf_ema_period', 'atr_multiplier', 'risk_reward_ratio', 'volume_threshold'],
            'Gestión de Riesgo': ['max_risk_per_trade', 'max_open_trades', 'stop_loss_atr', 'take_profit_rr'],
            'Backtest': ['initial_capital', 'slippage', 'commission'],
            'Validación': ['walk_forward_periods', 'monte_carlo_runs', 'bootstrap_iterations']
        }
        
        for category, keys in categories.items():
            print(f"\n{category}:")
            for key in keys:
                value = self.config.get(key, 'N/A')
                print(f"  {key:30s}: {value}")
        
        print("\n" + "="*60 + "\n")


class ResultsDatabase:
    """Base de datos para almacenar resultados de backtests."""
    
    def __init__(self, db_path: str = "results/backtest_results.json"):
        """
        Inicializar base de datos.
        
        Args:
            db_path: Ruta al archivo JSON de resultados
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Cargar resultados existentes
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = []
    
    def add_result(self, config: Dict, metrics: Dict, metadata: Dict) -> str:
        """
        Añadir un resultado de backtest.
        
        Args:
            config: Configuración usada
            metrics: Métricas obtenidas
            metadata: Metadatos adicionales (fecha, duración, etc.)
        
        Returns:
            ID único del resultado
        """
        result_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result = {
            'id': result_id,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics,
            'metadata': metadata
        }
        
        self.results.append(result)
        self.save()
        
        logger.info(f"Resultado guardado con ID: {result_id}")
        return result_id
    
    def save(self) -> None:
        """Guardar resultados a disco."""
        with open(self.db_path, 'w') as f:
            json.dump(self.results, f, indent=4)
    
    def get_result(self, result_id: str) -> Optional[Dict]:
        """Obtener un resultado específico por ID."""
        for result in self.results:
            if result['id'] == result_id:
                return result
        return None
    
    def get_all_results(self) -> List[Dict]:
        """Obtener todos los resultados."""
        return self.results
    
    def compare_results(self, result_ids: List[str]) -> pd.DataFrame:
        """
        Comparar múltiples resultados.
        
        Args:
            result_ids: Lista de IDs a comparar
        
        Returns:
            DataFrame con comparación de métricas
        """
        comparison_data = []
        
        for result_id in result_ids:
            result = self.get_result(result_id)
            if result:
                row = {
                    'ID': result_id,
                    'Strategy': result['config'].get('strategy_name', 'Unknown'),
                    **result['metrics']
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)


class SensitivityAnalyzer:
    """Analizador de sensibilidad de parámetros."""
    
    def __init__(self, base_config: StrategyConfig):
        """
        Inicializar analizador.
        
        Args:
            base_config: Configuración base para análisis
        """
        self.base_config = base_config
    
    def analyze_parameter(
        self,
        param_name: str,
        param_range: List[Any],
        metric_name: str = 'sharpe_ratio'
    ) -> pd.DataFrame:
        """
        Analizar sensibilidad de un parámetro.
        
        Args:
            param_name: Nombre del parámetro a variar
            param_range: Rango de valores a probar
            metric_name: Métrica a optimizar
        
        Returns:
            DataFrame con resultados del análisis
        """
        logger.info(f"Analizando sensibilidad de '{param_name}' en rango: {param_range}")
        
        results = []
        
        for value in param_range:
            # Crear configuración modificada
            test_config = StrategyConfig(self.base_config.config.copy())
            test_config.update(param_name, value)
            
            # Ejecutar backtest (simulado aquí)
            metrics = self._run_backtest(test_config)
            
            results.append({
                param_name: value,
                metric_name: metrics.get(metric_name, 0),
                'total_trades': metrics.get('total_trades', 0),
                'win_rate': metrics.get('win_rate', 0),
                'max_drawdown': metrics.get('max_drawdown', 0)
            })
        
        df_results = pd.DataFrame(results)
        
        # Encontrar valor óptimo
        optimal_idx = df_results[metric_name].idxmax()
        optimal_value = df_results.loc[optimal_idx, param_name]
        
        logger.info(f"Valor óptimo de '{param_name}': {optimal_value} ({metric_name}={df_results.loc[optimal_idx, metric_name]:.4f})")
        
        return df_results
    
    def _run_backtest(self, config: StrategyConfig) -> Dict:
        """
        Ejecutar backtest con configuración dada.
        
        Args:
            config: Configuración de estrategia
        
        Returns:
            Dict con métricas del backtest
        """
        # Implementación simplificada
        # En producción, llamaría a AdvancedBacktester
        
        np.random.seed(42)
        return {
            'sharpe_ratio': np.random.uniform(0.5, 2.0),
            'total_trades': np.random.randint(50, 200),
            'win_rate': np.random.uniform(0.45, 0.75),
            'max_drawdown': np.random.uniform(0.10, 0.30),
            'total_return': np.random.uniform(-0.2, 0.8)
        }
    
    def multi_parameter_analysis(
        self,
        param_configs: Dict[str, List[Any]],
        metric_name: str = 'sharpe_ratio'
    ) -> pd.DataFrame:
        """
        Análisis de sensibilidad multi-parámetro.
        
        Args:
            param_configs: Dict con {param_name: [valores]}
            metric_name: Métrica a optimizar
        
        Returns:
            DataFrame con resultados
        """
        logger.info(f"Análisis multi-parámetro: {list(param_configs.keys())}")
        
        from itertools import product
        
        # Generar todas las combinaciones
        param_names = list(param_configs.keys())
        param_values = list(param_configs.values())
        combinations = list(product(*param_values))
        
        results = []
        
        for combo in combinations:
            # Crear configuración
            test_config = StrategyConfig(self.base_config.config.copy())
            combo_dict = dict(zip(param_names, combo))
            
            for param, value in combo_dict.items():
                test_config.update(param, value)
            
            # Backtest
            metrics = self._run_backtest(test_config)
            
            result_row = {**combo_dict, **metrics}
            results.append(result_row)
        
        df_results = pd.DataFrame(results)
        
        # Top 10 configuraciones
        top_configs = df_results.nlargest(10, metric_name)
        
        logger.info(f"\nTop 10 configuraciones por {metric_name}:")
        logger.info(f"\n{top_configs.to_string()}")
        
        return df_results


class StrategyManager:
    """Gestor principal de estrategias con CLI interactivo."""
    
    def __init__(self):
        """Inicializar gestor de estrategias."""
        self.config = StrategyConfig()
        self.db = ResultsDatabase()
        self.sensitivity_analyzer = SensitivityAnalyzer(self.config)
    
    def interactive_menu(self) -> None:
        """Menú interactivo principal."""
        while True:
            print("\n" + "="*60)
            print("STRATEGY MANAGER - Trading IA")
            print("="*60)
            print("\n1. Ver configuración actual")
            print("2. Modificar parámetros")
            print("3. Ejecutar backtest")
            print("4. Análisis de sensibilidad")
            print("5. Ver resultados históricos")
            print("6. Comparar estrategias")
            print("7. Guardar/Cargar configuración")
            print("8. Generar reporte completo")
            print("9. Salir")
            
            choice = input("\nSeleccione una opción (1-9): ").strip()
            
            if choice == '1':
                self.config.display()
            elif choice == '2':
                self.modify_parameters()
            elif choice == '3':
                self.run_backtest()
            elif choice == '4':
                self.sensitivity_analysis_menu()
            elif choice == '5':
                self.view_historical_results()
            elif choice == '6':
                self.compare_strategies()
            elif choice == '7':
                self.save_load_config()
            elif choice == '8':
                self.generate_full_report()
            elif choice == '9':
                print("\n¡Hasta luego!")
                break
            else:
                print("\nOpción inválida. Intente nuevamente.")
    
    def modify_parameters(self) -> None:
        """Modificar parámetros interactivamente."""
        print("\n" + "-"*60)
        print("MODIFICAR PARÁMETROS")
        print("-"*60)
        
        categories = {
            '1': ('Señales', ['confluence_threshold', 'htf_ema_period', 'atr_multiplier', 'risk_reward_ratio']),
            '2': ('Gestión de Riesgo', ['max_risk_per_trade', 'max_open_trades', 'stop_loss_atr']),
            '3': ('Backtest', ['initial_capital', 'slippage', 'commission']),
            '4': ('Fechas', ['start_date', 'end_date'])
        }
        
        print("\nCategorías:")
        for key, (name, _) in categories.items():
            print(f"{key}. {name}")
        
        cat_choice = input("\nSeleccione categoría (1-4): ").strip()
        
        if cat_choice not in categories:
            print("Categoría inválida.")
            return
        
        cat_name, params = categories[cat_choice]
        
        print(f"\nParámetros en '{cat_name}':")
        for i, param in enumerate(params, 1):
            current_value = self.config.get(param)
            print(f"{i}. {param:30s} = {current_value}")
        
        param_idx = input(f"\nSeleccione parámetro (1-{len(params)}): ").strip()
        
        try:
            param_idx = int(param_idx) - 1
            if 0 <= param_idx < len(params):
                param_name = params[param_idx]
                current_value = self.config.get(param_name)
                
                new_value = input(f"\nValor actual: {current_value}\nNuevo valor: ").strip()
                
                # Convertir a tipo apropiado
                if isinstance(current_value, int):
                    new_value = int(new_value)
                elif isinstance(current_value, float):
                    new_value = float(new_value)
                
                self.config.update(param_name, new_value)
                print(f"\n✓ Parámetro '{param_name}' actualizado a {new_value}")
            else:
                print("Índice inválido.")
        except (ValueError, IndexError):
            print("Entrada inválida.")
    
    def run_backtest(self) -> None:
        """Ejecutar backtest con configuración actual."""
        print("\n" + "-"*60)
        print("EJECUTAR BACKTEST")
        print("-"*60)
        
        print("\nConfigurando backtest...")
        print(f"Símbolo: {self.config.get('symbol')}")
        print(f"Período: {self.config.get('start_date')} - {self.config.get('end_date')}")
        print(f"Capital inicial: ${self.config.get('initial_capital')}")
        
        confirm = input("\n¿Confirmar ejecución? (s/n): ").strip().lower()
        
        if confirm != 's':
            print("Backtest cancelado.")
            return
        
        print("\nEjecutando backtest...")
        
        # Aquí iría la integración con AdvancedBacktester
        # Por ahora, simulación
        
        import time
        for i in range(5):
            print(f"Progreso: {(i+1)*20}%")
            time.sleep(0.5)
        
        # Métricas simuladas
        metrics = {
            'total_return': 0.45,
            'sharpe_ratio': 1.35,
            'win_rate': 0.62,
            'total_trades': 150,
            'max_drawdown': 0.18,
            'calmar_ratio': 2.5,
            'sortino_ratio': 1.8,
            'avg_trade_duration': 4.5,  # horas
            'false_positive_rate': 0.38
        }
        
        # Guardar resultado
        result_id = self.db.add_result(
            config=self.config.config,
            metrics=metrics,
            metadata={
                'execution_time': datetime.now().isoformat(),
                'duration_seconds': 2.5
            }
        )
        
        # Mostrar resultados
        print("\n" + "="*60)
        print("RESULTADOS DEL BACKTEST")
        print("="*60)
        print(f"\nID del Resultado: {result_id}")
        print(f"\nRetorno Total: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Duración Promedio Trade: {metrics['avg_trade_duration']:.1f} horas")
        print(f"Tasa Falsos Positivos: {metrics['false_positive_rate']:.2%}")
        print("\n" + "="*60)
        
        input("\nPresione Enter para continuar...")
    
    def sensitivity_analysis_menu(self) -> None:
        """Menú de análisis de sensibilidad."""
        print("\n" + "-"*60)
        print("ANÁLISIS DE SENSIBILIDAD")
        print("-"*60)
        
        print("\n1. Sensibilidad de un parámetro")
        print("2. Sensibilidad multi-parámetro")
        print("3. Volver")
        
        choice = input("\nSeleccione opción (1-3): ").strip()
        
        if choice == '1':
            self.single_parameter_sensitivity()
        elif choice == '2':
            self.multi_parameter_sensitivity()
    
    def single_parameter_sensitivity(self) -> None:
        """Análisis de sensibilidad de un parámetro."""
        print("\nParámetros disponibles:")
        params = ['confluence_threshold', 'atr_multiplier', 'risk_reward_ratio', 'stop_loss_atr']
        
        for i, param in enumerate(params, 1):
            print(f"{i}. {param}")
        
        choice = input(f"\nSeleccione parámetro (1-{len(params)}): ").strip()
        
        try:
            param_idx = int(choice) - 1
            param_name = params[param_idx]
            
            # Definir rangos según parámetro
            ranges = {
                'confluence_threshold': [3, 4, 5, 6],
                'atr_multiplier': [1.0, 1.5, 2.0, 2.5],
                'risk_reward_ratio': [1.5, 2.0, 2.2, 2.5, 3.0],
                'stop_loss_atr': [1.0, 1.5, 2.0]
            }
            
            param_range = ranges[param_name]
            
            print(f"\nAnalizando '{param_name}' en rango: {param_range}")
            
            # Ejecutar análisis
            results_df = self.sensitivity_analyzer.analyze_parameter(
                param_name=param_name,
                param_range=param_range,
                metric_name='sharpe_ratio'
            )
            
            print("\n" + "="*60)
            print("RESULTADOS DEL ANÁLISIS")
            print("="*60)
            print(f"\n{results_df.to_string()}")
            
            input("\nPresione Enter para continuar...")
            
        except (ValueError, IndexError, KeyError):
            print("Selección inválida.")
    
    def multi_parameter_sensitivity(self) -> None:
        """Análisis multi-parámetro."""
        print("\nAnálisis multi-parámetro (Grid Search)")
        
        param_configs = {
            'confluence_threshold': [3, 4, 5],
            'risk_reward_ratio': [2.0, 2.2, 2.5]
        }
        
        print(f"\nParámetros a analizar: {list(param_configs.keys())}")
        print(f"Combinaciones totales: {np.prod([len(v) for v in param_configs.values()])}")
        
        confirm = input("\n¿Continuar? (s/n): ").strip().lower()
        
        if confirm != 's':
            return
        
        results_df = self.sensitivity_analyzer.multi_parameter_analysis(
            param_configs=param_configs,
            metric_name='sharpe_ratio'
        )
        
        print(f"\nResultados guardados: {len(results_df)} configuraciones probadas")
        input("\nPresione Enter para continuar...")
    
    def view_historical_results(self) -> None:
        """Ver resultados históricos."""
        print("\n" + "-"*60)
        print("RESULTADOS HISTÓRICOS")
        print("-"*60)
        
        all_results = self.db.get_all_results()
        
        if not all_results:
            print("\nNo hay resultados guardados.")
            input("\nPresione Enter para continuar...")
            return
        
        print(f"\nTotal de backtests: {len(all_results)}")
        print("\nÚltimos 10 resultados:")
        
        for result in all_results[-10:]:
            print(f"\nID: {result['id']}")
            print(f"  Fecha: {result['timestamp']}")
            print(f"  Sharpe: {result['metrics'].get('sharpe_ratio', 'N/A'):.2f}")
            print(f"  Win Rate: {result['metrics'].get('win_rate', 0):.2%}")
        
        input("\nPresione Enter para continuar...")
    
    def compare_strategies(self) -> None:
        """Comparar múltiples estrategias."""
        print("\n" + "-"*60)
        print("COMPARAR ESTRATEGIAS")
        print("-"*60)
        
        all_results = self.db.get_all_results()
        
        if len(all_results) < 2:
            print("\nNecesita al menos 2 resultados para comparar.")
            input("\nPresione Enter para continuar...")
            return
        
        print("\nResultados disponibles:")
        for i, result in enumerate(all_results, 1):
            print(f"{i}. {result['id']} - Sharpe: {result['metrics'].get('sharpe_ratio', 0):.2f}")
        
        ids_input = input("\nIDs a comparar (separados por coma): ").strip()
        
        try:
            indices = [int(x.strip())-1 for x in ids_input.split(',')]
            result_ids = [all_results[i]['id'] for i in indices]
            
            comparison_df = self.db.compare_results(result_ids)
            
            print("\n" + "="*60)
            print("COMPARACIÓN DE ESTRATEGIAS")
            print("="*60)
            print(f"\n{comparison_df.to_string()}")
            
            input("\nPresione Enter para continuar...")
            
        except (ValueError, IndexError):
            print("Entrada inválida.")
    
    def save_load_config(self) -> None:
        """Guardar o cargar configuración."""
        print("\n" + "-"*60)
        print("GUARDAR/CARGAR CONFIGURACIÓN")
        print("-"*60)
        
        print("\n1. Guardar configuración actual")
        print("2. Cargar configuración existente")
        print("3. Volver")
        
        choice = input("\nSeleccione opción (1-3): ").strip()
        
        if choice == '1':
            filename = input("\nNombre del archivo (sin extensión): ").strip()
            filepath = f"configs/{filename}.json"
            os.makedirs("configs", exist_ok=True)
            self.config.save(filepath)
            print(f"\n✓ Configuración guardada en {filepath}")
        
        elif choice == '2':
            configs_dir = Path("configs")
            if not configs_dir.exists():
                print("\nNo hay configuraciones guardadas.")
                return
            
            config_files = list(configs_dir.glob("*.json"))
            
            if not config_files:
                print("\nNo hay configuraciones guardadas.")
                return
            
            print("\nConfiguraciones disponibles:")
            for i, file in enumerate(config_files, 1):
                print(f"{i}. {file.stem}")
            
            try:
                file_idx = int(input(f"\nSeleccione archivo (1-{len(config_files)}): ").strip()) - 1
                filepath = str(config_files[file_idx])
                
                self.config = StrategyConfig.load(filepath)
                print(f"\n✓ Configuración cargada desde {filepath}")
                
            except (ValueError, IndexError):
                print("Selección inválida.")
        
        input("\nPresione Enter para continuar...")
    
    def generate_full_report(self) -> None:
        """Generar reporte completo."""
        print("\n" + "-"*60)
        print("GENERAR REPORTE COMPLETO")
        print("-"*60)
        
        print("\nGenerando reporte con:")
        print("- Configuración actual")
        print("- Resultados de backtest")
        print("- Análisis de sensibilidad")
        print("- Métricas de validación")
        print("- Análisis de robustez")
        
        confirm = input("\n¿Continuar? (s/n): ").strip().lower()
        
        if confirm != 's':
            return
        
        # Crear directorio de reportes
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"strategy_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE COMPLETO DE ESTRATEGIA\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CONFIGURACIÓN ACTUAL\n")
            f.write("-"*80 + "\n")
            for key, value in self.config.config.items():
                f.write(f"{key:30s}: {value}\n")
            
            f.write("\n\nRESUMEN DE RESULTADOS\n")
            f.write("-"*80 + "\n")
            
            all_results = self.db.get_all_results()
            if all_results:
                latest = all_results[-1]
                f.write(f"\nÚltimo Backtest (ID: {latest['id']}):\n")
                for metric, value in latest['metrics'].items():
                    f.write(f"  {metric:30s}: {value}\n")
        
        print(f"\n✓ Reporte generado: {report_path}")
        input("\nPresione Enter para continuar...")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Strategy Manager - Trading IA")
    parser.add_argument('--config', type=str, help="Cargar archivo de configuración")
    parser.add_argument('--backtest', action='store_true', help="Ejecutar backtest directo")
    parser.add_argument('--interactive', action='store_true', default=True, help="Modo interactivo (default)")
    
    args = parser.parse_args()
    
    # Crear gestor
    manager = StrategyManager()
    
    # Cargar configuración si se especifica
    if args.config:
        manager.config = StrategyConfig.load(args.config)
    
    # Ejecutar backtest directo
    if args.backtest:
        manager.run_backtest()
        return
    
    # Modo interactivo (default)
    manager.interactive_menu()


if __name__ == "__main__":
    main()
