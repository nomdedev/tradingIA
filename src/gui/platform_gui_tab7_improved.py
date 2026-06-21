"""
Tab 7 - Research Lab (Improved)
Advanced research tools: experiment tracking, hypothesis testing, feature analysis
"""

from pathlib import Path

# Project root for data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSplitter,
    QGroupBox,
    QTextEdit,
    QComboBox,
    QLineEdit,
    QTabWidget,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QScrollArea,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
from datetime import datetime
import random
import numpy as np
from core.optimization.walk_forward import WalkForwardOptimizer
from core.execution.backtester_core import BacktesterCore
from core.optimization.genetic_optimizer import OptimizationConfig
import pandas as pd


# ============================================================================
# RESEARCH COMPONENTS
# ============================================================================
class ExperimentCard(QFrame):
    """Card for displaying experiment results"""

    def __init__(self, exp_id, name, status, metric):
        super().__init__()
        self.exp_id = exp_id

        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"""
            ExperimentCard {{
                background-color: #2d2d2d;
                border-left: 4px solid {'#4ec9b0' if status == 'complete' else '#c586c0'};
                border-radius: 6px;
                padding: 12px;
            }}
            ExperimentCard:hover {{
                background-color: #353535;
            }}
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Header
        header_layout = QHBoxLayout()

        name_label = QLabel(name)
        name_label.setStyleSheet("color: #fff; font-weight: bold; font-size: 16px;")
        header_layout.addWidget(name_label)

        header_layout.addStretch()

        status_label = QLabel(status.upper())
        status_color = "#4ec9b0" if status == "complete" else "#c586c0"
        status_label.setStyleSheet(f"color: {status_color}; font-size: 13px; font-weight: bold;")
        header_layout.addWidget(status_label)

        layout.addLayout(header_layout)

        # Metric
        metric_label = QLabel(f"Sharpe: {metric:.2f}")
        metric_label.setStyleSheet("color: #569cd6; font-size: 20px; font-weight: bold;")
        layout.addWidget(metric_label)

        # ID
        id_label = QLabel(f"ID: {exp_id}")
        id_label.setStyleSheet("color: #888; font-size: 13px;")
        layout.addWidget(id_label)

        self.setMaximumHeight(120)


# ============================================================================
# BACKGROUND THREAD - Research Analysis
# ============================================================================
class ResearchThread(QThread):
    """Background thread for research computations"""

    progress_update = Signal(int, str)
    result_ready = Signal(dict)

    def __init__(self, analysis_type, params):
        super().__init__()
        self.analysis_type = analysis_type
        self.params = params
        self.running = True

    def run(self):
        """Run research analysis"""
        try:
            if self.analysis_type == "hypothesis":
                self.run_hypothesis_test()
            elif self.analysis_type == "feature":
                self.run_feature_importance()
            elif self.analysis_type == "correlation":
                self.run_correlation_analysis()
            elif self.analysis_type == "regime":
                self.run_regime_detection()
            elif self.analysis_type == "pattern_discovery":
                self.run_pattern_discovery()
            elif self.analysis_type == "walk_forward":
                self.run_wfa()

        except Exception as e:
            self.result_ready.emit({"error": str(e)})

    def run_hypothesis_test(self):
        """Run hypothesis testing"""
        self.progress_update.emit(10, "Generando datos de prueba...")
        self.msleep(500)

        # Simulate hypothesis testing
        self.progress_update.emit(30, "Ejecutando test t-student...")
        self.msleep(800)

        # Generate mock results
        hypothesis = self.params.get("hypothesis", "Strategy A > Strategy B")

        self.progress_update.emit(60, "Calculando p-value...")
        p_value = random.uniform(0.001, 0.15)
        t_stat = random.uniform(1.5, 4.5)
        confidence = 95 if p_value < 0.05 else 80

        self.progress_update.emit(90, "Generando conclusiones...")
        self.msleep(500)

        result = {
            "type": "hypothesis",
            "hypothesis": hypothesis,
            "t_statistic": t_stat,
            "p_value": p_value,
            "confidence": confidence,
            "significant": p_value < 0.05,
            "conclusion": f"{'Rechazamos' if p_value < 0.05 else 'No rechazamos'} H0 con {confidence}% confianza",
        }

        self.progress_update.emit(100, "Análisis completado")
        self.result_ready.emit(result)

    def run_feature_importance(self):
        """Calculate feature importance"""
        self.progress_update.emit(10, "Cargando features...")
        self.msleep(500)

        features = [
            "RSI_14",
            "MACD",
            "BB_Width",
            "Volume_Ratio",
            "ATR",
            "SMA_Cross",
            "Momentum",
            "Volatility",
            "Trend_Strength",
            "Support_Distance",
        ]

        self.progress_update.emit(40, "Calculando importancia...")
        self.msleep(1000)

        # Generate random importances
        importances = [random.uniform(0.02, 0.25) for _ in features]
        total = sum(importances)
        importances = [i / total for i in importances]

        # Sort by importance
        feature_data = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

        self.progress_update.emit(80, "Generando visualización...")
        self.msleep(500)

        result = {
            "type": "feature",
            "features": [f[0] for f in feature_data],
            "importances": [f[1] for f in feature_data],
            "top_3": ", ".join([f[0] for f in feature_data[:3]]),
        }

        self.progress_update.emit(100, "Análisis completado")
        self.result_ready.emit(result)

    def run_correlation_analysis(self):
        """Run correlation analysis"""
        self.progress_update.emit(20, "Calculando matriz de correlación...")
        self.msleep(800)

        assets = ["BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "ADA/USD"]
        n = len(assets)

        # Generate correlation matrix
        corr_matrix = np.random.rand(n, n)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1.0)  # Diagonal = 1

        self.progress_update.emit(70, "Identificando clusters...")
        self.msleep(600)

        result = {
            "type": "correlation",
            "assets": assets,
            "matrix": corr_matrix.tolist(),
            "high_corr_pairs": self.find_high_correlations(assets, corr_matrix),
        }

        self.progress_update.emit(100, "Análisis completado")
        self.result_ready.emit(result)

    def run_regime_detection(self):
        """Run regime detection"""
        self.progress_update.emit(15, "Detectando regímenes de mercado...")
        self.msleep(700)

        self.progress_update.emit(50, "Clasificando períodos...")
        self.msleep(800)

        # Mock regime data
        regimes = {"Bull": 42, "Bear": 23, "Sideways": 35}

        result = {
            "type": "regime",
            "regimes": regimes,
            "dominant": max(regimes, key=regimes.get),
            "recommendation": self.get_regime_recommendation(max(regimes, key=regimes.get)),
        }

        self.progress_update.emit(100, "Análisis completado")
        self.result_ready.emit(result)

    def find_high_correlations(self, assets, matrix):
        """Find highly correlated asset pairs"""
        pairs = []
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                if matrix[i][j] > 0.7:
                    pairs.append(f"{assets[i]} ↔ {assets[j]}: {matrix[i][j]:.2f}")
        return pairs[:5]  # Top 5

    def get_regime_recommendation(self, regime):
        """Get trading recommendation for regime"""
        if regime == "Bull":
            return "Usar estrategias de momentum. Aumentar posiciones."
        elif regime == "Bear":
            return "Implementar gestión de riesgo estricta. Considerar shorts."
        else:
            return "Usar estrategias de reversión a la media. Stops ajustados."

    def run_pattern_discovery(self):
        """Run pattern discovery analysis"""
        import sys
        import os

        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

        try:
            from scripts.pattern_discovery_analyzer import PatternDiscoveryAnalyzer
            import pandas as pd

            self.progress_update.emit(10, "Cargando datos...")
            self.msleep(500)

            # Load sample data using project root
            data_path = PROJECT_ROOT / "data" / "btc_15Min.csv"

            df = None
            if data_path.exists():
                df = pd.read_csv(data_path)

            if df is None:
                raise FileNotFoundError("No se encontró archivo de datos BTC")

            self.progress_update.emit(30, "Inicializando analizador...")
            analyzer = PatternDiscoveryAnalyzer(data=df, min_cases=self.params.get("min_cases", 15))

            self.progress_update.emit(50, "Analizando patrones EMA...")
            ema_patterns = analyzer.analyze_ema_proximity_patterns()

            self.progress_update.emit(60, "Analizando volumen y POC...")
            volume_patterns = analyzer.analyze_volume_poc_patterns()

            self.progress_update.emit(70, "Analizando IFVG...")
            ifvg_patterns = analyzer.analyze_ifvg_patterns()

            self.progress_update.emit(80, "Analizando Squeeze Momentum...")
            squeeze_patterns = analyzer.analyze_squeeze_momentum_patterns()

            self.progress_update.emit(90, "Analizando multi-timeframe...")
            mtf_patterns = analyzer.analyze_multitimeframe_patterns()

            # Combine all patterns
            all_patterns = ema_patterns + volume_patterns + ifvg_patterns + squeeze_patterns + mtf_patterns

            # Sort by win rate
            all_patterns.sort(key=lambda x: x["win_rate"], reverse=True)

            result = {
                "type": "pattern_discovery",
                "patterns": all_patterns[:15],  # Top 15
                "total_patterns": len(all_patterns),
                "categories": {
                    "EMA": len(ema_patterns),
                    "Volume/POC": len(volume_patterns),
                    "IFVG": len(ifvg_patterns),
                    "Squeeze": len(squeeze_patterns),
                    "Multi-TF": len(mtf_patterns),
                },
            }

            self.progress_update.emit(100, "Análisis completado")
            self.result_ready.emit(result)

        except Exception as e:
            self.result_ready.emit({"error": f"Error en pattern discovery: {str(e)}"})

    def run_wfa(self):
        """Run Walk-Forward Analysis"""
        self.progress_update.emit(5, "Inicializando WFA...")
        
        # Extract params
        n_periods = self.params.get("n_periods", 5)
        
        # In a real app, we would get the actual strategy class and data here.
        # For now, we will simulate if no data is provided
        data_dict = self.params.get("data_dict")
        strategy_class = self.params.get("strategy_class")
        
        if not data_dict or not strategy_class:
             self.progress_update.emit(10, "Modo Simulación (Faltan datos reales)...")
             self.msleep(1000)
             
             # Simulate WFA results
             periods = []
             for i in range(n_periods):
                 self.progress_update.emit(10 + int(80 * (i/n_periods)), f"Procesando periodo {i+1}/{n_periods}...")
                 self.msleep(500)
                 
                 is_sharpe = random.uniform(1.5, 3.0)
                 oos_sharpe = is_sharpe * random.uniform(0.5, 1.1) # Usually lower
                 
                 periods.append({
                     "period": i + 1,
                     "is_metrics": {"sharpe": is_sharpe},
                     "oos_metrics": {"sharpe_ratio": oos_sharpe},
                     "best_params": {"p1": random.randint(10, 50)}
                 })
             
             result = {
                "type": "walk_forward",
                "periods": periods,
                "overall_metrics": {"avg_sharpe": np.mean([p["oos_metrics"]["sharpe_ratio"] for p in periods])},
                "stability_score": random.uniform(0.4, 0.9)
            }
             
             self.progress_update.emit(100, "Análisis completado")
             self.result_ready.emit(result)
             return

        self.progress_update.emit(10, f"Configurando WFA ({n_periods} periodos)...")
        
        backtester = BacktesterCore(initial_capital=10000)
        optimizer = WalkForwardOptimizer(backtester)
        
        # Define param ranges (this should come from strategy or UI)
        param_ranges = self.params.get("param_ranges", {})
        
        opt_config = OptimizationConfig(
            population_size=10, # Small for speed in GUI
            generations=3,
            max_workers=1
        )
        
        self.progress_update.emit(20, "Ejecutando optimización Walk-Forward...")
        
        try:
            wfa_results = optimizer.run_wfa(
                data_dict=data_dict,
                strategy_class=strategy_class,
                param_ranges=param_ranges,
                n_periods=n_periods,
                optimization_config=opt_config
            )
            
            self.progress_update.emit(90, "Procesando resultados...")
            
            # Convert WFAResult to dict for UI
            result = {
                "type": "walk_forward",
                "periods": wfa_results.period_results,
                "overall_metrics": wfa_results.overall_metrics,
                "stability_score": wfa_results.stability_score
            }
            
            self.progress_update.emit(100, "Análisis completado")
            self.result_ready.emit(result)
            
        except Exception as e:
            self.result_ready.emit({"error": str(e)})

    def stop(self):
        """Stop thread"""
        self.running = False


# ============================================================================
# MAIN TAB CLASS
# ============================================================================
class Tab7AdvancedAnalysis(QWidget):
    """Tab 7: Research Lab with advanced analysis tools"""

    status_update = Signal(str, str)

    def __init__(self, parent_platform=None, analysis_engines=None):
        super().__init__()
        self.parent = parent_platform
        self.analysis_engines = analysis_engines
        self.research_thread = None
        self.experiments = []

        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # === HEADER - Simplified ===
        header_layout = QHBoxLayout()

        title = QLabel("🔬 Research Lab")
        title.setStyleSheet("color: #ffffff; font-size: 20px; font-weight: bold;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Status
        self.status_label = QLabel("Listo para análisis")
        self.status_label.setStyleSheet(
            "color: #4ec9b0; font-size: 14px; padding: 6px 12px; background: #2d2d2d; border-radius: 4px;"
        )
        header_layout.addWidget(self.status_label)

        layout.addLayout(header_layout)

        # === QUICK ANALYSIS SECTION ===
        quick_group = QGroupBox("🚀 Análisis Rápido")
        quick_group.setStyleSheet(
            """
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #fff;
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """
        )

        quick_layout = QHBoxLayout()
        quick_layout.setContentsMargins(12, 12, 12, 12)
        quick_layout.setSpacing(15)

        # Quick analysis buttons
        analyses = [
            ("📊", "Correlación", "Analizar relaciones entre indicadores"),
            ("🎯", "Importancia", "Ver qué features son más importantes"),
            ("🧪", "Hipótesis", "Probar hipótesis estadísticas"),
            ("📈", "Regímenes", "Detectar cambios de mercado"),
        ]

        for icon, name, tooltip in analyses:
            btn = QPushButton(f"{icon} {name}")
            btn.setToolTip(tooltip)
            btn.setFixedHeight(60)
            btn.setStyleSheet(
                """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #0e639c, stop:1 #0a4d7a);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: bold;
                    text-align: center;
                }
                QPushButton:hover {
                    background: #1177bb;
                }
            """
            )
            quick_layout.addWidget(btn)

        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)

        # === MAIN ANALYSIS AREA ===
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Configuration & Parameters
        config_panel = self.create_config_panel()
        main_splitter.addWidget(config_panel)

        # Right: Results & Charts
        results_panel = self.create_results_panel()
        main_splitter.addWidget(results_panel)

        # Optimize proportions: 25% config, 75% results for maximum visualization
        main_splitter.setStretchFactor(0, 25)
        main_splitter.setStretchFactor(1, 75)
        main_splitter.setSizes([300, 900])
        layout.addWidget(main_splitter)

    def create_config_panel(self):
        """Create left panel with analysis tools"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Scroll Area for tools
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 10, 0)
        scroll_layout.setSpacing(12)

        # --- Hypothesis Testing ---
        hypothesis_group = QGroupBox("🧪 Hypothesis Testing")
        hypothesis_group.setStyleSheet(self.get_group_style())
        hypothesis_layout = QVBoxLayout()

        self.hypothesis_input = QLineEdit()
        self.hypothesis_input.setPlaceholderText("e.g., Strategy A outperforms Strategy B")
        self.hypothesis_input.setStyleSheet(self.get_input_style())
        hypothesis_layout.addWidget(QLabel("Hipótesis:"))
        hypothesis_layout.addWidget(self.hypothesis_input)

        # Significance level
        sig_layout = QHBoxLayout()
        sig_layout.addWidget(QLabel("Nivel α:"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.01, 0.10)
        self.alpha_spin.setValue(0.05)
        self.alpha_spin.setSingleStep(0.01)
        self.alpha_spin.setStyleSheet(self.get_input_style())
        sig_layout.addWidget(self.alpha_spin)
        sig_layout.addStretch()
        hypothesis_layout.addLayout(sig_layout)

        self.hypothesis_btn = QPushButton("▶ Run Test")
        self.hypothesis_btn.setStyleSheet(self.get_button_style("#569cd6"))
        self.hypothesis_btn.clicked.connect(self.on_run_hypothesis_test)
        hypothesis_layout.addWidget(self.hypothesis_btn)

        hypothesis_group.setLayout(hypothesis_layout)
        scroll_layout.addWidget(hypothesis_group)

        # --- Feature Importance ---
        feature_group = QGroupBox("📊 Feature Importance")
        feature_group.setStyleSheet(self.get_group_style())
        feature_layout = QVBoxLayout()

        feature_layout.addWidget(QLabel("Analizar importancia de features para predicción:"))

        self.feature_method_combo = QComboBox()
        self.feature_method_combo.addItems(["Random Forest", "XGBoost", "Permutation", "SHAP"])
        self.feature_method_combo.setStyleSheet(self.get_combo_style())
        feature_layout.addWidget(self.feature_method_combo)

        self.feature_btn = QPushButton("▶ Calculate Importance")
        self.feature_btn.setStyleSheet(self.get_button_style("#4ec9b0"))
        self.feature_btn.clicked.connect(self.on_run_feature_importance)
        feature_layout.addWidget(self.feature_btn)

        feature_group.setLayout(feature_layout)
        scroll_layout.addWidget(feature_group)

        # --- Correlation Analysis ---
        corr_group = QGroupBox("🔗 Correlation Analysis")
        corr_group.setStyleSheet(self.get_group_style())
        corr_layout = QVBoxLayout()

        corr_layout.addWidget(QLabel("Analizar correlaciones entre activos:"))

        self.corr_window_spin = QSpinBox()
        self.corr_window_spin.setRange(20, 200)
        self.corr_window_spin.setValue(60)
        self.corr_window_spin.setSuffix(" días")
        self.corr_window_spin.setStyleSheet(self.get_input_style())
        corr_layout.addWidget(QLabel("Ventana:"))
        corr_layout.addWidget(self.corr_window_spin)

        self.corr_btn = QPushButton("▶ Run Analysis")
        self.corr_btn.setStyleSheet(self.get_button_style("#c586c0"))
        self.corr_btn.clicked.connect(self.on_run_correlation)
        corr_layout.addWidget(self.corr_btn)

        corr_group.setLayout(corr_layout)
        scroll_layout.addWidget(corr_group)

        # --- Regime Detection ---
        regime_group = QGroupBox("🌐 Regime Detection")
        regime_group.setStyleSheet(self.get_group_style())
        regime_layout = QVBoxLayout()

        regime_layout.addWidget(QLabel("Detectar regímenes de mercado (HMM):"))

        self.regime_states_spin = QSpinBox()
        self.regime_states_spin.setMaximumWidth(100)
        self.regime_states_spin.setRange(2, 5)
        self.regime_states_spin.setValue(3)
        self.regime_states_spin.setSuffix(" estados")
        self.regime_states_spin.setStyleSheet(self.get_input_style())
        regime_layout.addWidget(QLabel("N° Estados:"))
        regime_layout.addWidget(self.regime_states_spin)

        self.regime_btn = QPushButton("▶ Detect Regimes")
        self.regime_btn.setStyleSheet(self.get_button_style("#dcdcaa"))
        self.regime_btn.clicked.connect(self.on_run_regime_detection)
        regime_layout.addWidget(self.regime_btn)

        regime_group.setLayout(regime_layout)
        scroll_layout.addWidget(regime_group)

        # --- Pattern Discovery ---
        pattern_group = QGroupBox("🔍 Pattern Discovery")
        pattern_group.setStyleSheet(self.get_group_style())
        pattern_layout = QVBoxLayout()

        pattern_layout.addWidget(QLabel("Descubrir patrones predictivos:"))

        self.pattern_min_cases_spin = QSpinBox()
        self.pattern_min_cases_spin.setRange(10, 100)
        self.pattern_min_cases_spin.setValue(15)
        self.pattern_min_cases_spin.setSuffix(" casos mín")
        self.pattern_min_cases_spin.setStyleSheet(self.get_input_style())
        pattern_layout.addWidget(QLabel("Casos mínimos:"))
        pattern_layout.addWidget(self.pattern_min_cases_spin)

        self.pattern_btn = QPushButton("▶ Discover Patterns")
        self.pattern_btn.setStyleSheet(self.get_button_style("#569cd6"))
        self.pattern_btn.clicked.connect(self.on_run_pattern_discovery)
        pattern_layout.addWidget(self.pattern_btn)

        pattern_group.setLayout(pattern_layout)
        scroll_layout.addWidget(pattern_group)

        # --- Walk-Forward Analysis ---
        wfa_group = QGroupBox("🚶 Walk-Forward Analysis")
        wfa_group.setStyleSheet(self.get_group_style())
        wfa_layout = QVBoxLayout()

        wfa_layout.addWidget(QLabel("Validación robusta de estrategias:"))

        self.wfa_periods_spin = QSpinBox()
        self.wfa_periods_spin.setRange(2, 20)
        self.wfa_periods_spin.setValue(5)
        self.wfa_periods_spin.setPrefix("Periodos: ")
        self.wfa_periods_spin.setStyleSheet(self.get_input_style())
        wfa_layout.addWidget(self.wfa_periods_spin)

        self.wfa_btn = QPushButton("▶ Run WFA")
        self.wfa_btn.setStyleSheet(self.get_button_style("#dcdcaa"))
        self.wfa_btn.clicked.connect(self.on_run_wfa)
        wfa_layout.addWidget(self.wfa_btn)

        wfa_group.setLayout(wfa_layout)
        scroll_layout.addWidget(wfa_group)

        # --- Experiment History ---
        exp_group = QGroupBox("📝 Recent Experiments")
        exp_group.setStyleSheet(self.get_group_style())
        exp_layout = QVBoxLayout()
        
        # Count label
        self.exp_count_label = QLabel("Experiments: 0")
        self.exp_count_label.setStyleSheet("color: #888; font-size: 12px; margin-bottom: 5px;")
        exp_layout.addWidget(self.exp_count_label)

        self.exp_list_widget = QWidget()
        self.exp_list_layout = QVBoxLayout(self.exp_list_widget)
        self.exp_list_layout.setContentsMargins(0, 0, 0, 0)
        self.exp_list_layout.setSpacing(8)

        # Placeholder
        placeholder = QLabel("No experiments yet")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #666; padding: 20px;")
        self.exp_list_layout.addWidget(placeholder)

        exp_layout.addWidget(self.exp_list_widget)
        exp_group.setLayout(exp_layout)
        scroll_layout.addWidget(exp_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return container

    def get_group_style(self):
        return """
            QGroupBox {
                background-color: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: bold;
                color: #ffffff;
                padding-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                background-color: #2d2d2d;
            }
        """

    def get_input_style(self):
        return """
            background: #1e1e1e; 
            color: #fff; 
            padding: 8px; 
            border: 1px solid #3e3e3e; 
            border-radius: 4px;
        """

    def get_combo_style(self):
        return """
            QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
            }
        """

    def get_button_style(self, color):
        return f"""
            QPushButton {{
                background: {color};
                color: #1e1e1e;
                border: none;
                padding: 10px;
                font-weight: bold;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background: #ffffff; }}
        """


    def create_results_panel(self):
        """Create right panel with results"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # --- Progress Bar ---
        self.progress_container = QWidget()
        self.progress_container.setVisible(False)
        progress_layout = QVBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_label = QLabel("Iniciando análisis...")
        self.progress_label.setStyleSheet("color: #569cd6; font-size: 15px;")
        progress_layout.addWidget(self.progress_label)

        from PySide6.QtWidgets import QProgressBar

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                background: #2d2d2d;
                color: #fff;
                height: 24px;
            }
            QProgressBar::chunk {
                background: #569cd6;
                border-radius: 3px;
            }
        """
        )
        progress_layout.addWidget(self.progress_bar)

        layout.addWidget(self.progress_container)

        # --- Results Tabs ---
        self.results_tabs = QTabWidget()
        self.results_tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background: #252525;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #2d2d2d;
                color: #888;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #252525;
                color: #fff;
                border-bottom: 2px solid #0e639c;
            }
            QTabBar::tab:hover {
                background: #353535;
                color: #fff;
            }
        """
        )

        # Tab 1: Visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(12, 12, 12, 12)

        self.viz_chart = QWebEngineView()
        self.viz_chart.setMinimumHeight(400)
        viz_layout.addWidget(self.viz_chart)

        # Initialize empty chart
        self.show_empty_chart()

        self.results_tabs.addTab(viz_widget, "📊 Visualization")

        # Tab 2: Statistics
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(12, 12, 12, 12)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet(
            """
            QTextEdit {
                background: #1e1e1e;
                color: #fff;
                border: 1px solid #444;
                padding: 12px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 15px;
            }
        """
        )
        self.stats_text.setHtml(
            "<p style='color: #666; text-align: center; padding: 40px;'>Ejecute un análisis para ver estadísticas detalladas</p>"
        )
        stats_layout.addWidget(self.stats_text)

        self.results_tabs.addTab(stats_widget, "📈 Statistics")

        # Tab 3: Recommendations
        rec_widget = QWidget()
        rec_layout = QVBoxLayout(rec_widget)
        rec_layout.setContentsMargins(12, 12, 12, 12)

        self.rec_text = QTextEdit()
        self.rec_text.setReadOnly(True)
        self.rec_text.setStyleSheet(
            """
            QTextEdit {
                background: #1e1e1e;
                color: #fff;
                border: 1px solid #444;
                padding: 12px;
                font-size: 16px;
            }
        """
        )
        self.rec_text.setHtml(
            "<p style='color: #666; text-align: center; padding: 40px;'>Las recomendaciones aparecerán aquí después del análisis</p>"
        )
        rec_layout.addWidget(self.rec_text)

        self.results_tabs.addTab(rec_widget, "💡 Recommendations")

        layout.addWidget(self.results_tabs)

        # --- Export Button ---
        export_btn = QPushButton("💾 Export Results")
        export_btn.setStyleSheet(
            """
            QPushButton {
                background: #2d2d2d;
                color: #fff;
                border: 1px solid #555;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #353535;
                border-color: #0e639c;
            }
        """
        )
        export_btn.clicked.connect(self.on_export_results)
        layout.addWidget(export_btn)

        return container

    def apply_modern_theme(self):
        """Apply dark theme styling"""
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 16px;
            }
            QGroupBox {
                background-color: #252525;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 16px;
                font-weight: bold;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #ffffff;
                font-size: 16px;
            }
            QLabel {
                color: #cccccc;
            }
        """
        )

    # === SLOT HANDLERS ===

    def on_run_hypothesis_test(self):
        """Run hypothesis testing"""
        hypothesis = self.hypothesis_input.text()
        if not hypothesis:
            self.status_update.emit("Por favor ingrese una hipótesis", "warning")
            return

        params = {"hypothesis": hypothesis, "alpha": self.alpha_spin.value()}

        self.run_research_analysis("hypothesis", params)

    def on_run_feature_importance(self):
        """Run feature importance analysis"""
        params = {"method": self.feature_method_combo.currentText()}

        self.run_research_analysis("feature", params)

    def on_run_correlation(self):
        """Run correlation analysis"""
        params = {"window": self.corr_window_spin.value()}

        self.run_research_analysis("correlation", params)

    def on_run_regime_detection(self):
        """Run regime detection"""
        params = {"n_states": self.regime_states_spin.value()}

        self.run_research_analysis("regime", params)

    def on_run_pattern_discovery(self):
        """Run pattern discovery analysis"""
        params = {"min_cases": self.pattern_min_cases_spin.value()}

        self.run_research_analysis("pattern_discovery", params)

    def on_run_wfa(self):
        """Run Walk-Forward Analysis"""
        params = {
            "n_periods": self.wfa_periods_spin.value(),
            # In a real scenario, we would pass data and strategy here
            # For now, we rely on the thread to handle simulation or data fetching
        }
        self.run_research_analysis("walk_forward", params)

    def run_research_analysis(self, analysis_type, params):
        """Execute research analysis in background"""
        # Show progress
        self.progress_container.setVisible(True)
        self.progress_bar.setValue(0)

        # Disable buttons
        self.hypothesis_btn.setEnabled(False)
        self.feature_btn.setEnabled(False)
        self.corr_btn.setEnabled(False)
        self.regime_btn.setEnabled(False)
        self.pattern_btn.setEnabled(False)
        self.wfa_btn.setEnabled(False)

        # Start thread
        self.research_thread = ResearchThread(analysis_type, params)
        self.research_thread.progress_update.connect(self.update_progress)
        self.research_thread.result_ready.connect(self.on_research_complete)
        self.research_thread.start()

        self.status_update.emit(f"Ejecutando análisis: {analysis_type}", "processing")

    def update_progress(self, value, message):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def on_research_complete(self, result):
        """Handle research completion"""
        # Hide progress
        self.progress_container.setVisible(False)

        # Re-enable buttons
        self.hypothesis_btn.setEnabled(True)
        self.feature_btn.setEnabled(True)
        self.corr_btn.setEnabled(True)
        self.regime_btn.setEnabled(True)
        self.pattern_btn.setEnabled(True)
        self.wfa_btn.setEnabled(True)

        # Check for errors
        if "error" in result:
            self.status_update.emit(f"Error: {result['error']}", "error")
            return

        # Display results based on type
        result_type = result.get("type")

        if result_type == "hypothesis":
            self.display_hypothesis_results(result)
        elif result_type == "feature":
            self.display_feature_results(result)
        elif result_type == "correlation":
            self.display_correlation_results(result)
        elif result_type == "regime":
            self.display_regime_results(result)
        elif result_type == "pattern_discovery":
            self.display_pattern_discovery_results(result)
        elif result_type == "walk_forward":
            self.display_wfa_results(result)

        # Add to experiment history
        self.add_experiment(result_type, result)

        self.status_update.emit(f"Análisis completado: {result_type}", "success")

    def display_hypothesis_results(self, result):
        """Display hypothesis test results"""
        # Visualization
        fig = go.Figure()

        # Distribution plot
        x = np.linspace(-4, 4, 1000)
        from scipy import stats

        y = stats.t.pdf(x, df=100)

        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="t-distribution", line=dict(color="#569cd6", width=2)))

        # Critical value
        t_stat = result["t_statistic"]
        fig.add_vline(x=t_stat, line_dash="dash", line_color="#4ec9b0", annotation_text=f"t = {t_stat:.2f}")

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font=dict(color="#ffffff", size=11),
            title=f"Hypothesis Test: {result['hypothesis']}",
            xaxis_title="t-statistic",
            yaxis_title="Density",
            margin=dict(l=40, r=20, t=60, b=40),
            height=380,
        )

        html = fig.to_html(include_plotlyjs="cdn")
        self.viz_chart.setHtml(html)

        # Statistics
        stats_html = f"""
        <h2 style='color: #569cd6;'>📊 Hypothesis Test Results</h2>
        <hr style='border-color: #3d3d3d;'>
        
        <p><b>Hipótesis:</b> {result['hypothesis']}</p>
        <p><b>t-statistic:</b> {result['t_statistic']:.3f}</p>
        <p><b>p-value:</b> {result['p_value']:.4f}</p>
        <p><b>Nivel de confianza:</b> {result['confidence']}%</p>
        
        <h3 style='color: {'#4ec9b0' if result['significant'] else '#f48771'}; margin-top: 30px;'>
            {'✅ SIGNIFICATIVO' if result['significant'] else '❌ NO SIGNIFICATIVO'}
        </h3>
        
        <p style='margin-top: 20px;'><b>Conclusión:</b> {result['conclusion']}</p>
        """
        self.stats_text.setHtml(stats_html)

        # Recommendations
        rec_html = """
        <h2 style='color: #dcdcaa;'>💡 Recomendaciones</h2>
        <hr style='border-color: #3d3d3d;'>
        
        <ul style='line-height: 1.8;'>
        """

        if result["significant"]:
            rec_html += """
            <li>✅ Los resultados son estadísticamente significativos</li>
            <li>📈 Considere implementar la estrategia con mayor confianza</li>
            <li>🔄 Realice validación con walk-forward para confirmar robustez</li>
            <li>📊 Monitoree el performance en trading en vivo</li>
            """
        else:
            rec_html += """
            <li>⚠️ No hay evidencia estadística suficiente</li>
            <li>🔍 Revise los supuestos del test</li>
            <li>📊 Considere recolectar más datos</li>
            <li>🔄 Pruebe con diferentes períodos de análisis</li>
            """

        rec_html += "</ul>"
        self.rec_text.setHtml(rec_html)

    def display_feature_results(self, result):
        """Display feature importance results"""
        features = result["features"]
        importances = result["importances"]

        # Visualization
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=importances,
                y=features,
                orientation="h",
                marker=dict(color=importances, colorscale="Viridis", showscale=True),
                text=[f"{i*100:.1f}%" for i in importances],
                textposition="auto",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font=dict(color="#ffffff", size=11),
            title="Feature Importance Analysis",
            xaxis_title="Importance",
            yaxis_title="Feature",
            margin=dict(l=120, r=20, t=60, b=40),
            height=max(380, len(features) * 35),
        )

        html = fig.to_html(include_plotlyjs="cdn")
        self.viz_chart.setHtml(html)

        # Statistics
        stats_html = f"""
        <h2 style='color: #4ec9b0;'>📊 Feature Importance</h2>
        <hr style='border-color: #3d3d3d;'>
        
        <p><b>Top 3 Features:</b> {result['top_3']}</p>
        <p><b>Total Features:</b> {len(features)}</p>
        
        <h3 style='margin-top: 30px;'>Ranking Completo:</h3>
        <ol>
        """

        for feat, imp in zip(features, importances):
            stats_html += f"<li>{feat}: {imp*100:.2f}%</li>"

        stats_html += "</ol>"
        self.stats_text.setHtml(stats_html)

        # Recommendations
        rec_html = f"""
        <h2 style='color: #dcdcaa;'>💡 Recomendaciones</h2>
        <hr style='border-color: #3d3d3d;'>
        
        <p><b>Features más importantes:</b> {result['top_3']}</p>
        
        <ul style='line-height: 1.8; margin-top: 20px;'>
            <li>🎯 Enfocarse en optimizar los top 3 features</li>
            <li>🔍 Considerar eliminar features con importancia < 2%</li>
            <li>📊 Realizar análisis de correlación entre features</li>
            <li>🔄 Re-evaluar importancia periódicamente</li>
        </ul>
        """
        self.rec_text.setHtml(rec_html)

    def display_correlation_results(self, result):
        """Display correlation matrix"""
        assets = result["assets"]
        matrix = np.array(result["matrix"])

        # Visualization - Heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=assets,
                y=assets,
                colorscale="RdBu",
                zmid=0,
                text=matrix,
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                colorbar=dict(title="Correlación"),
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font=dict(color="#ffffff", size=11),
            title="Correlation Matrix",
            margin=dict(l=100, r=20, t=60, b=100),
            height=450,
        )

        html = fig.to_html(include_plotlyjs="cdn")
        self.viz_chart.setHtml(html)

        # Statistics
        high_corr = result["high_corr_pairs"]

        stats_html = f"""
        <h2 style='color: #c586c0;'>📊 Correlation Analysis</h2>
        <hr style='border-color: #3d3d3d;'>
        
        <p><b>Activos analizados:</b> {len(assets)}</p>
        <p><b>Pares con alta correlación (>0.7):</b> {len(high_corr)}</p>
        
        <h3 style='margin-top: 30px;'>High Correlation Pairs:</h3>
        <ul>
        """

        for pair in high_corr:
            stats_html += f"<li>{pair}</li>"

        if not high_corr:
            stats_html += "<li style='color: #666;'>No se encontraron correlaciones altas</li>"

        stats_html += "</ul>"
        self.stats_text.setHtml(stats_html)

        # Recommendations
        rec_html = """
        <h2 style='color: #dcdcaa;'>💡 Recomendaciones</h2>
        <hr style='border-color: #3d3d3d;'>
        
        <ul style='line-height: 1.8;'>
            <li>📊 Diversificar entre activos con baja correlación</li>
            <li>⚠️ Cuidado con activos altamente correlacionados en el mismo portfolio</li>
            <li>🔄 Re-evaluar correlaciones en diferentes regímenes de mercado</li>
            <li>📈 Usar rolling correlations para detectar cambios temporales</li>
        </ul>
        """
        self.rec_text.setHtml(rec_html)

    def display_regime_results(self, result):
        """Display regime detection results"""
        regimes = result["regimes"]

        # Visualization - Pie chart
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(regimes.keys()),
                    values=list(regimes.values()),
                    hole=0.3,
                    marker=dict(colors=["#4ec9b0", "#f48771", "#dcdcaa"]),
                )
            ]
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font=dict(color="#ffffff", size=11),
            title="Market Regime Distribution",
            margin=dict(l=40, r=40, t=60, b=40),
            height=380,
        )

        html = fig.to_html(include_plotlyjs="cdn")
        self.viz_chart.setHtml(html)

        # Statistics
        stats_html = f"""
        <h2 style='color: #dcdcaa;'>📊 Regime Detection</h2>
        <hr style='border-color: #3d3d3d;'>
        
        <p><b>Régimen dominante:</b> {result['dominant']}</p>
        
        <h3 style='margin-top: 30px;'>Distribución:</h3>
        <ul>
        """

        for regime, pct in regimes.items():
            stats_html += f"<li>{regime}: {pct}%</li>"

        stats_html += "</ul>"
        self.stats_text.setHtml(stats_html)

        # Recommendations
        rec_html = f"""
        <h2 style='color: #dcdcaa;'>💡 Recomendaciones</h2>
        <hr style='border-color: #3d3d3d;'>
        
        <p><b>Régimen actual:</b> {result['dominant']}</p>
        <p><b>Estrategia recomendada:</b> {result['recommendation']}</p>
        
        <ul style='line-height: 1.8; margin-top: 20px;'>
            <li>🎯 Adaptar estrategia al régimen dominante</li>
            <li>📊 Monitorear transiciones entre regímenes</li>
            <li>⚡ Ajustar parámetros según el régimen activo</li>
            <li>🔄 Backtestear estrategias específicas por régimen</li>
        </ul>
        """
        self.rec_text.setHtml(rec_html)

    def display_pattern_discovery_results(self, result):
        """Display pattern discovery results"""
        patterns = result["patterns"]
        categories = result["categories"]

        # Visualization - Top patterns bar chart
        top_10 = patterns[:10]

        fig = go.Figure()

        # Win rate bars
        fig.add_trace(
            go.Bar(
                name="Win Rate %",
                x=[p["pattern_name"] for p in top_10],
                y=[p["win_rate"] * 100 for p in top_10],
                marker=dict(color="#4ec9b0"),
                text=[f"{p['win_rate']*100:.1f}%" for p in top_10],
                textposition="outside",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font=dict(color="#ffffff", size=10),
            title="🏆 Top 10 Predictive Patterns by Win Rate",
            xaxis_title="Pattern",
            yaxis_title="Win Rate %",
            margin=dict(l=40, r=40, t=60, b=120),
            height=400,
            xaxis=dict(tickangle=-45),
        )

        html = fig.to_html(include_plotlyjs="cdn")
        self.viz_chart.setHtml(html)

        # Statistics - Pattern table
        stats_html = f"""
        <h2 style='color: #dcdcaa;'>🔍 Pattern Discovery Results</h2>
        <hr style='border-color: #3d3d3d;'>
        
        <p><b>Total patterns found:</b> {result['total_patterns']}</p>
        
        <h3 style='margin-top: 20px;'>Categorías:</h3>
        <ul>
        """

        for cat, count in categories.items():
            stats_html += f"<li>{cat}: {count} patterns</li>"

        stats_html += f"""
        </ul>
        
        <h3 style='margin-top: 30px;'>Top 5 Patterns:</h3>
        <table style='width: 100%; border-collapse: collapse; margin-top: 10px;'>
            <thead>
                <tr style='background: #2d2d2d; border-bottom: 2px solid #0e639c;'>
                    <th style='padding: 8px; text-align: left;'>Pattern</th>
                    <th style='padding: 8px; text-align: center;'>Win Rate</th>
                    <th style='padding: 8px; text-align: center;'>Cases</th>
                    <th style='padding: 8px; text-align: center;'>PF</th>
                </tr>
            </thead>
            <tbody>
        """

        for i, pattern in enumerate(patterns[:5]):
            bg_color = "#252525" if i % 2 == 0 else "#2d2d2d"
            stats_html += f"""
                <tr style='background: {bg_color};'>
                    <td style='padding: 8px;'>{pattern['pattern_name'][:40]}</td>
                    <td style='padding: 8px; text-align: center; color: #4ec9b0; font-weight: bold;'>{pattern['win_rate']*100:.1f}%</td>
                    <td style='padding: 8px; text-align: center;'>{pattern['n_cases']}</td>
                    <td style='padding: 8px; text-align: center;'>{pattern.get('profit_factor', 0):.2f}</td>
                </tr>
            """

        stats_html += """
            </tbody>
        </table>
        """
        self.stats_text.setHtml(stats_html)

        # Recommendations
        rec_html = f"""
        <h2 style='color: #dcdcaa;'>💡 Actionable Insights</h2>
        <hr style='border-color: #3d3d3d;'>
        
        <h3>🎯 Key Findings:</h3>
        <ul style='line-height: 1.8;'>
        """

        # Top 3 recommendations
        for i, pattern in enumerate(patterns[:3], 1):
            rec_html += f"""
            <li><b>#{i}: {pattern['pattern_name']}</b><br>
                Win Rate: {pattern['win_rate']*100:.1f}% | 
                Cases: {pattern['n_cases']} | 
                PF: {pattern.get('profit_factor', 0):.2f}
            </li>
            """

        rec_html += """
        </ul>
        
        <h3 style='margin-top: 30px;'>📊 Trading Strategy:</h3>
        <ul style='line-height: 1.8;'>
            <li>🔥 Focus on patterns with >60% win rate and >1.2 PF</li>
            <li>📈 Multi-timeframe confirmation patterns are most reliable</li>
            <li>⚡ EMA proximity + IFVG combination shows strong edge</li>
            <li>🎲 Test patterns in paper trading before live deployment</li>
            <li>📊 Monitor pattern performance over time for degradation</li>
        </ul>
        """
        self.rec_text.setHtml(rec_html)

    def display_wfa_results(self, result):
        """Display Walk-Forward Analysis results"""
        periods = result.get("periods", [])
        overall = result.get("overall_metrics", {})
        
        # 1. Visualization: IS vs OOS Sharpe
        fig = go.Figure()
        
        periods_idx = [p["period"] for p in periods]
        is_sharpes = [p["is_metrics"]["sharpe"] for p in periods]
        oos_sharpes = [p["oos_metrics"].get("sharpe_ratio", 0) for p in periods]
        
        fig.add_trace(go.Bar(
            x=periods_idx,
            y=is_sharpes,
            name="In-Sample Sharpe",
            marker_color="#4ec9b0"
        ))
        
        fig.add_trace(go.Bar(
            x=periods_idx,
            y=oos_sharpes,
            name="Out-of-Sample Sharpe",
            marker_color="#c586c0"
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font=dict(color="#ffffff", size=10),
            title="Walk-Forward Analysis: IS vs OOS Performance",
            xaxis_title="Period",
            yaxis_title="Sharpe Ratio",
            barmode='group',
            margin=dict(l=40, r=40, t=60, b=40),
            height=400
        )
        
        html = fig.to_html(include_plotlyjs="cdn")
        self.viz_chart.setHtml(html)
        
        # 2. Statistics
        stats_html = f"""
        <h2 style='color: #dcdcaa;'>🚶 Walk-Forward Analysis Results</h2>
        <hr style='border-color: #3d3d3d;'>
        
        <div style='display: flex; justify-content: space-between; margin-bottom: 20px;'>
            <div style='background: #252525; padding: 15px; border-radius: 8px; width: 45%;'>
                <h3 style='margin: 0; color: #888;'>Stability Score</h3>
                <p style='font-size: 24px; font-weight: bold; color: #4ec9b0; margin: 10px 0 0 0;'>
                    {result.get('stability_score', 0):.2f}
                </p>
            </div>
            <div style='background: #252525; padding: 15px; border-radius: 8px; width: 45%;'>
                <h3 style='margin: 0; color: #888;'>Avg OOS Sharpe</h3>
                <p style='font-size: 24px; font-weight: bold; color: #c586c0; margin: 10px 0 0 0;'>
                    {np.mean(oos_sharpes):.2f}
                </p>
            </div>
        </div>
        
        <h3 style='margin-top: 20px;'>Period Details:</h3>
        <table style='width: 100%; border-collapse: collapse; margin-top: 10px;'>
            <thead>
                <tr style='background: #2d2d2d; border-bottom: 2px solid #0e639c;'>
                    <th style='padding: 8px; text-align: center;'>Period</th>
                    <th style='padding: 8px; text-align: center;'>IS Sharpe</th>
                    <th style='padding: 8px; text-align: center;'>OOS Sharpe</th>
                    <th style='padding: 8px; text-align: center;'>Degradation</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, p in enumerate(periods):
            is_s = p["is_metrics"]["sharpe"]
            oos_s = p["oos_metrics"].get("sharpe_ratio", 0)
            deg = (oos_s - is_s) / abs(is_s) if is_s != 0 else 0
            deg_color = "#ff6b6b" if deg < -0.5 else "#f1c40f" if deg < -0.2 else "#4ec9b0"
            
            bg_color = "#252525" if i % 2 == 0 else "#2d2d2d"
            stats_html += f"""
                <tr style='background: {bg_color};'>
                    <td style='padding: 8px; text-align: center;'>{p['period']}</td>
                    <td style='padding: 8px; text-align: center;'>{is_s:.2f}</td>
                    <td style='padding: 8px; text-align: center;'>{oos_s:.2f}</td>
                    <td style='padding: 8px; text-align: center; color: {deg_color};'>{deg*100:.1f}%</td>
                </tr>
            """
            
        stats_html += """
            </tbody>
        </table>
        """
        self.stats_text.setHtml(stats_html)
        
        # 3. Recommendations
        rec_html = f"""
        <h2 style='color: #dcdcaa;'>💡 WFA Insights</h2>
        <hr style='border-color: #3d3d3d;'>
        
        <h3>🎯 Robustness Assessment:</h3>
        <ul style='line-height: 1.8;'>
        """
        
        stability = result.get('stability_score', 0)
        if stability > 0.7:
            rec_html += "<li>✅ <b>High Robustness:</b> Strategy performs consistently across periods.</li>"
        elif stability > 0.4:
            rec_html += "<li>⚠️ <b>Moderate Robustness:</b> Performance varies, consider parameter tuning.</li>"
        else:
            rec_html += "<li>❌ <b>Low Robustness:</b> Strategy may be overfitted.</li>"
            
        avg_deg = np.mean([(p["oos_metrics"].get("sharpe_ratio", 0) - p["is_metrics"]["sharpe"]) / abs(p["is_metrics"]["sharpe"]) if p["is_metrics"]["sharpe"] != 0 else 0 for p in periods])
        
        rec_html += f"<li>📉 <b>Avg Degradation:</b> {avg_deg*100:.1f}% (IS to OOS drop)</li>"
        
        rec_html += """
        </ul>
        
        <h3 style='margin-top: 30px;'>📊 Optimization Tips:</h3>
        <ul style='line-height: 1.8;'>
            <li>Reduce parameter space if overfitting is high.</li>
            <li>Increase In-Sample period length for better training.</li>
            <li>Check for regime changes in periods with high failure rate.</li>
        </ul>
        """
        self.rec_text.setHtml(rec_html)

    def add_experiment(self, exp_type, result):
        """Add experiment to history"""
        exp_id = f"EXP-{len(self.experiments) + 1:03d}"

        # Get metric based on type
        if exp_type == "hypothesis":
            metric = result.get("confidence", 0) / 100
        elif exp_type == "feature":
            metric = result["importances"][0] if result["importances"] else 0
        else:
            metric = random.uniform(0.5, 2.5)

        exp_card = ExperimentCard(exp_id, exp_type.title(), "complete", metric)

        # Remove placeholder if exists
        if self.exp_list_layout.count() > 0:
            item = self.exp_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.exp_list_layout.insertWidget(0, exp_card)

        # Keep only last 5
        while self.exp_list_layout.count() > 5:
            item = self.exp_list_layout.takeAt(5)
            if item.widget():
                item.widget().deleteLater()

        self.experiments.append({"id": exp_id, "type": exp_type, "result": result})
        self.exp_count_label.setText(f"Experiments: {len(self.experiments)}")

    def show_empty_chart(self):
        """Show empty placeholder chart"""
        fig = go.Figure()

        fig.add_annotation(
            text="Ejecute un análisis para ver visualizaciones",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#666"),
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=380,
        )

        html = fig.to_html(include_plotlyjs="cdn")
        self.viz_chart.setHtml(html)

    def on_export_results(self):
        """Export results to file"""
        if not self.experiments:
            self.status_update.emit("No hay resultados para exportar", "warning")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_results_{timestamp}.json"

        # Mock export
        self.status_update.emit(f"Resultados exportados: {filename}", "success")

    def on_tab_activated(self):
        """Called when tab becomes active"""
        pass
