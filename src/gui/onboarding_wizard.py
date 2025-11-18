"""
Onboarding Wizard for TradingIA Platform
Guides new users through initial setup and configuration
"""

import sys
import os
from PySide6.QtWidgets import (QWizard, QWizardPage, QLabel, QVBoxLayout, QHBoxLayout,
                               QRadioButton, QButtonGroup, QPushButton, QCheckBox,
                               QLineEdit, QComboBox, QTextEdit, QProgressBar, QFrame,
                               QMessageBox, QGroupBox, QListWidget, QListWidgetItem)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon

# Import configuration manager
from src.config.user_config import UserConfigManager


class WelcomePage(QWizardPage):
    """Welcome page with user type selection"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("🎉 ¡Bienvenido a TradingIA!")
        self.setSubTitle("La plataforma más avanzada para trading algorítmico")

        layout = QVBoxLayout()

        # Welcome message
        welcome_label = QLabel(
            "¡Gracias por elegir TradingIA!\n\n"
            "Esta plataforma te ayudará a crear, probar y ejecutar estrategias "
            "de trading algorítmico de manera profesional.\n\n"
            "¿Cuál describe mejor tu situación?"
        )
        welcome_label.setWordWrap(True)
        welcome_label.setFont(QFont("Arial", 11))
        layout.addWidget(welcome_label)

        # User type selection
        self.user_type_group = QButtonGroup(self)

        self.newbie_radio = QRadioButton("🆕 Soy nuevo en el trading algorítmico")
        self.newbie_radio.setChecked(True)
        self.user_type_group.addButton(self.newbie_radio, 1)
        layout.addWidget(self.newbie_radio)

        self.experienced_radio = QRadioButton("⚡ Ya tengo experiencia con trading algorítmico")
        self.user_type_group.addButton(self.experienced_radio, 2)
        layout.addWidget(self.experienced_radio)

        self.demo_radio = QRadioButton("🎮 Solo quiero probar la plataforma (modo demo)")
        self.user_type_group.addButton(self.demo_radio, 3)
        layout.addWidget(self.demo_radio)

        # Spacer
        layout.addStretch()

        # Features preview
        features_group = QGroupBox("✨ Lo que podrás hacer:")
        features_layout = QVBoxLayout()

        features = [
            "• Crear y probar estrategias de trading automatizadas",
            "• Analizar datos históricos con backtesting avanzado",
            "• Ejecutar trading en vivo con brokers integrados",
            "• Comparar estrategias con pruebas A/B",
            "• Monitorear rendimiento en tiempo real",
            "• Gestionar riesgos automáticamente"
        ]

        for feature in features:
            feature_label = QLabel(feature)
            features_layout.addWidget(feature_label)

        features_group.setLayout(features_layout)
        layout.addWidget(features_group)

        self.setLayout(layout)

    def nextId(self):
        """Determine next page based on user type"""
        if self.newbie_radio.isChecked():
            return 1  # Tutorial page
        elif self.experienced_radio.isChecked():
            return 2  # Quick setup
        else:  # Demo mode
            return 3  # Demo setup


class TutorialPage(QWizardPage):
    """Interactive tutorial for new users"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("📚 Tutorial Interactivo")
        self.setSubTitle("Aprende los conceptos básicos del trading algorítmico")

        layout = QVBoxLayout()

        # Tutorial content
        tutorial_text = QTextEdit()
        tutorial_text.setReadOnly(True)
        tutorial_text.setPlainText("""TRADING ALGORÍTMICO PARA PRINCIPIANTES
========================================

1. ¿QUÉ ES EL TRADING ALGORÍTMICO?
• Es usar computadoras para ejecutar operaciones automáticamente
• Sigue reglas predefinidas sin emociones humanas
• Puede operar 24/7 sin intervención manual

2. COMPONENTES PRINCIPALES:
• Estrategia: Las reglas que deciden cuándo comprar/vender
• Backtesting: Probar la estrategia con datos históricos
• Live Trading: Ejecutar la estrategia con dinero real
• Risk Management: Controlar pérdidas y ganancias

3. VENTAJAS:
• Elimina errores emocionales
• Mayor velocidad de ejecución
• Puede monitorear múltiples mercados
• Resultados consistentes y medibles

4. RIESGOS IMPORTANTES:
• El pasado no garantiza el futuro
• Siempre hay riesgo de pérdida de capital
• Las estrategias necesitan mantenimiento
• Nunca inviertas más de lo que puedas perder

¡RECUERDA! El trading algorítmico requiere:
• Conocimiento técnico
• Gestión adecuada del riesgo
• Pruebas exhaustivas
• Monitoreo continuo""")
        tutorial_text.setFont(QFont("Consolas", 9))
        layout.addWidget(tutorial_text)

        # Understanding checkbox
        self.understanding_check = QCheckBox(
            "✓ He leído y entiendo los conceptos básicos y riesgos del trading algorítmico"
        )
        self.understanding_check.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.understanding_check)

        self.setLayout(layout)

        # Register field for validation
        self.registerField("understanding*", self.understanding_check)

    def nextId(self):
        return 4  # Configuration page


class QuickSetupPage(QWizardPage):
    """Quick setup for experienced users"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("⚡ Configuración Rápida")
        self.setSubTitle("Para usuarios con experiencia en trading algorítmico")

        layout = QVBoxLayout()

        # Quick setup info
        info_label = QLabel(
            "Como usuario experimentado, puedes saltar el tutorial básico.\n\n"
            "Te recomendamos revisar la documentación avanzada en la pestaña 'Ayuda' "
            "una vez completada la configuración."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Skip tutorial checkbox
        self.skip_tutorial_check = QCheckBox(
            "✓ He usado plataformas de trading algorítmico anteriormente"
        )
        self.skip_tutorial_check.setChecked(True)
        layout.addWidget(self.skip_tutorial_check)

        layout.addStretch()

        self.setLayout(layout)

    def nextId(self):
        return 4  # Configuration page


class DemoSetupPage(QWizardPage):
    """Demo mode setup"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("🎮 Modo Demo")
        self.setSubTitle("Prueba la plataforma sin riesgos")

        layout = QVBoxLayout()

        # Demo explanation
        demo_label = QLabel(
            "El modo demo te permite:\n\n"
            "• Explorar todas las funcionalidades sin configurar APIs\n"
            "• Probar estrategias con datos históricos\n"
            "• Aprender a usar la plataforma paso a paso\n"
            "• Ver resultados sin arriesgar dinero real\n\n"
            "¡Perfecto para aprender y experimentar!"
        )
        demo_label.setWordWrap(True)
        layout.addWidget(demo_label)

        # Demo features
        features_list = QListWidget()
        features_list.setMaximumHeight(150)

        demo_features = [
            "✅ Datos históricos incluidos (BTC/USD)",
            "✅ Estrategias de ejemplo preconfiguradas",
            "✅ Backtesting completo disponible",
            "✅ Sin necesidad de configurar brokers",
            "✅ Tutoriales y ayuda integrada"
        ]

        for feature in demo_features:
            item = QListWidgetItem(feature)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)  # Make non-selectable
            features_list.addItem(item)

        layout.addWidget(features_list)

        # Start demo checkbox
        self.start_demo_check = QCheckBox(
            "✓ Quiero comenzar en modo demo"
        )
        self.start_demo_check.setChecked(True)
        layout.addWidget(self.start_demo_check)

        layout.addStretch()

        self.setLayout(layout)

        # Register field
        self.registerField("demo_mode*", self.start_demo_check)

    def nextId(self):
        return 5  # Demo configuration


class ConfigurationPage(QWizardPage):
    """Main configuration page"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("⚙️ Configuración Inicial")
        self.setSubTitle("Personaliza tu experiencia de trading")

        layout = QVBoxLayout()

        # Configuration sections
        self.create_data_config_section(layout)
        self.create_strategy_config_section(layout)
        self.create_risk_config_section(layout)

        self.setLayout(layout)

    def create_data_config_section(self, layout):
        """Create data configuration section"""
        data_group = QGroupBox("📊 Configuración de Datos")
        data_layout = QVBoxLayout()

        # Default data selection
        data_layout.addWidget(QLabel("Datos por defecto para comenzar:"))

        self.data_combo = QComboBox()
        self.data_combo.addItems([
            "BTC/USD - Bitcoin vs Dólar (Recomendado)",
            "ETH/USD - Ethereum vs Dólar",
            "EUR/USD - Euro vs Dólar (Forex)",
            "SPY - S&P 500 ETF (Acciones)"
        ])
        data_layout.addWidget(self.data_combo)

        # Auto-load data
        self.auto_load_check = QCheckBox("Cargar datos automáticamente al iniciar")
        self.auto_load_check.setChecked(True)
        data_layout.addWidget(self.auto_load_check)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

    def create_strategy_config_section(self, layout):
        """Create strategy configuration section"""
        strategy_group = QGroupBox("🎯 Estrategia Inicial")
        strategy_layout = QVBoxLayout()

        strategy_layout.addWidget(QLabel("Elige una estrategia para comenzar:"))

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Bollinger Bands - Seguir tendencias con bandas",
            "RSI Mean Reversion - Comprar oversold, vender overbought",
            "MACD Momentum - Seguir momentum del mercado",
            "MA Crossover - Cruce de medias móviles",
            "Volume Breakout - Rompimientos de volumen"
        ])
        strategy_layout.addWidget(self.strategy_combo)

        # Strategy info
        strategy_info = QLabel(
            "💡 Todas las estrategias incluyen:\n"
            "• Backtesting automático\n"
            "• Análisis de riesgo\n"
            "• Optimización de parámetros"
        )
        strategy_info.setWordWrap(True)
        strategy_layout.addWidget(strategy_info)

        strategy_group.setLayout(strategy_layout)
        layout.addWidget(strategy_group)

    def create_risk_config_section(self, layout):
        """Create risk management configuration section"""
        risk_group = QGroupBox("⚠️ Gestión de Riesgos")
        risk_layout = QVBoxLayout()

        # Risk level selection
        risk_layout.addWidget(QLabel("Nivel de riesgo inicial:"))

        self.risk_combo = QComboBox()
        self.risk_combo.addItems([
            "Conservador (1% riesgo por operación)",
            "Moderado (2% riesgo por operación)",
            "Agresivo (3% riesgo por operación)"
        ])
        self.risk_combo.setCurrentIndex(1)  # Moderate default
        risk_layout.addWidget(self.risk_combo)

        # Risk warning
        risk_warning = QLabel(
            "⚠️ IMPORTANTE: Nunca inviertas más de lo que puedas perder. "
            "Las estrategias de trading pueden generar pérdidas."
        )
        risk_warning.setWordWrap(True)
        risk_warning.setStyleSheet("color: #ff6b35; font-weight: bold;")
        risk_layout.addWidget(risk_warning)

        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)

    def nextId(self):
        return 6  # Final page


class DemoConfigPage(QWizardPage):
    """Demo-specific configuration"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("🎮 Configuración Demo")
        self.setSubTitle("Elige tus preferencias para el modo demo")

        layout = QVBoxLayout()

        # Demo preferences
        demo_group = QGroupBox("Preferencias de Demo")
        demo_layout = QVBoxLayout()

        self.tutorial_check = QCheckBox("Mostrar tutoriales interactivos")
        self.tutorial_check.setChecked(True)
        demo_layout.addWidget(self.tutorial_check)

        self.sample_data_check = QCheckBox("Cargar datos de ejemplo automáticamente")
        self.sample_data_check.setChecked(True)
        demo_layout.addWidget(self.sample_data_check)

        self.sample_strategies_check = QCheckBox("Incluir estrategias de ejemplo")
        self.sample_strategies_check.setChecked(True)
        demo_layout.addWidget(self.sample_strategies_check)

        demo_group.setLayout(demo_layout)
        layout.addWidget(demo_group)

        # Demo limitations info
        limitations_group = QGroupBox("ℹ️ Limitaciones del Modo Demo")
        limitations_layout = QVBoxLayout()

        limitations = [
            "• No se ejecutan operaciones reales",
            "• Datos limitados a períodos históricos",
            "• Sin conexión a brokers reales",
            "• Resultados simulados"
        ]

        for limitation in limitations:
            limitations_layout.addWidget(QLabel(limitation))

        limitations_group.setLayout(limitations_layout)
        layout.addWidget(limitations_group)

        layout.addStretch()

        self.setLayout(layout)

    def nextId(self):
        return 7  # Demo final page


class FinalPage(QWizardPage):
    """Final setup page"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("✅ ¡Todo Listo!")
        self.setSubTitle("Tu plataforma TradingIA está configurada")

        layout = QVBoxLayout()

        # Success message
        success_label = QLabel(
            "🎉 ¡Felicitaciones! Tu plataforma está lista para comenzar.\n\n"
            "Recuerda:\n"
            "• Las estrategias necesitan backtesting antes de usar en vivo\n"
            "• Monitorea siempre el rendimiento de tus estrategias\n"
            "• Ajusta los parámetros según las condiciones del mercado\n"
            "• Nunca inviertas más de lo que puedas perder"
        )
        success_label.setWordWrap(True)
        success_label.setFont(QFont("Arial", 11))
        layout.addWidget(success_label)

        # Quick start tips
        tips_group = QGroupBox("🚀 Próximos Pasos Recomendados")
        tips_layout = QVBoxLayout()

        tips = [
            "1. Ve a la pestaña 'Data' para explorar los datos",
            "2. Prueba una estrategia en la pestaña 'Backtest'",
            "3. Revisa los resultados en 'Results'",
            "4. Consulta la ayuda en cualquier momento con F1"
        ]

        for tip in tips:
            tips_layout.addWidget(QLabel(tip))

        tips_group.setLayout(tips_layout)
        layout.addWidget(tips_group)

        # Final checkbox
        self.ready_check = QCheckBox(
            "✓ Estoy listo para comenzar a usar TradingIA"
        )
        self.ready_check.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.ready_check)

        layout.addStretch()

        self.setLayout(layout)

        # Register field
        self.registerField("ready*", self.ready_check)


class DemoFinalPage(QWizardPage):
    """Demo final page"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("🎮 ¡Modo Demo Activado!")
        self.setSubTitle("Comienza tu viaje en el trading algorítmico")

        layout = QVBoxLayout()

        # Demo success message
        demo_success_label = QLabel(
            "🎉 ¡Bienvenido al modo demo de TradingIA!\n\n"
            "Este modo te permite explorar todas las funcionalidades sin riesgos:\n\n"
            "• Datos históricos precargados\n"
            "• Estrategias de ejemplo listas para usar\n"
            "• Backtesting completo disponible\n"
            "• Tutoriales y ayuda integrada\n\n"
            "Cuando estés listo para trading real, puedes configurar APIs "
            "y brokers desde la pestaña 'Configuración'."
        )
        demo_success_label.setWordWrap(True)
        demo_success_label.setFont(QFont("Arial", 11))
        layout.addWidget(demo_success_label)

        # Demo features highlight
        features_group = QGroupBox("✨ Lo que puedes hacer ahora:")
        features_layout = QVBoxLayout()

        features = [
            "• Explorar datos históricos de BTC/USD",
            "• Probar estrategias preconfiguradas",
            "• Ejecutar backtests completos",
            "• Ver análisis detallados de rendimiento",
            "• Aprender sin arriesgar capital real"
        ]

        for feature in features:
            feature_label = QLabel(feature)
            features_layout.addWidget(feature_label)

        features_group.setLayout(features_layout)
        layout.addWidget(features_group)

        # Start demo checkbox
        self.start_demo_check = QCheckBox(
            "✓ Comenzar explorando la plataforma"
        )
        self.start_demo_check.setChecked(True)
        self.start_demo_check.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.start_demo_check)

        layout.addStretch()

        self.setLayout(layout)

        # Register field
        self.registerField("start_demo*", self.start_demo_check)


class OnboardingWizard(QWizard):
    """Main onboarding wizard"""

    # Signals
    onboarding_completed = Signal(dict)  # config_dict

    def __init__(self, parent=None):
        super().__init__(parent)

        # Wizard properties
        self.setWindowTitle("🚀 TradingIA - Configuración Inicial")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoCancelButton, False)
        self.setOption(QWizard.WizardOption.NoDefaultButton, False)

        # Set window properties
        self.setMinimumSize(700, 600)
        self.setMaximumSize(800, 700)

        # Apply modern styling
        self.setStyleSheet("""
            QWizard {
                background-color: #f8f9fa;
            }
            QWizardPage {
                background-color: white;
            }
            QLabel {
                color: #2c3e50;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QCheckBox {
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                min-width: 200px;
            }
        """)

        # Add pages
        self.addPage(WelcomePage())
        self.addPage(TutorialPage())
        self.addPage(QuickSetupPage())
        self.addPage(DemoSetupPage())
        self.addPage(ConfigurationPage())
        self.addPage(DemoConfigPage())
        self.addPage(FinalPage())
        self.addPage(DemoFinalPage())

        # Set start page
        self.setStartId(0)

        # Connect signals
        self.finished.connect(self.on_wizard_finished)

    def on_wizard_finished(self):
        """Handle wizard completion"""
        if self.result() == QWizard.DialogCode.Accepted:
            # Collect configuration
            config = self.collect_configuration()

            # Save configuration
            self.save_configuration(config)

            # Emit completion signal
            self.onboarding_completed.emit(config)
        else:
            # User cancelled - use defaults
            default_config = self.get_default_config()
            self.onboarding_completed.emit(default_config)

    def collect_configuration(self):
        """Collect configuration from wizard pages"""
        config = {
            'first_run': False,
            'user_type': 'unknown',
            'demo_mode': False,
            'tutorial_completed': False,
            'data_config': {},
            'strategy_config': {},
            'risk_config': {}
        }

        # Determine user type and demo mode
        welcome_page = self.page(0)  # WelcomePage
        if hasattr(welcome_page, 'newbie_radio') and welcome_page.newbie_radio.isChecked():
            config['user_type'] = 'newbie'
            config['tutorial_completed'] = True
        elif hasattr(welcome_page, 'experienced_radio') and welcome_page.experienced_radio.isChecked():
            config['user_type'] = 'experienced'
        elif hasattr(welcome_page, 'demo_radio') and welcome_page.demo_radio.isChecked():
            config['user_type'] = 'demo'
            config['demo_mode'] = True

        # Collect configuration data
        if not config['demo_mode']:
            config_page = self.page(4)  # ConfigurationPage

            # Data config
            data_options = ["BTC/USD", "ETH/USD", "EUR/USD", "SPY"]
            if hasattr(config_page, 'data_combo'):
                config['data_config'] = {
                    'default_symbol': data_options[config_page.data_combo.currentIndex()],
                    'auto_load': config_page.auto_load_check.isChecked() if hasattr(config_page, 'auto_load_check') else True
                }

            # Strategy config
            strategy_options = ["bollinger_bands", "rsi_mean_reversion", "macd_momentum",
                              "ma_crossover", "volume_breakout"]
            if hasattr(config_page, 'strategy_combo'):
                config['strategy_config'] = {
                    'default_strategy': strategy_options[config_page.strategy_combo.currentIndex()]
                }

            # Risk config
            risk_options = [0.01, 0.02, 0.03]  # 1%, 2%, 3%
            if hasattr(config_page, 'risk_combo'):
                config['risk_config'] = {
                    'default_risk_per_trade': risk_options[config_page.risk_combo.currentIndex()]
                }
        else:
            # Demo configuration
            demo_config_page = self.page(5)  # DemoConfigPage
            if hasattr(demo_config_page, 'tutorial_check'):
                config['demo_config'] = {
                    'show_tutorials': demo_config_page.tutorial_check.isChecked(),
                    'load_sample_data': demo_config_page.sample_data_check.isChecked() if hasattr(demo_config_page, 'sample_data_check') else True,
                    'include_sample_strategies': demo_config_page.sample_strategies_check.isChecked() if hasattr(demo_config_page, 'sample_strategies_check') else True
                }

        return config

    def save_configuration(self, config):
        """Save configuration to user config manager"""
        try:
            config_manager = UserConfigManager()

            # Save onboarding completion
            config_manager.config['onboarding_completed'] = True
            config_manager.config['user_type'] = config.get('user_type', 'unknown')
            config_manager.config['demo_mode'] = config.get('demo_mode', False)

            # Save data config
            if 'data_config' in config:
                config_manager.config['default_data'] = config['data_config']

            # Save strategy config
            if 'strategy_config' in config:
                config_manager.config['default_strategy'] = config['strategy_config']

            # Save risk config
            if 'risk_config' in config:
                config_manager.config['risk_management'] = config['risk_config']

            # Save demo config
            if 'demo_config' in config:
                config_manager.config['demo_settings'] = config['demo_config']

            # Save to file
            config_manager.save_config()

        except Exception as e:
            print(f"Error saving onboarding configuration: {e}")

    def get_default_config(self):
        """Get default configuration for cancelled wizard"""
        return {
            'first_run': False,
            'user_type': 'unknown',
            'demo_mode': True,  # Default to demo mode
            'tutorial_completed': False,
            'data_config': {
                'default_symbol': 'BTC/USD',
                'auto_load': True
            },
            'strategy_config': {
                'default_strategy': 'bollinger_bands'
            },
            'risk_config': {
                'default_risk_per_trade': 0.02
            }
        }


def should_show_onboarding():
    """Check if onboarding wizard should be shown"""
    try:
        config_manager = UserConfigManager()
        onboarding_completed = config_manager.config.get('onboarding_completed', False)

        # Also check if it's been more than 30 days since last use
        # (for users who might need a refresher)

        return not onboarding_completed
    except Exception as e:
        print(f"Error checking onboarding status: {e}")
        return True  # Show onboarding if we can't check


def run_onboarding_wizard(parent=None):
    """Run the onboarding wizard and return configuration"""
    if not should_show_onboarding():
        return None  # Skip onboarding

    wizard = OnboardingWizard(parent)

    # Run wizard
    result = wizard.exec()

    if result == QWizard.DialogCode.Accepted:
        # Configuration will be emitted via signal
        return wizard
    else:
        return None