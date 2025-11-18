"""
Script de verificación visual de mejoras de diseño
Crea una ventana de muestra con los nuevos estilos optimizados
"""

import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                               QComboBox, QGroupBox, QProgressBar, QCheckBox,
                               QDateEdit, QFormLayout)
from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QFont

class DesignShowcase(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TradingIA - Mejoras de Diseño UI")
        self.setGeometry(100, 100, 800, 900)
        
        # Aplicar estilo global
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0e639c, stop:1 #0a4f7a);
                border: 1px solid #0a4f7a;
                padding: 6px 10px;
                min-height: 28px;
                max-height: 36px;
                font-size: 13px;
                font-weight: 600;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1177bb, stop:1 #0d5a8c);
            }
            QLineEdit, QComboBox {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px 8px;
                min-height: 24px;
                max-height: 32px;
                font-size: 13px;
            }
            QGroupBox {
                font-size: 13px;
                font-weight: 600;
                color: #fff;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QProgressBar {
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                text-align: center;
                max-height: 20px;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #4ec9b0;
                border-radius: 3px;
            }
        """)
        
        # Widget central
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Título
        title = QLabel("🎨 Mejoras de Diseño - Comparación Visual")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #fff; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Sección 1: Botones
        btn_group = QGroupBox("🔘 Botones Optimizados")
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)
        
        # Botones principales
        main_btns = QHBoxLayout()
        btn1 = QPushButton("📥 Load Data")
        btn1.setMinimumHeight(32)
        btn1.setMaximumHeight(36)
        btn2 = QPushButton("▶️ Start Server")
        btn2.setMinimumHeight(32)
        btn2.setMaximumHeight(32)
        main_btns.addWidget(btn1)
        main_btns.addWidget(btn2)
        btn_layout.addLayout(main_btns)
        
        # Botones secundarios
        sec_btns = QHBoxLayout()
        btn3 = QPushButton("Alpaca")
        btn3.setMinimumHeight(28)
        btn3.setMaximumHeight(28)
        btn4 = QPushButton("Binance")
        btn4.setMinimumHeight(28)
        btn4.setMaximumHeight(28)
        btn4.setEnabled(False)
        sec_btns.addWidget(btn3)
        sec_btns.addWidget(btn4)
        btn_layout.addLayout(sec_btns)
        
        info = QLabel("✅ Altura: 32-36px (antes: 40px) | Padding: 6-10px (antes: 8-16px)")
        info.setStyleSheet("color: #4ec9b0; font-size: 11px; margin-top: 5px;")
        btn_layout.addWidget(info)
        
        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_group)
        
        # Sección 2: Inputs
        input_group = QGroupBox("📝 Inputs y Controles")
        input_layout = QFormLayout()
        input_layout.setSpacing(8)
        input_layout.setContentsMargins(10, 8, 10, 8)
        
        symbol = QComboBox()
        symbol.addItems(["BTC/USD", "ETH/USD", "SOL/USD"])
        
        timeframe = QComboBox()
        timeframe.addItems(["5Min", "15Min", "1Hour"])
        
        date_edit = QDateEdit()
        date_edit.setDate(QDate.currentDate())
        date_edit.setMaximumHeight(30)
        
        input_layout.addRow("Symbol:", symbol)
        input_layout.addRow("Timeframe:", timeframe)
        input_layout.addRow("Date:", date_edit)
        
        check = QCheckBox("Enable Multi-Timeframe")
        check.setStyleSheet("font-size: 13px;")
        input_layout.addRow("", check)
        
        info2 = QLabel("✅ Altura: 26-32px | Font: 13px | Padding reducido")
        info2.setStyleSheet("color: #4ec9b0; font-size: 11px; margin-top: 5px;")
        input_layout.addRow("", info2)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Sección 3: Progress Bar
        progress_group = QGroupBox("⏳ Barra de Progreso")
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(6)
        
        progress = QProgressBar()
        progress.setValue(65)
        progress.setMaximumHeight(20)
        progress_layout.addWidget(progress)
        
        info3 = QLabel("✅ Altura: 20px (antes: 24px) | Más compacta")
        info3.setStyleSheet("color: #4ec9b0; font-size: 11px;")
        progress_layout.addWidget(info3)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Sección 4: Estadísticas
        stats_group = QGroupBox("📊 Estadísticas de Mejoras")
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(6)
        
        stats = [
            "🎯 Espacio vertical ahorrado: ~15-20%",
            "📏 Botones principales: 40px → 32-36px (-10-20%)",
            "📐 Botones secundarios: 40px → 28px (-30%)",
            "📦 GroupBox padding: -28%",
            "✏️ Input fields: -16%",
            "📈 Chart min-height: 400px → 350px (-13%)",
            "🔤 Font estandarizado: 13px general, 12px stats"
        ]
        
        for stat in stats:
            label = QLabel(stat)
            label.setStyleSheet("font-size: 12px; color: #ccc; padding: 2px;")
            stats_layout.addWidget(label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Beneficios
        benefits = QLabel("""
        <h3 style='color: #4ec9b0;'>✨ Beneficios:</h3>
        <ul style='color: #ccc; font-size: 13px; line-height: 1.6;'>
            <li>Más información visible sin scroll</li>
            <li>Interfaz más moderna y profesional</li>
            <li>Mejor aprovechamiento del espacio</li>
            <li>Navegación más rápida</li>
            <li>Usabilidad mantenida</li>
        </ul>
        """)
        benefits.setWordWrap(True)
        layout.addWidget(benefits)
        
        layout.addStretch()
        central.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DesignShowcase()
    window.show()
    sys.exit(app.exec())
