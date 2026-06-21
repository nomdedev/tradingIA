class DarkTheme:
    """
    Modern Dark Theme for TradingIA Platform.
    Inspired by VS Code Dark+ and modern financial dashboards.
    """
    
    # Color Palette
    BG_PRIMARY = "#1e1e1e"
    BG_SECONDARY = "#252526"
    BG_TERTIARY = "#2d2d2d"
    BG_HOVER = "#3e3e3e"
    BG_SELECTED = "#37373d"
    
    TEXT_PRIMARY = "#cccccc"
    TEXT_SECONDARY = "#969696"
    TEXT_HIGHLIGHT = "#ffffff"
    
    ACCENT_PRIMARY = "#007acc"
    ACCENT_HOVER = "#0098ff"
    ACCENT_PRESSED = "#005a9e"
    
    BORDER_COLOR = "#454545"
    
    SUCCESS = "#4ec9b0"
    WARNING = "#cca700"
    ERROR = "#f48771"
    INFO = "#569cd6"

    @staticmethod
    def get_stylesheet():
        return f"""
            /* GLOBAL RESET */
            * {{
                outline: none;
            }}

            QMainWindow, QDialog {{
                background-color: {DarkTheme.BG_PRIMARY};
                color: {DarkTheme.TEXT_PRIMARY};
            }}

            QWidget {{
                background-color: {DarkTheme.BG_PRIMARY};
                color: {DarkTheme.TEXT_PRIMARY};
                font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
                font-size: 14px;
            }}

            /* TAB WIDGET */
            QTabWidget::pane {{
                border: 1px solid {DarkTheme.BORDER_COLOR};
                background-color: {DarkTheme.BG_PRIMARY};
                border-radius: 4px;
                top: -1px; 
            }}

            QTabWidget::tab-bar {{
                alignment: left;
            }}

            QTabBar::tab {{
                background-color: {DarkTheme.BG_SECONDARY};
                color: {DarkTheme.TEXT_SECONDARY};
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border: 1px solid transparent;
                font-weight: 500;
            }}

            QTabBar::tab:selected {{
                background-color: {DarkTheme.BG_PRIMARY};
                color: {DarkTheme.TEXT_HIGHLIGHT};
                border-top: 2px solid {DarkTheme.ACCENT_PRIMARY};
                border-bottom: 1px solid {DarkTheme.BG_PRIMARY}; /* Blend with pane */
            }}

            QTabBar::tab:hover:!selected {{
                background-color: {DarkTheme.BG_TERTIARY};
                color: {DarkTheme.TEXT_PRIMARY};
            }}

            /* BUTTONS */
            QPushButton {{
                background-color: {DarkTheme.ACCENT_PRIMARY};
                color: {DarkTheme.TEXT_HIGHLIGHT};
                border: 1px solid {DarkTheme.ACCENT_PRIMARY};
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: 600;
                min-height: 24px;
            }}

            QPushButton:hover {{
                background-color: {DarkTheme.ACCENT_HOVER};
                border-color: {DarkTheme.ACCENT_HOVER};
            }}

            QPushButton:pressed {{
                background-color: {DarkTheme.ACCENT_PRESSED};
                border-color: {DarkTheme.ACCENT_PRESSED};
            }}

            QPushButton:disabled {{
                background-color: {DarkTheme.BG_TERTIARY};
                border-color: {DarkTheme.BORDER_COLOR};
                color: {DarkTheme.TEXT_SECONDARY};
            }}
            
            /* Secondary Button Style (use objectName="secondary") */
            QPushButton#secondary {{
                background-color: {DarkTheme.BG_TERTIARY};
                border: 1px solid {DarkTheme.BORDER_COLOR};
                color: {DarkTheme.TEXT_PRIMARY};
            }}
            
            QPushButton#secondary:hover {{
                background-color: {DarkTheme.BG_HOVER};
                border-color: {DarkTheme.TEXT_SECONDARY};
            }}

            /* INPUTS */
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {DarkTheme.BG_SECONDARY};
                color: {DarkTheme.TEXT_PRIMARY};
                border: 1px solid {DarkTheme.BORDER_COLOR};
                border-radius: 4px;
                padding: 6px;
                selection-background-color: {DarkTheme.ACCENT_PRIMARY};
            }}

            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {DarkTheme.ACCENT_PRIMARY};
                background-color: {DarkTheme.BG_PRIMARY};
            }}

            /* COMBO BOX */
            QComboBox {{
                background-color: {DarkTheme.BG_SECONDARY};
                color: {DarkTheme.TEXT_PRIMARY};
                border: 1px solid {DarkTheme.BORDER_COLOR};
                border-radius: 4px;
                padding: 6px;
                min-width: 6em;
            }}

            QComboBox:hover {{
                border-color: {DarkTheme.TEXT_SECONDARY};
            }}

            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 0px;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }}
            
            QComboBox QAbstractItemView {{
                background-color: {DarkTheme.BG_SECONDARY};
                color: {DarkTheme.TEXT_PRIMARY};
                border: 1px solid {DarkTheme.BORDER_COLOR};
                selection-background-color: {DarkTheme.BG_SELECTED};
            }}

            /* GROUP BOX */
            QGroupBox {{
                border: 1px solid {DarkTheme.BORDER_COLOR};
                border-radius: 6px;
                margin-top: 1.5em;
                padding-top: 10px;
                font-weight: bold;
                color: {DarkTheme.TEXT_HIGHLIGHT};
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                background-color: {DarkTheme.BG_PRIMARY}; /* Match parent bg */
            }}

            /* TABLES */
            QTableWidget, QTableView {{
                background-color: {DarkTheme.BG_PRIMARY};
                gridline-color: {DarkTheme.BORDER_COLOR};
                border: 1px solid {DarkTheme.BORDER_COLOR};
                border-radius: 4px;
                selection-background-color: {DarkTheme.BG_SELECTED};
                selection-color: {DarkTheme.TEXT_HIGHLIGHT};
            }}

            QHeaderView::section {{
                background-color: {DarkTheme.BG_SECONDARY};
                color: {DarkTheme.TEXT_PRIMARY};
                padding: 6px;
                border: none;
                border-right: 1px solid {DarkTheme.BG_PRIMARY};
                border-bottom: 1px solid {DarkTheme.BORDER_COLOR};
                font-weight: 600;
            }}
            
            QTableWidget::item {{
                padding: 4px;
            }}

            /* SCROLL BARS */
            QScrollBar:vertical {{
                border: none;
                background: {DarkTheme.BG_PRIMARY};
                width: 10px;
                margin: 0px;
            }}

            QScrollBar::handle:vertical {{
                background: {DarkTheme.BG_HOVER};
                min-height: 20px;
                border-radius: 5px;
            }}

            QScrollBar::handle:vertical:hover {{
                background: {DarkTheme.TEXT_SECONDARY};
            }}

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            
            QScrollBar:horizontal {{
                border: none;
                background: {DarkTheme.BG_PRIMARY};
                height: 10px;
                margin: 0px;
            }}

            QScrollBar::handle:horizontal {{
                background: {DarkTheme.BG_HOVER};
                min-width: 20px;
                border-radius: 5px;
            }}

            QScrollBar::handle:horizontal:hover {{
                background: {DarkTheme.TEXT_SECONDARY};
            }}

            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}

            /* SPLITTER */
            QSplitter::handle {{
                background-color: {DarkTheme.BG_TERTIARY};
            }}
            
            QSplitter::handle:hover {{
                background-color: {DarkTheme.ACCENT_PRIMARY};
            }}

            /* STATUS BAR */
            QStatusBar {{
                background-color: {DarkTheme.ACCENT_PRIMARY};
                color: {DarkTheme.TEXT_HIGHLIGHT};
            }}
            
            /* TOOLTIPS */
            QToolTip {{
                background-color: {DarkTheme.BG_SECONDARY};
                color: {DarkTheme.TEXT_PRIMARY};
                border: 1px solid {DarkTheme.BORDER_COLOR};
                padding: 4px;
            }}
            
            /* CUSTOM CLASSES */
            .card {{
                background-color: {DarkTheme.BG_SECONDARY};
                border: 1px solid {DarkTheme.BORDER_COLOR};
                border-radius: 8px;
            }}
            
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: {DarkTheme.ACCENT_PRIMARY};
            }}
            
            .metric-label {{
                font-size: 12px;
                color: {DarkTheme.TEXT_SECONDARY};
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
        """
