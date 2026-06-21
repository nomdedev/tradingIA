import sys
import os
import logging
import time
from PySide6.QtWidgets import QApplication, QTabWidget, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QTableWidget
from PySide6.QtCore import Qt, QTimer

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import GUI classes
from src.gui.platform_gui_tab1_improved import Tab1DataManagement
from src.gui.platform_gui_tab2_improved import Tab2StrategyConfig
from src.gui.platform_gui_tab3_improved import Tab3BacktestRunner
from src.gui.platform_gui_tab6_improved import Tab6LiveMonitor
from src.gui.platform_gui_tab11_improved import Tab11RiskMetrics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UI_Audit")
print(f"DEBUG: Python executable: {sys.executable}")


class MockPlatform(QWidget):
    def __init__(self):
        super().__init__()
        self.data_dict = {}
        self.config_dict = {}
        self.last_backtest_results = {}
        self.session_logger = logging.getLogger("Session")
        self.data_manager = None
        self.strategy_engine = None
        self.backtester = None
        self.analysis_engines = None
        self.live_monitor = None
        self.settings = None
        self.reporters = None
        self.broker_manager = None
        self.api_app = None
        self.BROKERS_AVAILABLE = False
        self.API_AVAILABLE = False

    def update_status(self, msg, level="info"):
        logger.info(f"STATUS UPDATE: {msg} ({level})")

def audit_widget_tree(widget, indent=0):
    """Recursively audit widget tree for common UI issues"""
    issues = []
    prefix = "  " * indent
    
    # Check 1: Hardcoded colors in styleSheet (simple check)
    if hasattr(widget, "styleSheet") and widget.styleSheet():
        style = widget.styleSheet()
        if "#" in style and "DarkTheme" not in style:
            # This is a heuristic, not a hard error, but worth noting for "Expert Review"
            pass 

    # Check 2: Widgets without layouts (if container)
    if isinstance(widget, QWidget) and widget.children():
        has_layout = widget.layout() is not None
        # Some widgets like QLabel have children but no layout, which is fine.
        # But a QFrame or QGroupBox usually should have a layout.
        if not has_layout and isinstance(widget, (QTabWidget,)):
             issues.append(f"{prefix}⚠️ Container {widget.__class__.__name__} has no layout!")

    # Check 3: Default font usage (hard to check programmatically without context)
    
    # Recurse
    for child in widget.children():
        if isinstance(child, QWidget):
            issues.extend(audit_widget_tree(child, indent + 1))
            
    return issues

def run_audit():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    mock_platform = MockPlatform()
    
    report = []
    report.append("# UI/UX Expert Council Audit Report\n")
    report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    tabs_to_audit = [
        ("Tab 1: Data Management", Tab1DataManagement),
        ("Tab 2: Strategy Config", Tab2StrategyConfig),
        ("Tab 3: Backtest Runner", Tab3BacktestRunner),
        ("Tab 6: Live Monitor", Tab6LiveMonitor),
        ("Tab 11: Risk Metrics", Tab11RiskMetrics)
    ]

    for name, cls in tabs_to_audit:
        logger.info(f"Auditing {name}...")
        report.append(f"## {name}\n")
        
        try:
            # Instantiate
            if cls == Tab2StrategyConfig:
                tab = cls(mock_platform, None)
            elif cls == Tab3BacktestRunner:
                tab = cls(mock_platform, None)
            elif cls == Tab6LiveMonitor:
                tab = cls(mock_platform, None)
                print(f"DEBUG: Tab6 layout after init: {tab.layout()}")
            else:
                tab = cls(mock_platform)
            
            # 1. Visual Inspection (Simulated)
            report.append("### Visual Structure Analysis")
            
            # Check layout type
            layout = tab.layout()
            if not layout:
                report.append("- ❌ **CRITICAL**: Root widget has no layout.")
            else:
                report.append(f"- ✅ Root layout: {layout.__class__.__name__}")
                
            # Count widgets
            children = tab.findChildren(QWidget)
            report.append(f"- ℹ️ Total widgets: {len(children)}")
            
            # Check for specific "bad" widgets
            raw_labels = [c for c in children if isinstance(c, QLabel) and not c.styleSheet()]
            if raw_labels:
                report.append(f"- ⚠️ **Warning**: {len(raw_labels)} QLabels have no custom style (might look default/ugly).")
                
            # Check for overcrowding
            if len(children) > 50:
                report.append("- ⚠️ **UX Warning**: High widget density. Consider splitting into sub-tabs or collapsing groups.")

            # 2. Functional Smoke Test
            report.append("\n### Functional Smoke Test")
            try:
                # Try to find key interaction points
                buttons = tab.findChildren(QPushButton)
                combos = tab.findChildren(QComboBox)
                
                report.append(f"- Found {len(buttons)} buttons and {len(combos)} dropdowns.")
                
                if not buttons and not combos:
                     report.append("- ⚠️ **UX Warning**: No interactive elements found at top level.")
                
                report.append("- ✅ Instantiation successful.")
                
            except Exception as e:
                report.append(f"- ❌ **Functional Error**: {str(e)}")

        except Exception as e:
            report.append(f"- ❌ **CRITICAL FAILURE**: Could not instantiate tab. Error: {str(e)}")
            logger.error(f"Failed to audit {name}: {e}")
        
        report.append("\n---\n")

    # Write report
    with open("docs/UI_UX_AUDIT_REPORT.md", "w", encoding="utf-8") as f:
        f.writelines(report)
    
    logger.info("Audit complete. Report saved to docs/UI_UX_AUDIT_REPORT.md")

if __name__ == "__main__":
    run_audit()
