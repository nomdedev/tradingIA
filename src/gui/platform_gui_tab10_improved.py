"""
TradingIA Platform - Tab 10: Help & Documentation (Improved)
Integrated help system and user manual
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem, QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWebEngineWidgets import QWebEngineView
import markdown
import os
from src.gui.styles import DarkTheme


class Tab10Help(QWidget):
    """
    Help & Documentation Tab (Improved)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        """Initialize the help interface"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(
            f"""
            QSplitter::handle {{
                background-color: {DarkTheme.BG_HOVER};
            }}
        """
        )

        # Left panel - Navigation
        self.nav_panel = self.create_navigation_panel()
        splitter.addWidget(self.nav_panel)

        # Right panel - Content
        self.content_browser = self.create_content_browser()
        splitter.addWidget(self.content_browser)

        # Set splitter proportions
        splitter.setStretchFactor(0, 25)
        splitter.setStretchFactor(1, 75)
        splitter.setSizes([300, 900])
        splitter.setCollapsible(0, False)

        layout.addWidget(splitter)

        # Load initial content
        self.show_welcome()

    def create_navigation_panel(self):
        """Create navigation panel with tree"""
        panel = QFrame()
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(400)
        panel.setStyleSheet(f"background-color: {DarkTheme.BG_SECONDARY}; border-right: 1px solid {DarkTheme.BG_HOVER};")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Tree
        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        tree.setStyleSheet(
            f"""
            QTreeWidget {{
                background-color: {DarkTheme.BG_SECONDARY};
                color: {DarkTheme.TEXT_PRIMARY};
                border: none;
                font-size: 13px;
                padding: 10px;
            }}
            QTreeWidget::item {{
                padding: 6px;
                border-radius: 4px;
                margin-bottom: 2px;
            }}
            QTreeWidget::item:selected {{
                background-color: {DarkTheme.BG_SELECTED};
                color: {DarkTheme.TEXT_HIGHLIGHT};
            }}
            QTreeWidget::item:hover {{
                background-color: {DarkTheme.BG_TERTIARY};
            }}
            QTreeView::branch:has-children:!has-siblings:closed,
            QTreeView::branch:closed:has-children:has-siblings {{
                border-image: none;
                image: url(:/icons/chevron-right.svg);
            }}
            QTreeView::branch:open:has-children:!has-siblings,
            QTreeView::branch:open:has-children:has-siblings {{
                border-image: none;
                image: url(:/icons/chevron-down.svg);
            }}
        """
        )

        # Connect signal
        tree.itemClicked.connect(self.on_topic_selected)

        # Create main categories
        self.create_help_structure(tree)

        tree.expandAll()
        layout.addWidget(tree)
        
        return panel

    def create_help_structure(self, tree):
        """Create the hierarchical help structure"""
        categories = {
            "🚀 Quick Start": [
                "Welcome to TradingIA",
                "Getting Started",
                "Initial Setup",
                "Auto Data Loading",
            ],
            "📊 Dashboard": ["Overview", "System Metrics", "Quick Actions", "System Status"],
            "📥 Data Management": [
                "Downloading Data",
                "Supported Formats",
                "Storage",
                "Integrity Check",
            ],
            "⚙️ Strategies": ["Configuration", "Parameters", "Optimization", "Backtesting"],
            "▶️ Backtesting": [
                "Running Backtests",
                "Analyzing Results",
                "Performance Metrics",
                "Validation",
            ],
            "📈 Results Analysis": [
                "Performance Charts",
                "Detailed Statistics",
                "Strategy Comparison",
                "Exporting Reports",
            ],
            "🆚 A/B Testing": [
                "Test Configuration",
                "Automated Execution",
                "Statistical Analysis",
                "Recommendations",
            ],
            "📊 Live Monitor": [
                "Paper Trading",
                "Alpaca Connection",
                "Real-time Monitoring",
                "Alerts & Notifications",
            ],
            "🔬 Advanced Analysis": [
                "Technical Analysis",
                "Machine Learning",
                "Risk Management",
                "Risk Management Guide",  # NEW
                "Advanced Optimization",
            ],
            "🔧 Settings": [
                "System Settings",
                "User Preferences",
                "API Configuration",
                "Backup & Restore",
            ],
            "❓ Troubleshooting": [
                "Common Issues",
                "Error Messages",
                "Performance Issues",
                "Support",
            ],
        }

        for category, topics in categories.items():
            category_item = QTreeWidgetItem([category])
            category_item.setExpanded(True)
            
            # Style category items
            font = category_item.font(0)
            font.setBold(True)
            category_item.setFont(0, font)
            category_item.setForeground(0, QColor("#ffffff"))

            for topic in topics:
                topic_item = QTreeWidgetItem([topic])
                category_item.addChild(topic_item)

            tree.addTopLevelItem(category_item)

    def create_content_browser(self):
        """Create content display browser"""
        browser = QWebEngineView()
        browser.setStyleSheet("background-color: #1e1e1e;")
        return browser

    def on_topic_selected(self, item, column):
        """Handle topic selection"""
        topic_text = item.text(column)
        
        # Map topics to content methods (simplified mapping)
        # In a real app, this would be more robust
        if topic_text == "Welcome to TradingIA":
            self.show_welcome()
        elif topic_text == "Getting Started":
            self.show_getting_started()
        elif topic_text == "Risk Management Guide":
            self.show_risk_guide()
        # ... add other mappings as needed ...
        else:
            # Default content for unmapped topics
            self.show_placeholder(topic_text)

    def _get_css_style(self):
        """Get common CSS style for help content"""
        return """
        <style>
            body { 
                color: #cccccc; 
                font-family: 'Segoe UI', Arial, sans-serif; 
                font-size: 14px; 
                background-color: #1e1e1e; 
                margin: 20px;
            }
            h1 { color: #4ec9b0; font-size: 24px; margin-bottom: 15px; font-weight: bold; }
            h2 { color: #569cd6; font-size: 18px; margin-top: 20px; margin-bottom: 10px; font-weight: bold; }
            h3 { color: #dcdcaa; font-size: 16px; margin-top: 15px; margin-bottom: 8px; font-weight: bold; }
            p { margin-bottom: 10px; line-height: 1.5; }
            li { margin-bottom: 5px; }
            b { color: #ffffff; font-weight: bold; }
            i { color: #9cdcfe; font-style: italic; }
            hr { border: 1px solid #3e3e3e; margin: 20px 0; }
            a { color: #3794ff; text-decoration: none; }
            code { background-color: #2d2d2d; padding: 2px 4px; border-radius: 3px; font-family: Consolas, monospace; }
            pre { background-color: #2d2d2d; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; border: 1px solid #3e3e3e; }
        </style>
        """

    def show_welcome(self):
        """Show welcome message"""
        html = self._get_css_style() + """
        <h1 style='color: #4ec9b0; font-size: 28px;'>Welcome to TradingIA Platform</h1>
        <hr style='border-color: #3e3e3e;'>
        <p>TradingIA is a comprehensive algorithmic trading platform designed for developing, testing, and deploying automated trading strategies.</p>
        
        <h2 style='color: #569cd6;'>Key Features</h2>
        <ul>
            <li><b>Advanced Backtesting:</b> Test strategies with historical data</li>
            <li><b>Strategy Optimization:</b> Fine-tune parameters for better performance</li>
            <li><b>Live Monitoring:</b> Real-time tracking of paper trading</li>
            <li><b>Risk Management:</b> Integrated risk metrics and controls</li>
            <li><b>A/B Testing:</b> Compare strategies head-to-head</li>
        </ul>
        
        <h2 style='color: #569cd6;'>Getting Started</h2>
        <p>Select a topic from the navigation menu on the left to learn more about specific features.</p>
        """
        self.content_browser.setHtml(html)

    def show_getting_started(self):
        """Show getting started guide"""
        html = self._get_css_style() + """
        <h1 style='color: #4ec9b0;'>Getting Started</h1>
        <hr style='border-color: #3e3e3e;'>
        <p>Follow these steps to start using the platform:</p>
        
        <ol>
            <li><b>Load Data:</b> Go to the <i>Data Management</i> tab to download or load historical data.</li>
            <li><b>Configure Strategy:</b> Visit the <i>Strategy</i> tab to select and configure your trading strategy.</li>
            <li><b>Run Backtest:</b> Use the <i>Backtest</i> tab to test your strategy against historical data.</li>
            <li><b>Analyze Results:</b> Review performance metrics in the <i>Results Analysis</i> tab.</li>
        </ol>
        """
        self.content_browser.setHtml(html)

    def show_risk_guide(self):
        """Show the Risk Management Guide from markdown file"""
        try:
            # Path to docs
            docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'RISK_MANAGEMENT_GUIDE.md'))
            
            if os.path.exists(docs_path):
                with open(docs_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                
                # Convert markdown to HTML
                html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
                
                # Wrap in style
                full_html = self._get_css_style() + f"""
                <div class="markdown-body">
                    {html_content}
                </div>
                """
                self.content_browser.setHtml(full_html)
            else:
                self.show_placeholder("Risk Management Guide (File not found)")
        except Exception as e:
            self.show_placeholder(f"Error loading guide: {e}")

    def show_placeholder(self, topic):
        """Show placeholder for unimplemented topics"""
        html = self._get_css_style() + f"""
        <h1 style='color: #4ec9b0;'>{topic}</h1>
        <hr style='border-color: #3e3e3e;'>
        <p>Documentation for <b>{topic}</b> is coming soon.</p>
        """
        self.content_browser.setHtml(html)

from PySide6.QtGui import QColor
