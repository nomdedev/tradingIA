"""
Trading IA GUI Application Launcher
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(project_root / 'logs' / 'trading_ia_gui.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if required dependencies are installed"""
    import importlib.util

    required_packages = ['PySide6', 'pandas', 'numpy', 'matplotlib']
    missing_packages = []

    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing dependencies: {', '.join(missing_packages)}")
        print("Please install required packages:")
        print("pip install PySide6 pandas numpy matplotlib")
        return False

    return True

def main():
    """Main application entry point"""
    print("Starting Trading IA GUI Application...")

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    try:
        # Import and run GUI
        from PySide6.QtWidgets import QApplication
        from core.ui.main_window import MainWindow

        # Create Qt application
        app = QApplication(sys.argv)

        # Set application properties
        app.setApplicationName("Trading IA")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("Trading IA Team")

        # Create and show main window
        logger.info("Initializing main window...")
        window = MainWindow()
        window.show()

        logger.info("Application started successfully")

        # Start event loop
        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()