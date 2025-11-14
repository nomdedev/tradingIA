#!/usr/bin/env python3
"""
BTC Trading Strategy Platform Launcher
=====================================

This script provides an easy way to launch the BTC Trading Strategy Platform.

Requirements:
- Python 3.8+
- Dependencies listed in requirements_platform.txt

Installation:
1. Create a virtual environment (recommended):
   python -m venv trading_platform_env

2. Activate the virtual environment:
   - Windows: trading_platform_env\Scripts\activate
   - Linux/Mac: source trading_platform_env/bin/activate

3. Install dependencies:
   pip install -r requirements_platform.txt

4. Run the platform:
   python launch_platform.py

Or run directly:
python -m src.main_platform

Author: TradingIA Team
Version: 1.0.0
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'PyQt6', 'pandas', 'numpy', 'alpaca_trade_api',
        'plotly', 'backtesting', 'skopt'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 Install with: pip install -r requirements_platform.txt")
        return False

    return True

def launch_platform():
    """Launch the trading platform"""
    try:
        print("🚀 Launching BTC Trading Strategy Platform...")

        # Add src directory to path
        src_path = Path(__file__).parent.parent / 'src'
        sys.path.insert(0, str(src_path))

        # Import and run main platform
        from main_platform import main
        main()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error launching platform: {e}")
        return False

    return True

def install_dependencies():
    """Install dependencies from requirements file"""
    requirements_file = Path(__file__).parent / 'requirements_platform.txt'

    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False

    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main launcher function"""
    print("BTC Trading Strategy Platform Launcher")
    print("=" * 40)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check if install flag is provided
    if len(sys.argv) > 1 and sys.argv[1] == '--install':
        if not install_dependencies():
            sys.exit(1)
        return

    # Check dependencies
    if not check_dependencies():
        print("\n💡 Run with --install flag to install dependencies:")
        print("   python launch_platform.py --install")
        sys.exit(1)

    # Launch platform
    if not launch_platform():
        sys.exit(1)

if __name__ == '__main__':
    main()