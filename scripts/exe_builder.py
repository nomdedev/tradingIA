#!/usr/bin/env python3
"""
EXE Builder for BTC Trading System
===================================

Creates standalone executable with PyInstaller for 24/7 monitoring.

Features:
- CLI interface with modes: backtest/opt/sensitivity/monitor
- Embedded Python environment
- Auto-restart on crashes
- Logging to file
- System tray icon (optional)

Usage:
    python exe_builder.py --mode=monitor --symbol=BTCUSD
    python exe_builder.py --mode=backtest --config=config.json
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
import shutil
import platform
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimizer import StrategyOptimizer
from src.backtester import AdvancedBacktester
from src.mtf_data_handler import MultiTFDataHandler
from config.mtf_config import TRADING_CONFIG

class EXEBuilder:
    """Builds standalone executable for trading system"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dist_dir = project_root / 'dist'
        self.build_dir = project_root / 'build'
        self.specs_dir = project_root / 'specs'

        # Create directories
        for dir_path in [self.dist_dir, self.build_dir, self.specs_dir]:
            dir_path.mkdir(exist_ok=True)

    def create_spec_file(self, mode: str = 'monitor') -> str:
        """Create PyInstaller spec file"""

        spec_content = f'''
# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

a = Analysis(
    ['scripts/run_trading.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / 'config'), 'config'),
        (str(project_root / 'models'), 'models'),
        (str(project_root / 'logs'), 'logs'),
    ],
    hiddenimports=[
        'pandas',
        'numpy',
        'ta',
        'talib',
        'scipy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'plotly',
        'flask',
        'alpaca_trade_api',
        'alpaca_py',
        'config.mtf_config',
        'src.optimizer',
        'src.backtester',
        'src.mtf_data_handler',
        'src.rules',
        'src.indicators',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BTC_Trading_System_{mode}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
'''

        spec_file = self.specs_dir / f'btc_trading_{mode}.spec'
        with open(spec_file, 'w') as f:
            f.write(spec_content)

        return str(spec_file)

    def build_executable(self, mode: str = 'monitor') -> bool:
        """Build standalone executable"""

        print(f"🏗️  Building executable for mode: {mode}")

        # Create spec file
        spec_file = self.create_spec_file(mode)

        # Run PyInstaller
        cmd = [
            sys.executable, '-m', 'PyInstaller',
            '--clean',
            '--noconfirm',
            spec_file
        ]

        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Executable built successfully!")
                exe_path = self.dist_dir / f'BTC_Trading_System_{mode}'
                if platform.system() == 'Windows':
                    exe_path = exe_path.with_suffix('.exe')
                print(f"📁 Executable location: {exe_path}")
                return True
            else:
                print("❌ Build failed!")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False

        except Exception as e:
            print(f"❌ Build error: {e}")
            return False

    def create_launcher_script(self, mode: str = 'monitor') -> str:
        """Create launcher script for the executable"""

        launcher_content = f'''#!/usr/bin/env python3
"""
Launcher for BTC Trading System Executable
Mode: {mode}
Built: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/trading_launcher_{mode}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_exe_path() -> Path:
    """Get executable path"""
    dist_dir = Path(__file__).parent / 'dist'
    exe_name = f'BTC_Trading_System_{mode}'

    if platform.system() == 'Windows':
        exe_name += '.exe'

    exe_path = dist_dir / exe_name

    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {{exe_path}}")

    return exe_path

def run_executable(auto_restart: bool = True, max_restarts: int = 3):
    """Run executable with auto-restart"""

    exe_path = get_exe_path()
    restart_count = 0

    while restart_count < max_restarts:
        try:
            logger.info(f"🚀 Starting executable: {{exe_path}}")
            logger.info(f"Mode: {mode}")

            # Set environment variables
            env = os.environ.copy()
            env['TRADING_MODE'] = mode
            env['PYTHONPATH'] = str(Path(__file__).parent)

            # Run executable
            result = subprocess.run(
                [str(exe_path)],
                env=env,
                capture_output=False,  # Show output in console
                check=True
            )

            logger.info("✅ Executable completed successfully")
            break

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Executable failed with code: {{e.returncode}}")
            restart_count += 1

            if auto_restart and restart_count < max_restarts:
                logger.info(f"🔄 Restarting... ({{restart_count}}/{{max_restarts}})")
                time.sleep(5)  # Wait before restart
            else:
                logger.error("❌ Max restarts reached, giving up")
                break

        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
            break

        except Exception as e:
            logger.error(f"❌ Unexpected error: {{e}}")
            break

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BTC Trading System Launcher')
    parser.add_argument('--mode', default='{mode}', help='Trading mode')
    parser.add_argument('--no-restart', action='store_true', help='Disable auto-restart')
    parser.add_argument('--max-restarts', type=int, default=3, help='Max restart attempts')

    args = parser.parse_args()

    auto_restart = not args.no_restart
    run_executable(auto_restart=auto_restart, max_restarts=args.max_restarts)
'''

        launcher_file = self.project_root / f'launch_{mode}.py'
        with open(launcher_file, 'w') as f:
            f.write(launcher_content)

        # Make executable on Unix
        if platform.system() != 'Windows':
            os.chmod(launcher_file, 0o755)

        return str(launcher_file)

    def create_config_template(self) -> str:
        """Create configuration template for executable"""

        config = {
            "trading": {
                "symbol": "BTCUSD",
                "mode": "monitor",
                "initial_capital": 10000,
                "risk_per_trade": 0.01,
                "max_positions": 3
            },
            "optimization": {
                "enabled": False,
                "method": "bayes",
                "n_calls": 50,
                "target_metric": "sharpe_ratio"
            },
            "monitoring": {
                "check_interval": 60,  # seconds
                "log_level": "INFO",
                "alerts": {
                    "email": False,
                    "telegram": False
                }
            },
            "api": {
                "alpaca_key": "",
                "alpaca_secret": "",
                "use_paper": True
            }
        }

        config_file = self.project_root / 'config_template.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        return str(config_file)

def main():
    """Main CLI interface"""

    parser = argparse.ArgumentParser(
        description='BTC Trading System EXE Builder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build monitor executable
  python exe_builder.py --mode=monitor

  # Build backtest executable
  python exe_builder.py --mode=backtest

  # Build with custom spec
  python exe_builder.py --mode=opt --spec=custom.spec

  # Create launcher only
  python exe_builder.py --launcher-only --mode=monitor
        """
    )

    parser.add_argument(
        '--mode',
        choices=['monitor', 'backtest', 'opt', 'sensitivity'],
        default='monitor',
        help='Trading mode for executable'
    )

    parser.add_argument(
        '--build',
        action='store_true',
        help='Build executable'
    )

    parser.add_argument(
        '--launcher-only',
        action='store_true',
        help='Create launcher script only'
    )

    parser.add_argument(
        '--config-template',
        action='store_true',
        help='Create configuration template'
    )

    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean build directories'
    )

    args = parser.parse_args()

    # Initialize builder
    project_root = Path(__file__).parent
    builder = EXEBuilder(project_root)

    # Clean if requested
    if args.clean:
        print("🧹 Cleaning build directories...")
        shutil.rmtree(builder.build_dir, ignore_errors=True)
        shutil.rmtree(builder.dist_dir, ignore_errors=True)
        print("✅ Cleaned")

    # Create config template
    if args.config_template or not (args.build or args.launcher_only):
        config_file = builder.create_config_template()
        print(f"📄 Config template created: {config_file}")

    # Create launcher
    if args.launcher_only or not args.build:
        launcher_file = builder.create_launcher_script(args.mode)
        print(f"🚀 Launcher script created: {launcher_file}")

    # Build executable
    if args.build:
        success = builder.build_executable(args.mode)
        if success:
            print("\n📦 Build Summary:")
            print(f"  - Executable: dist/BTC_Trading_System_{args.mode}")
            print(f"  - Launcher: launch_{args.mode}.py")
            print("  - Config: config_template.json")
            print(f"\n🎯 To run: python launch_{args.mode}.py")
        else:
            print("❌ Build failed")
            sys.exit(1)

if __name__ == '__main__':
    main()