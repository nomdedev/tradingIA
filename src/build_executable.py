import os
import shutil
import subprocess

# PyInstaller command
cmd = [
    'pyinstaller',
    '--onefile',
    '--windowed',
    '--icon=assets/icon.ico',
    '--add-data=config:config',
    '--add-data=assets:assets',
    '--add-data=docs:docs',
    '--name=btc_trading_platform',
    '--distpath=dist',
    '--workpath=build',
    '--specpath=.',
    'src/main_platform.py'
]

print("Building executable...")
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print("✓ Build successful! Executable: dist/btc_trading_platform.exe")
else:
    print(f"✗ Build failed: {result.stderr}")
