#!/usr/bin/env python3
"""
Script para corregir imports en tests
"""
import os
import re
from pathlib import Path

fixes = {
    "test_backtester_core.py": ("from src.backtester_core import", "from core.execution.backtester_core import"),
    "test_backtesting.py": ("from backtester_core import", "from core.execution.backtester_core import"),
    "test_backtesting_new.py": ("from backtester_core import", "from core.execution.backtester_core import"),
    "test_data_validation_comprehensive.py": ("from src.backend_core import", "from core.backend_core import"),
}

tests_dir = Path("tests")

for test_file, (old_import, new_import) in fixes.items():
    filepath = tests_dir / test_file
    if filepath.exists():
        try:
            content = filepath.read_text(encoding='utf-8')
            if old_import in content:
                new_content = content.replace(old_import, new_import)
                filepath.write_text(new_content, encoding='utf-8')
                print(f"✅ Fixed {test_file}")
            else:
                print(f"⏭️  Skipped {test_file} (no match)")
        except Exception as e:
            print(f"❌ Error fixing {test_file}: {e}")
    else:
        print(f"⚠️  File not found: {test_file}")

print("\n✨ Import fixes complete!")
