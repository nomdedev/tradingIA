"""
Script para aplicar correcciones de ancho de widgets automáticamente
"""

import re
from pathlib import Path


def apply_width_fixes():
    """Aplicar correcciones de ancho a widgets"""
    
    fixes = [
        # Tab3_improved
        {
            'file': 'src/gui/platform_gui_tab3_improved.py',
            'replacements': [
                {
                    'old': '        self.periods_spin = QSpinBox()\n        self.periods_spin.setRange(3, 12)',
                    'new': '        self.periods_spin = QSpinBox()\n        self.periods_spin.setMaximumWidth(100)  # Ancho optimizado\n        self.periods_spin.setRange(3, 12)'
                },
                {
                    'old': '        self.runs_spin = QSpinBox()\n        self.runs_spin.setRange(100, 2000)',
                    'new': '        self.runs_spin = QSpinBox()\n        self.runs_spin.setMaximumWidth(120)  # Ancho optimizado\n        self.runs_spin.setRange(100, 2000)'
                }
            ]
        },
        # Tab6_improved
        {
            'file': 'src/gui/platform_gui_tab6_improved.py',
            'replacements': [
                {
                    'old': '        self.dd_alert_spin = QDoubleSpinBox()\n        self.dd_alert_spin.setRange(1.0, 50.0)',
                    'new': '        self.dd_alert_spin = QDoubleSpinBox()\n        self.dd_alert_spin.setMaximumWidth(120)  # Ancho optimizado para porcentajes\n        self.dd_alert_spin.setRange(1.0, 50.0)'
                },
                {
                    'old': '        self.loss_alert_spin = QDoubleSpinBox()\n        self.loss_alert_spin.setRange(1.0, 20.0)',
                    'new': '        self.loss_alert_spin = QDoubleSpinBox()\n        self.loss_alert_spin.setMaximumWidth(120)  # Ancho optimizado para porcentajes\n        self.loss_alert_spin.setRange(1.0, 20.0)'
                },
                {
                    'old': '        self.win_streak_spin = QSpinBox()\n        self.win_streak_spin.setRange(1, 20)',
                    'new': '        self.win_streak_spin = QSpinBox()\n        self.win_streak_spin.setMaximumWidth(100)  # Ancho optimizado\n        self.win_streak_spin.setRange(1, 20)'
                },
                {
                    'old': '        self.loss_streak_spin = QSpinBox()\n        self.loss_streak_spin.setRange(1, 20)',
                    'new': '        self.loss_streak_spin = QSpinBox()\n        self.loss_streak_spin.setMaximumWidth(100)  # Ancho optimizado\n        self.loss_streak_spin.setRange(1, 20)'
                }
            ]
        },
        # Tab7_improved
        {
            'file': 'src/gui/platform_gui_tab7_improved.py',
            'replacements': [
                {
                    'old': '        self.alpha_spin = QDoubleSpinBox()\n        self.alpha_spin.setRange(0.01, 1.0)',
                    'new': '        self.alpha_spin = QDoubleSpinBox()\n        self.alpha_spin.setMaximumWidth(120)  # Ancho optimizado para decimales\n        self.alpha_spin.setRange(0.01, 1.0)'
                },
                {
                    'old': '        self.corr_window_spin = QSpinBox()\n        self.corr_window_spin.setRange(10, 200)',
                    'new': '        self.corr_window_spin = QSpinBox()\n        self.corr_window_spin.setMaximumWidth(100)  # Ancho optimizado\n        self.corr_window_spin.setRange(10, 200)'
                },
                {
                    'old': '        self.regime_states_spin = QSpinBox()\n        self.regime_states_spin.setRange(2, 5)',
                    'new': '        self.regime_states_spin = QSpinBox()\n        self.regime_states_spin.setMaximumWidth(100)  # Ancho optimizado\n        self.regime_states_spin.setRange(2, 5)'
                }
            ]
        },
        # Tab7
        {
            'file': 'src/gui/platform_gui_tab7.py',
            'replacements': [
                {
                    'old': '        self.order_size_spinbox = QSpinBox()\n        self.order_size_spinbox.setMinimum(1)',
                    'new': '        self.order_size_spinbox = QSpinBox()\n        self.order_size_spinbox.setMaximumWidth(120)  # Ancho optimizado\n        self.order_size_spinbox.setMinimum(1)'
                }
            ]
        }
    ]
    
    total_fixes = 0
    
    for fix_config in fixes:
        file_path = Path(fix_config['file'])
        
        if not file_path.exists():
            print(f"⚠️  {file_path} no encontrado")
            continue
        
        print(f"\n📄 {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        modified = False
        fixes_applied = 0
        
        for replacement in fix_config['replacements']:
            if replacement['old'] in content:
                content = content.replace(replacement['old'], replacement['new'])
                modified = True
                fixes_applied += 1
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  ✅ {fixes_applied} correcciones aplicadas")
            total_fixes += fixes_applied
        else:
            print(f"  ℹ️  No se encontraron patrones para corregir")
    
    print(f"\n{'='*80}")
    print(f"✅ Total: {total_fixes} correcciones aplicadas exitosamente")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    print("\n🔧 APLICANDO CORRECCIONES DE ANCHO DE WIDGETS\n")
    apply_width_fixes()
