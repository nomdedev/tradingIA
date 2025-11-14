"""
Script para ajustar anchos de widgets en los archivos GUI
Establece anchos apropiados para QSpinBox, QDoubleSpinBox y otros widgets
"""

import re
from pathlib import Path


def fix_widget_widths():
    """Corregir anchos de widgets en archivos GUI"""
    
    gui_dir = Path("src/gui")
    
    # Patrones a buscar y corregir
    patterns = {
        # QSpinBox sin setMaximumWidth
        r'(self\.\w+\s*=\s*QSpinBox\(\))': {
            'add_after': '\n        {widget}.setMaximumWidth(120)  # Ancho optimizado'
        },
        # QDoubleSpinBox sin setMaximumWidth
        r'(self\.\w+\s*=\s*QDoubleSpinBox\(\))': {
            'add_after': '\n        {widget}.setMaximumWidth(150)  # Ancho optimizado para decimales'
        },
    }
    
    files_to_fix = list(gui_dir.glob("platform_gui_tab*.py"))
    
    print(f"🔧 Analizando {len(files_to_fix)} archivos GUI...\n")
    
    for file_path in files_to_fix:
        print(f"📄 {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        lines = content.split('\n')
        
        # Buscar widgets sin ancho configurado
        widgets_found = []
        
        for i, line in enumerate(lines):
            # Buscar QSpinBox sin setMaximumWidth después
            if 'QSpinBox()' in line and 'self.' in line:
                widget_name = re.search(r'self\.(\w+)\s*=', line)
                if widget_name:
                    widget = widget_name.group(0)
                    # Verificar si ya tiene setMaximumWidth
                    has_width = False
                    for j in range(i+1, min(i+5, len(lines))):
                        if widget_name.group(1) in lines[j] and 'setMaximumWidth' in lines[j]:
                            has_width = True
                            break
                    
                    if not has_width:
                        widgets_found.append(('QSpinBox', widget_name.group(1), i))
            
            # Buscar QDoubleSpinBox sin setMaximumWidth después
            if 'QDoubleSpinBox()' in line and 'self.' in line:
                widget_name = re.search(r'self\.(\w+)\s*=', line)
                if widget_name:
                    widget = widget_name.group(0)
                    # Verificar si ya tiene setMaximumWidth
                    has_width = False
                    for j in range(i+1, min(i+5, len(lines))):
                        if widget_name.group(1) in lines[j] and 'setMaximumWidth' in lines[j]:
                            has_width = True
                            break
                    
                    if not has_width:
                        widgets_found.append(('QDoubleSpinBox', widget_name.group(1), i))
        
        if widgets_found:
            print(f"  ⚠️  Encontrados {len(widgets_found)} widgets sin ancho configurado:")
            for widget_type, name, line_num in widgets_found:
                print(f"     • {widget_type}: self.{name} (línea {line_num + 1})")
        else:
            print(f"  ✅ Todos los widgets tienen anchos configurados")
        
        print()


def generate_fix_recommendations():
    """Generar recomendaciones de corrección"""
    
    print("\n" + "="*80)
    print("RECOMENDACIONES DE CORRECCIÓN")
    print("="*80)
    
    recommendations = """
Para corregir los anchos de widgets, sigue estos patrones:

1. QSpinBox (números enteros):
   
   self.max_positions = QSpinBox()
   self.max_positions.setMaximumWidth(100)  # Para valores de 1-2 dígitos
   self.max_positions.setMaximumWidth(120)  # Para valores de 3-4 dígitos
   
2. QDoubleSpinBox (números decimales):
   
   self.risk_per_trade = QDoubleSpinBox()
   self.risk_per_trade.setMaximumWidth(120)  # Para porcentajes
   self.risk_per_trade.setMaximumWidth(150)  # Para valores monetarios

3. QLineEdit (texto corto):
   
   self.symbol_input = QLineEdit()
   self.symbol_input.setMaximumWidth(150)  # Para símbolos/códigos
   
4. QComboBox (desplegables):
   
   self.timeframe_combo = QComboBox()
   self.timeframe_combo.setMaximumWidth(120)  # Para opciones cortas

IMPORTANTE:
- Siempre agregar el setMaximumWidth() inmediatamente después de crear el widget
- Dejar al menos 20-30px extra para acomodar prefijos/sufijos ($, %, etc.)
- Testear con los valores máximos esperados
"""
    
    print(recommendations)


if __name__ == '__main__':
    print("\n🔍 ANÁLISIS DE ANCHOS DE WIDGETS\n")
    fix_widget_widths()
    generate_fix_recommendations()
