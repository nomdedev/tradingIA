
import os

file_path = 'src/gui/platform_gui_tab1_improved.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# The problematic block - adjusted based on debug output
# Note the double braces for QLabel which was partially edited before
old_block = """            QLabel {{
                font-size: 14px;
                color: {DarkTheme.TEXT_PRIMARY};
            }
            QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
                color: white;
                font-size: 14px;
                min-height: 25px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QDateEdit {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
                color: white;
                font-size: 14px;
                min-height: 25px;
            }
        \"\"\"
        )"""

new_block = """            QLabel {{
                font-size: 14px;
                color: {DarkTheme.TEXT_PRIMARY};
            }}
            QComboBox {{
                background-color: {DarkTheme.BG_PRIMARY};
                border: 1px solid {DarkTheme.BORDER_COLOR};
                border-radius: 4px;
                padding: 5px;
                color: {DarkTheme.TEXT_HIGHLIGHT};
                font-size: 14px;
                min-height: 25px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QDateEdit {{
                background-color: {DarkTheme.BG_PRIMARY};
                border: 1px solid {DarkTheme.BORDER_COLOR};
                border-radius: 4px;
                padding: 5px;
                color: {DarkTheme.TEXT_HIGHLIGHT};
                font-size: 14px;
                min-height: 25px;
            }}
        \"\"\"
        )"""

if old_block in content:
    new_content = content.replace(old_block, new_block)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully replaced block")
else:
    print("Could not find block")
