import re

# Read the file
with open("src/gui/platform_gui_tab6_user_friendly.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix the f-string syntax error
content = re.sub(
    r"self\.strategy_info_label\.setText\(f.*params_text.*\)",
    "self.strategy_info_label.setText(f\" {strategy_name}\\n\\n{strategies_info[strategy_name]['description']}\\n\\n Parámetros:\\n{params_text}\")",
    content,
    flags=re.DOTALL
)

# Write back
with open("src/gui/platform_gui_tab6_user_friendly.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed f-string syntax")
