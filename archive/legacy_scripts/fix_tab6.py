import os

file_path = r"d:\martin\Proyectos\tradingIA\src\gui\platform_gui_tab6_improved.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

print(f"Read {len(content)} bytes.")

target_str = "class Tab6LiveMonitorFixed(QWidget):"
new_str = "class Tab6LiveMonitor(QWidget):"

if target_str in content:
    print("Found class def. Renaming back...")
    new_content = content.replace(target_str, new_str)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("File patched.")
else:
    print("Class def NOT found.")

