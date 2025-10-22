#!/usr/bin/env python3
"""
Script para crear la estructura completa de directorios del proyecto de trading algorítmico.
"""

import os

def create_structure():
    """Crea la estructura completa de directorios y archivos __init__.py"""

    # Definir la estructura de directorios
    base_dir = "trading_competition"

    directories = [
        f"{base_dir}/data/raw",
        f"{base_dir}/data/processed",
        f"{base_dir}/agents",
        f"{base_dir}/environments",
        f"{base_dir}/strategies",
        f"{base_dir}/utils",
        f"{base_dir}/models/checkpoints",
        f"{base_dir}/results/logs",
        f"{base_dir}/results/figures",
        f"{base_dir}/results/backtests",
        f"{base_dir}/notebooks",
        f"{base_dir}/tests",
    ]

    # Archivos __init__.py a crear
    init_files = [
        f"{base_dir}/agents/__init__.py",
        f"{base_dir}/environments/__init__.py",
        f"{base_dir}/strategies/__init__.py",
        f"{base_dir}/utils/__init__.py",
    ]

    dirs_created = 0
    files_created = 0

    print("🏗️  Creando estructura de directorios para trading_competition...\n")

    # Crear directorios
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✓ {directory}")
            dirs_created += 1
        else:
            print(f"✓ {directory} (ya existía)")

    print("\n📄 Creando archivos __init__.py...\n")

    # Crear archivos __init__.py
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write("")  # Archivo vacío
            print(f"✓ {init_file}")
            files_created += 1
        else:
            print(f"✓ {init_file} (ya existía)")

    print(f"\n✅ Estructura creada: {dirs_created} directorios nuevos, {files_created} archivos nuevos")

def verify_structure():
    """Verifica que toda la estructura existe"""

    base_dir = "trading_competition"

    paths_to_check = [
        f"{base_dir}/data",
        f"{base_dir}/data/raw",
        f"{base_dir}/data/processed",
        f"{base_dir}/agents",
        f"{base_dir}/agents/__init__.py",
        f"{base_dir}/environments",
        f"{base_dir}/environments/__init__.py",
        f"{base_dir}/strategies",
        f"{base_dir}/strategies/__init__.py",
        f"{base_dir}/utils",
        f"{base_dir}/utils/__init__.py",
        f"{base_dir}/models",
        f"{base_dir}/models/checkpoints",
        f"{base_dir}/results",
        f"{base_dir}/results/logs",
        f"{base_dir}/results/figures",
        f"{base_dir}/results/backtests",
        f"{base_dir}/notebooks",
        f"{base_dir}/tests",
    ]

    print("\n🔍 Verificando estructura...\n")

    all_exist = True
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"✓ {path}")
        else:
            print(f"✗ {path} - FALTA")
            all_exist = False

    if all_exist:
        print("\n✅ ¡Toda la estructura está completa!")
    else:
        print("\n⚠️  Algunos elementos faltan. Revisa arriba.")

    return all_exist

def main():
    """Función principal"""
    print("🚀 Iniciando creación de estructura del proyecto de trading\n")

    create_structure()
    verify_structure()

    print("\n🎯 Proyecto listo para desarrollo!")

if __name__ == "__main__":
    main()