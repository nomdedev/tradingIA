#!/usr/bin/env python3
"""
Script de prueba para verificar que todas las librerías necesarias estén instaladas correctamente.
"""

def test_import(library_name, import_statement):
    """Prueba importar una librería y reporta el resultado."""
    try:
        if import_statement.startswith("from "):
            # Handle 'from x import y' statements
            parts = import_statement.split()
            module_name = parts[1]
            __import__(module_name)
        else:
            # Handle 'import x as y' or 'import x' statements
            module_name = import_statement.split()[1]
            __import__(module_name)
        print(f"✓ {library_name} instalado correctamente")
        return True
    except ImportError as e:
        print(f"✗ {library_name} no instalado: {e}")
        return False

def main():
    print("🔍 Verificando instalación de librerías para trading IA...\n")

    libraries = [
        ("pandas", "import pandas as pd"),
        ("numpy", "import numpy as np"),
        ("scipy", "import scipy"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("seaborn", "import seaborn as sns"),
        ("stable-baselines3", "import stable_baselines3"),
        ("torch", "import torch"),
        ("deap", "import deap"),
        ("gym", "import gym"),
        ("gymnasium", "import gymnasium"),
        ("backtesting", "import backtesting"),
        ("yfinance", "import yfinance"),
        ("plotly", "import plotly"),
        ("tensorboard", "import tensorboard"),
        ("scikit-learn", "from sklearn import linear_model"),
        # pandas-ta no incluido por incompatibilidad con Python 3.11
    ]

    successful = 0
    total = len(libraries)

    for lib_name, import_stmt in libraries:
        if test_import(lib_name, import_stmt):
            successful += 1

    print(f"\n📊 Resultados: {successful}/{total} librerías instaladas correctamente")

    if successful == total:
        print("✅ ¡TODO INSTALADO CORRECTAMENTE!")
        print("🚀 Listo para comenzar con el proyecto de trading IA")
    else:
        print("⚠️  Algunas librerías faltan. Revisa los errores arriba.")
        print("💡 Para pandas-ta, considera usar una versión anterior de Python (3.10)")

if __name__ == "__main__":
    main()