#!/usr/bin/env python3
"""
Script maestro para ejecutar la competición completa de agentes de trading.

Este script entrena ambos agentes (RL y GA) y luego ejecuta la competición.
"""

import os
import subprocess

def run_command(command, description):
    """Ejecutar un comando y mostrar su salida"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print('='*60)

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.getcwd())

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            print(f"❌ Error ejecutando: {command}")
            print(f"Código de retorno: {result.returncode}")
            return False

        print(f"✅ {description} completado exitosamente")
        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def check_data_exists():
    """Verificar que los datos estén disponibles"""
    data_path = "data/processed/SPY_with_indicators.csv"
    if not os.path.exists(data_path):
        print(f"❌ Datos no encontrados: {data_path}")
        print("Ejecutando descarga de datos...")
        if not run_command("python download_data.py", "Descargar datos SPY"):
            return False
    else:
        print("✅ Datos encontrados")

    return True

def train_rl_agent():
    """Entrenar agente RL"""
    print("\n🤖 ENTRENANDO AGENTE RL...")

    # Verificar si ya existe
    model_path = "models/ppo_trading_agent.zip"
    if os.path.exists(model_path):
        print(f"✅ Modelo RL ya existe: {model_path}")
        return True

    # Entrenar
    command = "python trading_competition/agents/train_rl_agent.py"
    return run_command(command, "Entrenamiento del agente RL")

def train_ga_agent():
    """Entrenar agente GA"""
    print("\n🧬 ENTRENANDO AGENTE GA...")

    # Verificar si ya existe
    model_path = "models/ga_best_individual.pkl"
    if os.path.exists(model_path):
        print(f"✅ Modelo GA ya existe: {model_path}")
        return True

    # Entrenar
    command = "python trading_competition/agents/train_ga_agent.py"
    return run_command(command, "Entrenamiento del agente GA")

def run_competition():
    """Ejecutar competición"""
    print("\n🏆 EJECUTANDO COMPETICIÓN...")

    command = "python trading_competition/competition.py"
    return run_command(command, "Competición entre agentes")

def main():
    """Función principal"""
    print("🎯 INICIANDO COMPETICIÓN COMPLETA DE AGENTES DE TRADING")
    print("="*70)

    # Verificar datos
    if not check_data_exists():
        print("❌ No se pueden continuar sin datos")
        return

    # Entrenar agentes
    success = True

    if not train_rl_agent():
        print("❌ Falló el entrenamiento del agente RL")
        success = False

    if not train_ga_agent():
        print("❌ Falló el entrenamiento del agente GA")
        success = False

    if not success:
        print("❌ No se puede ejecutar la competición sin ambos agentes entrenados")
        return

    # Ejecutar competición
    if run_competition():
        print("\n🎉 ¡COMPETICIÓN COMPLETA EXITOSAMENTE!")
        print("📊 Revisa los resultados en la carpeta 'results/'")
    else:
        print("❌ La competición falló")

if __name__ == "__main__":
    main()