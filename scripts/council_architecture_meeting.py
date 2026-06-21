import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.council import Council


def main():
    print("🏛️  Iniciando Sesión del Consejo de Arquitectura...\n")

    council = Council()

    # Registrar Expertos en Programación y Arquitectura
    council.register_expert(
        "Architect_Prime",
        "System Architect",
        domain="system",
        weight=2.0,
        notes="Responsable de la integridad estructural y escalabilidad.",
    )
    council.register_expert(
        "Code_Guardian", "QA Lead", domain="system", weight=1.5, notes="Enfocado en testing, calidad de código y CI/CD."
    )
    council.register_expert(
        "Data_Oracle",
        "Data Engineer",
        domain="data",
        weight=1.5,
        notes="Experto en pipelines de datos y almacenamiento.",
    )
    council.register_expert(
        "Risk_Warden", "Security Officer", domain="risk", weight=2.0, notes="Seguridad operativa y gestión de riesgos."
    )

    # Listar Expertos
    print("👥 Miembros del Consejo Presentes:")
    for name, details in council.experts.items():
        print(f"  - {name} ({details['role']}) [{details['domain'].upper()}]")

    print("\n📋 Agenda del Día: Evaluación de Arquitectura y Plan de Acción")
    print("=" * 60)

    # Simulación de Análisis (basado en el archivo generado)
    print("\n1. Análisis de Estado Actual:")
    print("   - Architect_Prime: 'El núcleo es sólido. Backtester y Council están bien desacoplados.'")
    print("   - Code_Guardian: 'Detecto falta de tests automáticos. Es un riesgo crítico para la refactorización.'")
    print("   - Data_Oracle: 'SQLite aguanta por ahora, pero debemos planear la migración para HFT.'")

    print("\n2. Votación de Prioridades (Plan de Acción):")
    print("   - Fase 6 (Calidad): APROBADO (Unánime)")
    print("   - Fase 7 (Live Trading): APROBADO (Prioridad Alta)")
    print("   - Fase 8 (MLOps): APROBADO (Prioridad Media)")

    print("\n✅ Conclusiones:")
    print("   Se ha generado el documento 'docs/ARCHITECTURE_REVIEW_AND_PLAN.md' con los detalles.")
    print("   El sistema Council ha sido actualizado para soportar dominios de expertise.")


if __name__ == "__main__":
    main()
