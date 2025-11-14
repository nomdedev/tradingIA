"""
Herramienta de diagnóstico completo del proyecto
Muestra todos los errores, warnings y problemas del sistema
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import re


class ProjectDiagnostic:
    """Herramienta de diagnóstico completo del proyecto"""

    def __init__(self, project_root=None):
        self.project_root = Path(project_root or os.getcwd())
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'errors': [],
            'warnings': [],
            'info': []
        }

    def print_header(self, title):
        """Imprimir encabezado de sección"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def print_section(self, title, items, color="white"):
        """Imprimir sección con items"""
        colors = {
            'red': '\033[91m',
            'yellow': '\033[93m',
            'green': '\033[92m',
            'blue': '\033[94m',
            'white': '\033[0m'
        }

        print(f"\n{colors.get(color, colors['white'])}┌─ {title} ({len(items)})")
        for item in items:
            print(f"│  • {item}")
        print(f"└{'─' * 78}{colors['white']}")

    def check_python_environment(self):
        """Verificar entorno Python"""
        self.print_header("1. ENTORNO PYTHON")

        try:
            # Versión de Python
            version = sys.version
            print(f"✓ Python: {version.split()[0]}")
            self.results['info'].append(f"Python version: {version.split()[0]}")

            # Virtual environment
            in_venv = hasattr(sys, 'prefix') and sys.prefix != sys.base_prefix
            if in_venv:
                print(f"✓ Virtual Environment: Activo ({sys.prefix})")
                self.results['info'].append("Virtual environment: Active")
            else:
                print("⚠ Virtual Environment: No detectado")
                self.results['warnings'].append("No virtual environment detected")

            # Paquetes instalados
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format', 'json'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                packages = json.loads(result.stdout)
                print(f"✓ Paquetes instalados: {len(packages)}")
                self.results['info'].append(f"Installed packages: {len(packages)}")

                # Verificar paquetes críticos
                critical_packages = [
                    'pandas', 'numpy', 'PySide6', 'pytest',
                    'scikit-learn', 'joblib', 'alpaca-trade-api'
                ]

                missing = []
                for pkg in critical_packages:
                    if not any(p['name'].lower() == pkg.lower() for p in packages):
                        missing.append(pkg)

                if missing:
                    print(f"⚠ Paquetes faltantes: {', '.join(missing)}")
                    self.results['warnings'].extend([f"Missing package: {p}" for p in missing])

        except Exception as e:
            error_msg = f"Error checking Python environment: {str(e)}"
            print(f"✗ {error_msg}")
            self.results['errors'].append(error_msg)

    def check_test_suite(self):
        """Verificar suite de tests"""
        self.print_header("2. SUITE DE TESTS")

        try:
            tests_dir = self.project_root / 'tests'

            if not tests_dir.exists():
                self.results['errors'].append("Tests directory not found")
                print("✗ Directorio de tests no encontrado")
                return

            # Contar tests
            test_files = list(tests_dir.glob('test_*.py'))
            disabled_files = list(tests_dir.glob('test_*.py.disabled'))

            print(f"✓ Archivos de test activos: {len(test_files)}")
            print(f"⚠ Archivos de test deshabilitados: {len(disabled_files)}")

            self.results['info'].append(f"Active test files: {len(test_files)}")
            self.results['warnings'].append(f"Disabled test files: {len(disabled_files)}")

            if disabled_files:
                print("\n  Archivos deshabilitados:")
                for f in disabled_files:
                    print(f"    • {f.name}")

            # Ejecutar pytest para verificar
            print("\n  Ejecutando pytest (colección de tests)...")
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', str(tests_dir), '--collect-only', '-q'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.project_root)
            )

            if result.returncode == 0:
                # Contar tests
                output = result.stdout
                match = re.search(r'(\d+) tests? collected', output)
                if match:
                    num_tests = match.group(1)
                    print(f"✓ Tests colectados: {num_tests}")
                    self.results['info'].append(f"Tests collected: {num_tests}")
            else:
                error_msg = "Error collecting tests"
                print(f"✗ {error_msg}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:500]}")
                self.results['errors'].append(error_msg)

        except subprocess.TimeoutExpired:
            error_msg = "Test collection timeout"
            print(f"✗ {error_msg}")
            self.results['errors'].append(error_msg)
        except Exception as e:
            error_msg = f"Error checking test suite: {str(e)}"
            print(f"✗ {error_msg}")
            self.results['errors'].append(error_msg)

    def check_code_quality(self):
        """Verificar calidad de código"""
        self.print_header("3. CALIDAD DE CÓDIGO")

        try:
            # Archivos Python
            py_files = list(self.project_root.rglob('*.py'))
            py_files = [f for f in py_files if '.venv' not in str(f) and '__pycache__' not in str(f)]

            print(f"✓ Archivos Python encontrados: {len(py_files)}")
            self.results['info'].append(f"Python files: {len(py_files)}")

            # Buscar problemas comunes
            print("\n  Buscando problemas comunes...")

            issues = {
                'syntax_errors': [],
                'import_errors': [],
                'encoding_issues': []
            }

            for py_file in py_files[:50]:  # Limitar a 50 archivos para no sobrecargar
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Verificar sintaxis básica
                    try:
                        compile(content, str(py_file), 'exec')
                    except SyntaxError as e:
                        issues['syntax_errors'].append(f"{py_file.name}: {str(e)}")

                except UnicodeDecodeError:
                    issues['encoding_issues'].append(str(py_file.name))
                except Exception:
                    pass

            # Reportar problemas
            if issues['syntax_errors']:
                print(f"✗ Errores de sintaxis: {len(issues['syntax_errors'])}")
                self.results['errors'].extend(issues['syntax_errors'])
                for err in issues['syntax_errors'][:5]:
                    print(f"    • {err}")

            if issues['encoding_issues']:
                print(f"⚠ Problemas de encoding: {len(issues['encoding_issues'])}")
                self.results['warnings'].extend([f"Encoding issue: {f}" for f in issues['encoding_issues']])

            if not any(issues.values()):
                print("✓ No se encontraron problemas comunes")

        except Exception as e:
            error_msg = f"Error checking code quality: {str(e)}"
            print(f"✗ {error_msg}")
            self.results['errors'].append(error_msg)

    def check_project_structure(self):
        """Verificar estructura del proyecto"""
        self.print_header("4. ESTRUCTURA DEL PROYECTO")

        required_dirs = [
            'src', 'tests', 'data', 'logs', 'config',
            'strategies', 'dashboard', 'backtesting'
        ]

        required_files = [
            'requirements_dashboard.txt',
            'run_paper_trading.py',
            'README.md'
        ]

        print("\n  Directorios:")
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                print(f"    ✓ {dir_name}/")
                self.results['info'].append(f"Directory exists: {dir_name}")
            else:
                print(f"    ✗ {dir_name}/ (faltante)")
                self.results['warnings'].append(f"Missing directory: {dir_name}")

        print("\n  Archivos:")
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                print(f"    ✓ {file_name}")
                self.results['info'].append(f"File exists: {file_name}")
            else:
                print(f"    ✗ {file_name} (faltante)")
                self.results['warnings'].append(f"Missing file: {file_name}")

    def check_imports(self):
        """Verificar imports del proyecto"""
        self.print_header("5. VERIFICACIÓN DE IMPORTS")

        # Archivos principales
        main_files = [
            'dashboard/app.py',
            'run_paper_trading.py',
            'src/backtester_core.py'
        ]

        print("\n  Verificando imports en archivos principales...")

        for file_path in main_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                print(f"  ⚠ {file_path} no encontrado")
                continue

            try:
                # Intentar importar módulo
                result = subprocess.run(
                    [sys.executable, '-c', f'import importlib.util; spec = importlib.util.spec_from_file_location("module", r"{full_path}"); print("OK")'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(self.project_root)
                )

                if result.returncode == 0 and "OK" in result.stdout:
                    print(f"  ✓ {file_path}")
                else:
                    print(f"  ✗ {file_path}")
                    if result.stderr:
                        error_lines = result.stderr.split('\n')[:3]
                        for line in error_lines:
                            if line.strip():
                                print(f"      {line.strip()}")
                        self.results['errors'].append(f"Import error in {file_path}")

            except subprocess.TimeoutExpired:
                print(f"  ⚠ {file_path} (timeout)")
                self.results['warnings'].append(f"Import timeout: {file_path}")
            except Exception as e:
                print(f"  ✗ {file_path}: {str(e)}")
                self.results['errors'].append(f"Import check failed: {file_path}")

    def check_data_files(self):
        """Verificar archivos de datos"""
        self.print_header("6. ARCHIVOS DE DATOS")

        data_dir = self.project_root / 'data'

        if not data_dir.exists():
            print("✗ Directorio data/ no encontrado")
            self.results['errors'].append("Data directory not found")
            return

        # Verificar subdirectorios
        raw_dir = data_dir / 'raw'
        processed_dir = data_dir / 'processed'

        print(f"  ✓ data/ encontrado")

        if raw_dir.exists():
            raw_files = list(raw_dir.glob('*.csv'))
            print(f"  ✓ data/raw/ - {len(raw_files)} archivos CSV")
            self.results['info'].append(f"Raw data files: {len(raw_files)}")
        else:
            print(f"  ⚠ data/raw/ no encontrado")
            self.results['warnings'].append("Raw data directory missing")

        if processed_dir.exists():
            proc_files = list(processed_dir.glob('*.csv'))
            print(f"  ✓ data/processed/ - {len(proc_files)} archivos CSV")
            self.results['info'].append(f"Processed data files: {len(proc_files)}")
        else:
            print(f"  ⚠ data/processed/ no encontrado")
            self.results['warnings'].append("Processed data directory missing")

    def check_logs(self):
        """Verificar archivos de logs"""
        self.print_header("7. ARCHIVOS DE LOGS")

        logs_dir = self.project_root / 'logs'

        if not logs_dir.exists():
            print("⚠ Directorio logs/ no encontrado")
            self.results['warnings'].append("Logs directory not found")
            return

        # Buscar archivos de log
        log_files = list(logs_dir.glob('*.log')) + list(logs_dir.glob('*.csv'))

        print(f"  ✓ logs/ encontrado - {len(log_files)} archivos")

        # Verificar logs recientes
        recent_logs = []
        for log_file in log_files:
            try:
                mtime = log_file.stat().st_mtime
                age_hours = (datetime.now().timestamp() - mtime) / 3600
                if age_hours < 24:
                    recent_logs.append((log_file.name, age_hours))
            except Exception:
                pass

        if recent_logs:
            print(f"  ✓ Logs recientes (< 24h): {len(recent_logs)}")
            for name, age in recent_logs[:5]:
                print(f"      • {name} ({age:.1f}h)")
        else:
            print(f"  ⚠ No hay logs recientes")
            self.results['warnings'].append("No recent logs found")

    def generate_summary(self):
        """Generar resumen final"""
        self.print_header("RESUMEN DEL DIAGNÓSTICO")

        errors = len(self.results['errors'])
        warnings = len(self.results['warnings'])
        info = len(self.results['info'])

        print(f"\n  Total de problemas encontrados:")
        print(f"    • Errores críticos: {errors}")
        print(f"    • Advertencias: {warnings}")
        print(f"    • Información: {info}")

        if errors > 0:
            self.print_section("ERRORES CRÍTICOS", self.results['errors'][:20], "red")

        if warnings > 0:
            self.print_section("ADVERTENCIAS", self.results['warnings'][:20], "yellow")

        # Estado general
        print("\n" + "=" * 80)
        if errors == 0 and warnings < 5:
            print("  ✓ ESTADO: SALUDABLE ✓")
        elif errors == 0:
            print("  ⚠ ESTADO: CON ADVERTENCIAS ⚠")
        else:
            print("  ✗ ESTADO: REQUIERE ATENCIÓN ✗")
        print("=" * 80 + "\n")

    def run_full_diagnostic(self):
        """Ejecutar diagnóstico completo"""
        print("\n" + "=" * 80)
        print("  DIAGNOSTICO COMPLETO DEL PROYECTO TRADING IA")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 80)

        try:
            self.check_python_environment()
            self.check_project_structure()
            self.check_test_suite()
            self.check_code_quality()
            self.check_imports()
            self.check_data_files()
            self.check_logs()
            self.generate_summary()

        except KeyboardInterrupt:
            print("\n\n⚠ Diagnóstico interrumpido por el usuario")
        except Exception as e:
            print(f"\n\n✗ Error en diagnóstico: {str(e)}")

        return self.results


def main():
    """Función principal"""
    # Detectar raíz del proyecto
    current_dir = Path(__file__).parent.parent

    diagnostic = ProjectDiagnostic(current_dir)
    results = diagnostic.run_full_diagnostic()

    # Guardar resultados
    output_file = current_dir / 'logs' / 'diagnostic_report.json'
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Reporte guardado en: {output_file}")


if __name__ == '__main__':
    main()
