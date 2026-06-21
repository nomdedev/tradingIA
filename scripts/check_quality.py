import subprocess
import sys
import os


def run_command(command, description):
    print(f"Running {description}...")
    try:
        result = subprocess.run(command, shell=True, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} Passed")
            return True
        else:
            print(f"❌ {description} Failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error running {description}: {e}")
        return False


def main():
    print("🔍 Starting Code Quality Checks...")

    # 1. Black (Formatter)
    # --check only checks, doesn't modify
    black_passed = run_command("python -m black --check core src scripts tests", "Black Format Check")

    # 2. Flake8 (Linter)
    # We ignore some common errors for now to not be too strict initially
    # E501: Line too long (handled by black mostly, but sometimes comments exceed)
    flake8_passed = run_command(
        "python -m flake8 core src scripts tests --ignore=E501,W503,E203 --max-line-length=120", "Flake8 Lint Check"
    )

    # 3. Pytest (Tests)
    pytest_passed = run_command(
        "python -m pytest tests/test_council_protocol.py tests/test_risk_manager.py", "Critical Tests"
    )

    if black_passed and flake8_passed and pytest_passed:
        print("\n✨ All Quality Checks Passed! Ready to Commit.")
        sys.exit(0)
    else:
        print("\n⚠️ Some checks failed. Please fix them before committing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
