import pytest
import os
import json
from datetime import date, timedelta
from core.risk.risk_manager import RiskManager


@pytest.fixture
def risk_manager():
    config = {"max_daily_drawdown": 0.10, "kill_switch_file": "test_kill_switch.json"}  # 10%
    rm = RiskManager(config)
    yield rm
    # Cleanup
    if os.path.exists("test_kill_switch.json"):
        os.remove("test_kill_switch.json")


def test_initialization(risk_manager):
    assert risk_manager.max_daily_drawdown == 0.10
    assert not risk_manager.is_halted


def test_daily_drawdown_check(risk_manager):
    today = date(2024, 1, 1)

    # Start day with 1000
    risk_manager.update_state(1000.0, today)
    assert risk_manager.check_order({})["allowed"] == True

    # Drop to 950 (5% loss) - Should be OK
    risk_manager.update_state(950.0, today)
    assert risk_manager.check_order({})["allowed"] == True

    # Drop to 890 (11% loss) - Should Fail
    risk_manager.update_state(890.0, today)
    check = risk_manager.check_order({})
    assert check["allowed"] == False
    assert "Max Daily Drawdown exceeded" in check["reason"]
    assert risk_manager.is_halted


def test_kill_switch_manual(risk_manager):
    risk_manager.activate_kill_switch("Manual Test")
    assert risk_manager.is_halted

    check = risk_manager.check_order({})
    assert check["allowed"] == False
    assert "Kill Switch" in check["reason"]

    # Verify persistence
    assert os.path.exists("test_kill_switch.json")

    risk_manager.reset_kill_switch()
    assert not risk_manager.is_halted
    assert not os.path.exists("test_kill_switch.json")


def test_kill_switch_file_detection(risk_manager):
    # Create external kill switch file
    with open("test_kill_switch.json", "w") as f:
        json.dump({"active": True, "reason": "External Halt"}, f)

    # Update state should trigger check
    risk_manager.update_state(1000.0, date(2024, 1, 1))

    assert risk_manager.is_halted
    check = risk_manager.check_order({})
    assert check["allowed"] == False


def test_new_day_reset(risk_manager):
    day1 = date(2024, 1, 1)
    day2 = date(2024, 1, 2)

    # Day 1: Start 1000
    risk_manager.update_state(1000.0, day1)
    assert risk_manager.daily_start_equity == 1000.0

    # Day 2: Start 1100 (Profit from previous day)
    risk_manager.update_state(1100.0, day2)
    assert risk_manager.daily_start_equity == 1100.0
    assert risk_manager.current_date == day2
