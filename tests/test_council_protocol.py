import pytest
import math
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.council import Council


@pytest.fixture
def council():
    """Fixture to provide a fresh Council instance."""
    c = Council()
    c.register_standard_experts()
    return c


def test_expert_registration(council):
    """Test that standard experts are registered correctly."""
    experts = council.experts
    assert "Risk Warden" in experts
    assert math.isclose(experts["Risk Warden"]["weight"], 2.5)
    assert math.isclose(experts["Trend Master"]["weight"], 2.0)
    assert math.isclose(experts["Data Oracle"]["weight"], 1.0)


def test_consensus_decision(council):
    """Test a clear consensus scenario."""
    # Add rules that pass
    council.add_rule("rule1", lambda c: {"signal": 1, "score": 1.0}, expert="Trend Master")
    council.add_rule("rule2", lambda c: {"signal": 1, "score": 1.0}, expert="Risk Warden")

    context = {}
    decision = council.decide(context)

    assert decision["decision"] == 1
    assert decision["phase"] == "CONSENSUS"
    assert decision["aggregate_score"] > 0.5


def test_risk_veto(council):
    """Test that Risk Warden can veto even if others approve."""
    # Trend Master says YES
    council.add_rule("trend_yes", lambda c: {"signal": 1, "score": 1.0}, expert="Trend Master")

    # Risk Warden says NO (Veto)
    council.add_rule("risk_no", lambda c: {"signal": -1, "score": -1.0}, expert="Risk Warden")

    context = {}
    decision = council.decide(context)

    assert decision["decision"] == 0  # Rejected
    assert decision["phase"] == "VETO"
    assert "Risk Warden VETO" in decision["reason"]


def test_data_veto(council):
    """Test that Data Oracle can veto."""
    # Trend Master says YES
    council.add_rule("trend_yes", lambda c: {"signal": 1, "score": 1.0}, expert="Trend Master")

    # Data Oracle says NO
    council.add_rule("data_no", lambda c: {"signal": -1, "score": -1.0}, expert="Data Oracle")

    context = {}
    decision = council.decide(context)

    assert decision["decision"] == 0
    assert decision["phase"] == "VETO"
    assert "Data Oracle VETO" in decision["reason"]


def test_weighted_voting(council):
    """Test that weights affect the outcome."""
    # Trend Master (2.0) says YES
    council.add_rule("trend_yes", lambda c: {"signal": 1, "score": 1.0}, expert="Trend Master")

    # Sentiment Seer (1.0) says NO (but not a veto expert)
    council.add_rule("sentiment_no", lambda c: {"signal": -1, "score": -1.0}, expert="Sentiment Seer")

    # Score calculation:
    # Trend Master: 1 * 2.0 = 2.0
    # Sentiment Seer: -1 * 1.0 = -1.0
    # Total Score: (2.0 - 1.0) / (2.0 + 1.0) = 1.0 / 3.0 = 0.33

    context = {}
    decision = council.decide(context)

    assert decision["aggregate_score"] == pytest.approx(0.333, 0.01)
    # 0.33 is > 0 but < 0.5, so it should be WEAK BUY (1) or depending on threshold
    # In code: if final_score > 0.0: decision = 1
    assert decision["decision"] == 1


def test_tie_breaking(council):
    """Test tie breaking logic."""
    # Let's modify weights for this test
    council.register_expert("Equal_Risk", "Risk", "risk", 2.0)
    council.register_expert("Equal_Trend", "Trend", "trading", 2.0)

    council.add_rule("r1", lambda c: {"signal": 1, "score": 1.0}, expert="Equal_Trend")
    council.add_rule("r2", lambda c: {"signal": -1, "score": -1.0}, expert="Equal_Risk")

    # Score: (2 - 2) / 4 = 0

    context = {}
    decision = council.decide(context)

    assert math.isclose(decision["aggregate_score"], 0.0)
    # If score is 0, decision is 0 (NO TRADE)
    # But if Risk Warden is involved in the tie (via negative vote), it might force negative
    # In our code: if final_score == 0.0 ... risk_vote = expert_votes.get("Risk Warden", 0)
    # Here "Risk Warden" didn't vote, "Equal_Risk" did.
    # So it should be 0.
    assert decision["decision"] == 0


if __name__ == "__main__":
    # Manual run helper
    c = council()
    test_expert_registration(c)
    test_consensus_decision(c)
    test_risk_veto(c)
    test_data_veto(c)
    test_weighted_voting(c)
    test_tie_breaking(c)
    print("All tests passed!")
