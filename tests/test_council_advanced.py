import pytest
import math
from core.council import Council

@pytest.fixture
def council():
    return Council()

def test_wfa_certification_logic(council):
    # 1. Register a certified strategy
    council.certify_strategy("Strat_A", stability_score=0.8, details={"note": "Very robust"})
    
    # 2. Register a rejected strategy
    council.certify_strategy("Strat_B", stability_score=0.2, details={"note": "Overfitted"})
    
    # 3. Register a warning strategy
    council.certify_strategy("Strat_C", stability_score=0.5, details={"note": "Unstable"})
    
    # Test Strat_A (Should be APPROVED)
    context_a = {"strategy_id": "Strat_A"}
    decision_a = council.decide(context_a)
    # Trend Master should vote YES (1) because of the rule
    # But wait, decide() aggregates votes.
    # If only Trend Master votes, and he votes 1, result is 1.
    
    # Let's check the rule output directly first
    rule_out = council._check_strategy_certification(context_a)
    assert rule_out["signal"] == 1
    assert math.isclose(rule_out["score"], 0.8)
    
    # Test Strat_B (Should be REJECTED)
    context_b = {"strategy_id": "Strat_B"}
    rule_out_b = council._check_strategy_certification(context_b)
    assert rule_out_b["signal"] == -1
    
    # Test Strat_C (Should be WARNING/Neutral)
    context_c = {"strategy_id": "Strat_C"}
    rule_out_c = council._check_strategy_certification(context_c)
    assert rule_out_c["signal"] == 0

def test_pattern_confluence_logic(council):
    # 1. Register known patterns
    council.register_pattern({"pattern_name": "Golden Cross", "win_rate": 0.75})
    council.register_pattern({"pattern_name": "Squeeze", "win_rate": 0.45}) # Low win rate
    
    # 2. Context with Golden Cross
    context_good = {"active_patterns": ["Golden Cross"]}
    rule_out = council._check_pattern_confluence(context_good)
    assert rule_out["signal"] == 1
    assert rule_out["score"] > 0
    
    # 3. Context with Squeeze (Low WR)
    context_bad = {"active_patterns": ["Squeeze"]}
    rule_out_bad = council._check_pattern_confluence(context_bad)
    assert rule_out_bad["signal"] == 0
    
    # 4. Context with Unknown Pattern
    context_unknown = {"active_patterns": ["Mysterious Pattern"]}
    rule_out_unknown = council._check_pattern_confluence(context_unknown)
    assert rule_out_unknown["signal"] == 0

def test_full_council_decision_with_wfa(council):
    # Setup
    council.certify_strategy("Strat_Robust", 0.9)
    
    # Context
    context = {
        "strategy_id": "Strat_Robust",
        "price": 100,
        # Add dummy data for other rules if any
    }
    
    # Decide
    decision = council.decide(context)
    
    # Verify Trend Master voted
    votes = decision["expert_votes"]
    assert "Trend Master" in votes
    assert votes["Trend Master"] == 1
    assert decision["decision"] == 1
