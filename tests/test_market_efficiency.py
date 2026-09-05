from __future__ import annotations

from football_vnext.domain.value.market_efficiency import (
    MarketEfficiencyClassifier,
    MarketEfficiencyTier,
)


def test_known_high_efficiency_league():
    classifier = MarketEfficiencyClassifier()
    assert classifier.classify("Premier League") == MarketEfficiencyTier.HIGH
    assert classifier.classify("  premier league  ") == MarketEfficiencyTier.HIGH


def test_unknown_league_defaults_to_medium():
    classifier = MarketEfficiencyClassifier()
    assert classifier.classify("Some Obscure Regional Cup") == MarketEfficiencyTier.MEDIUM


def test_known_low_efficiency_league():
    classifier = MarketEfficiencyClassifier()
    assert classifier.classify("Sample League") == MarketEfficiencyTier.LOW


def test_required_edge_multiplier_ordering():
    classifier = MarketEfficiencyClassifier()
    high = classifier.required_edge_multiplier(MarketEfficiencyTier.HIGH)
    medium = classifier.required_edge_multiplier(MarketEfficiencyTier.MEDIUM)
    low = classifier.required_edge_multiplier(MarketEfficiencyTier.LOW)
    assert high > medium > low


def test_risk_contribution_ordering():
    classifier = MarketEfficiencyClassifier()
    high = classifier.risk_contribution(MarketEfficiencyTier.HIGH)
    medium = classifier.risk_contribution(MarketEfficiencyTier.MEDIUM)
    low = classifier.risk_contribution(MarketEfficiencyTier.LOW)
    assert high > medium > low
    assert 0.0 <= low and high <= 1.0
