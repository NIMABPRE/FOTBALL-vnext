from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from football_vnext.domain.features.news_impact import (
    NewsAnalysisError,
    NewsImpactAnalyzer,
    NewsImpactAssessment,
)


def _mock_anthropic_response(json_payload: dict):
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(json_payload)
    response = MagicMock()
    response.content = [block]
    return response


def test_rejects_empty_api_key():
    with pytest.raises(ValueError):
        NewsImpactAnalyzer(api_key="")


def test_rejects_invalid_min_confidence():
    with pytest.raises(ValueError):
        NewsImpactAnalyzer(api_key="x", min_confidence_to_apply=1.5)


def test_assessment_rejects_out_of_bounds_multiplier():
    with pytest.raises(NewsAnalysisError):
        NewsImpactAssessment(
            team_name="Arsenal", attack_multiplier=1.5, defense_multiplier=1.0,
            confidence=0.8, reasoning="test",
        )


def test_assessment_rejects_out_of_bounds_confidence():
    with pytest.raises(NewsAnalysisError):
        NewsImpactAssessment(
            team_name="Arsenal", attack_multiplier=1.0, defense_multiplier=1.0,
            confidence=1.5, reasoning="test",
        )


def test_rejects_empty_news_text():
    analyzer = NewsImpactAnalyzer(api_key="test-key")
    with pytest.raises(ValueError):
        analyzer.assess("Arsenal", "")


@patch("football_vnext.domain.features.news_impact.anthropic")
def test_assess_parses_valid_response(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(
        {
            "attack_multiplier": 0.90,
            "defense_multiplier": 0.95,
            "confidence": 0.8,
            "reasoning": "Starting striker injured, out for the match.",
        }
    )
    mock_anthropic_cls.Anthropic.return_value = mock_client

    analyzer = NewsImpactAnalyzer(api_key="test-key")
    result = analyzer.assess("Arsenal", "Starting striker ruled out with hamstring injury.")

    assert result.team_name == "Arsenal"
    assert result.attack_multiplier == pytest.approx(0.90)
    assert result.defense_multiplier == pytest.approx(0.95)
    assert result.confidence == pytest.approx(0.8)
    assert "injured" in result.reasoning or "injury" in result.reasoning.lower() or len(result.reasoning) > 0


@patch("football_vnext.domain.features.news_impact.anthropic")
def test_assess_strips_markdown_code_fences(mock_anthropic_cls):
    mock_client = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = '```json\n{"attack_multiplier": 1.0, "defense_multiplier": 1.0, "confidence": 0.1, "reasoning": "Nothing notable."}\n```'
    response = MagicMock()
    response.content = [block]
    mock_client.messages.create.return_value = response
    mock_anthropic_cls.Anthropic.return_value = mock_client

    analyzer = NewsImpactAnalyzer(api_key="test-key")
    result = analyzer.assess("Chelsea", "No news of note this week.")

    assert result.attack_multiplier == 1.0
    assert result.confidence == pytest.approx(0.1)


@patch("football_vnext.domain.features.news_impact.anthropic")
def test_raises_on_unparseable_json(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response({})  # placeholder, overridden below
    block = MagicMock()
    block.type = "text"
    block.text = "I'm not sure, here's some prose instead of JSON."
    response = MagicMock()
    response.content = [block]
    mock_client.messages.create.return_value = response
    mock_anthropic_cls.Anthropic.return_value = mock_client

    analyzer = NewsImpactAnalyzer(api_key="test-key")
    with pytest.raises(NewsAnalysisError):
        analyzer.assess("Arsenal", "Some news")


@patch("football_vnext.domain.features.news_impact.anthropic")
def test_raises_on_missing_required_field(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(
        {"attack_multiplier": 1.0}  # missing defense_multiplier and confidence
    )
    mock_anthropic_cls.Anthropic.return_value = mock_client

    analyzer = NewsImpactAnalyzer(api_key="test-key")
    with pytest.raises(NewsAnalysisError):
        analyzer.assess("Arsenal", "Some news")


@patch("football_vnext.domain.features.news_impact.anthropic")
def test_raises_on_out_of_bounds_llm_value(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(
        {"attack_multiplier": 2.0, "defense_multiplier": 1.0, "confidence": 0.5, "reasoning": "test"}
    )
    mock_anthropic_cls.Anthropic.return_value = mock_client

    analyzer = NewsImpactAnalyzer(api_key="test-key")
    with pytest.raises(NewsAnalysisError):
        analyzer.assess("Arsenal", "Some dramatic news")


@patch("football_vnext.domain.features.news_impact.anthropic")
def test_raises_on_empty_content(mock_anthropic_cls):
    mock_client = MagicMock()
    response = MagicMock()
    response.content = []
    mock_client.messages.create.return_value = response
    mock_anthropic_cls.Anthropic.return_value = mock_client

    analyzer = NewsImpactAnalyzer(api_key="test-key")
    with pytest.raises(NewsAnalysisError):
        analyzer.assess("Arsenal", "Some news")


def test_apply_news_adjustments_increases_lambda_for_positive_attack():
    from football_vnext.domain.features.news_impact import apply_news_adjustments
    from football_vnext.domain.statistics.dixon_coles import DixonColesEngine
    from football_vnext.sample_data import generate_sample_matches

    matches = generate_sample_matches(n_rounds=20, seed=1)
    engine = DixonColesEngine()
    engine.fit(matches)

    base_pred = engine.predict_match("T1", "Arsenal", "Chelsea")

    boost = NewsImpactAssessment(
        team_name="Arsenal", attack_multiplier=1.10, defense_multiplier=1.0,
        confidence=0.9, reasoning="Key players returning from injury.",
    )
    adjusted_pred = apply_news_adjustments(
        engine, "T1", "Arsenal", "Chelsea",
        base_pred.lambda_home, base_pred.lambda_away,
        home_assessment=boost, away_assessment=None,
    )

    assert adjusted_pred.lambda_home > base_pred.lambda_home
    assert adjusted_pred.lambda_away == pytest.approx(base_pred.lambda_away)
    assert adjusted_pred.model_name == "dixon_coles+news_adjustment"


def test_apply_news_adjustments_ignores_low_confidence():
    from football_vnext.domain.features.news_impact import apply_news_adjustments
    from football_vnext.domain.statistics.dixon_coles import DixonColesEngine
    from football_vnext.sample_data import generate_sample_matches

    matches = generate_sample_matches(n_rounds=20, seed=1)
    engine = DixonColesEngine()
    engine.fit(matches)
    base_pred = engine.predict_match("T1", "Arsenal", "Chelsea")

    weak_signal = NewsImpactAssessment(
        team_name="Arsenal", attack_multiplier=1.10, defense_multiplier=1.0,
        confidence=0.2, reasoning="Vague rumor, unconfirmed.",
    )
    adjusted_pred = apply_news_adjustments(
        engine, "T1", "Arsenal", "Chelsea",
        base_pred.lambda_home, base_pred.lambda_away,
        home_assessment=weak_signal, away_assessment=None,
        min_confidence_to_apply=0.5,
    )

    assert adjusted_pred.lambda_home == pytest.approx(base_pred.lambda_home)


def test_apply_news_adjustments_worse_defense_boosts_opponent_lambda():
    from football_vnext.domain.features.news_impact import apply_news_adjustments
    from football_vnext.domain.statistics.dixon_coles import DixonColesEngine
    from football_vnext.sample_data import generate_sample_matches

    matches = generate_sample_matches(n_rounds=20, seed=1)
    engine = DixonColesEngine()
    engine.fit(matches)
    base_pred = engine.predict_match("T1", "Arsenal", "Chelsea")

    weak_defense = NewsImpactAssessment(
        team_name="Arsenal", attack_multiplier=1.0, defense_multiplier=0.90,
        confidence=0.9, reasoning="Two starting defenders injured.",
    )
    adjusted_pred = apply_news_adjustments(
        engine, "T1", "Arsenal", "Chelsea",
        base_pred.lambda_home, base_pred.lambda_away,
        home_assessment=weak_defense, away_assessment=None,
    )

    # Arsenal's (home) worse defense should INCREASE Chelsea's (away) lambda
    assert adjusted_pred.lambda_away > base_pred.lambda_away
    assert adjusted_pred.lambda_home == pytest.approx(base_pred.lambda_home)
