from __future__ import annotations

import pytest

from football_vnext.domain.statistics.dixon_coles import DixonColesEngine, DixonColesFitError
from football_vnext.sample_data import generate_sample_matches


def test_fit_raises_with_too_few_matches():
    engine = DixonColesEngine()
    with pytest.raises(DixonColesFitError):
        engine.fit([])


def test_predict_before_fit_raises():
    engine = DixonColesEngine()
    with pytest.raises(DixonColesFitError):
        engine.predict_match("X", "Arsenal", "Chelsea")


def test_fit_and_predict_end_to_end():
    matches = generate_sample_matches(n_rounds=20)
    engine = DixonColesEngine(xi=0.0018)
    engine.fit(matches)

    assert engine.rho is not None
    assert -1.0 < engine.rho < 1.0
    assert set(engine.team_strengths.keys()) == {m.home_team_id for m in matches} | {
        m.away_team_id for m in matches
    }

    prediction = engine.predict_match("T1", "Arsenal", "Chelsea")
    total = prediction.home_win_prob + prediction.draw_prob + prediction.away_win_prob
    assert abs(total - 1.0) < 1e-6
    assert 0.0 <= prediction.home_win_prob <= 1.0


def test_predict_unknown_team_raises():
    matches = generate_sample_matches(n_rounds=20)
    engine = DixonColesEngine()
    engine.fit(matches)
    with pytest.raises(ValueError):
        engine.predict_match("T2", "Arsenal", "Nonexistent FC")


def test_team_match_counts_tracked_after_fit():
    matches = generate_sample_matches(n_rounds=20)
    engine = DixonColesEngine()
    engine.fit(matches)

    expected_arsenal_count = sum(
        1 for m in matches if m.home_team_id == "Arsenal" or m.away_team_id == "Arsenal"
    )
    assert engine.team_match_counts["Arsenal"] == expected_arsenal_count
    assert sum(engine.team_match_counts.values()) == 2 * len(matches)
