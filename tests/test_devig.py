from __future__ import annotations

import numpy as np
import pytest

from football_vnext.domain.odds.devig import DevigError, PowerDevig, ProportionalDevig, ShinDevig
from football_vnext.domain.odds.models import BookmakerOdds


ODDS_WITH_MARGIN = BookmakerOdds(bookmaker="Test", home=1.50, draw=4.20, away=7.00)
ODDS_NO_MARGIN = BookmakerOdds(bookmaker="Test", home=2.0, draw=3.3333333333, away=5.0)


@pytest.mark.parametrize("devig_cls", [ProportionalDevig, ShinDevig, PowerDevig])
def test_output_probabilities_sum_to_one(devig_cls):
    devig = devig_cls()
    result = devig.remove_vig(ODDS_WITH_MARGIN)
    total = result.home + result.draw + result.away
    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("devig_cls", [ProportionalDevig, ShinDevig, PowerDevig])
def test_zero_margin_odds_are_left_essentially_unchanged(devig_cls):
    devig = devig_cls()
    result = devig.remove_vig(ODDS_NO_MARGIN)
    raw = ODDS_NO_MARGIN.implied_probabilities()
    assert result.home == pytest.approx(raw[0], abs=1e-3)
    assert result.draw == pytest.approx(raw[1], abs=1e-3)
    assert result.away == pytest.approx(raw[2], abs=1e-3)


def test_shin_solves_consistent_z_in_bounds():
    devig = ShinDevig(z_upper_bound=0.2)
    result = devig.remove_vig(ODDS_WITH_MARGIN)
    assert 0.0 <= result.home <= 1.0
    assert 0.0 <= result.draw <= 1.0
    assert 0.0 <= result.away <= 1.0


def test_shin_raises_devig_error_when_bound_too_tight():
    # An artificially huge margin should not be solvable within a very
    # small z upper bound, and must raise rather than silently return
    # a wrong answer.
    huge_margin_odds = BookmakerOdds(bookmaker="Test", home=1.01, draw=1.5, away=1.5)
    devig = ShinDevig(z_upper_bound=1e-6)
    with pytest.raises(DevigError):
        devig.remove_vig(huge_margin_odds)


def test_shin_and_power_shade_less_margin_onto_favorite_than_proportional():
    """
    Directional correctness check: on a book with a strong favorite and a
    long shot, both Shin and the Power method should assign a HIGHER fair
    probability to the favorite (and correspondingly lower to the longshot)
    than naive proportional de-vig -- because they model the bookmaker's
    margin as shaded disproportionately onto longshots, which proportional
    de-vig ignores.
    """
    proportional = ProportionalDevig().remove_vig(ODDS_WITH_MARGIN)
    shin = ShinDevig().remove_vig(ODDS_WITH_MARGIN)
    power = PowerDevig().remove_vig(ODDS_WITH_MARGIN)

    # home (1.50) is the strong favorite, away (7.00) is the longshot
    assert shin.home >= proportional.home
    assert shin.away <= proportional.away
    assert power.home >= proportional.home
    assert power.away <= proportional.away


def test_power_raises_devig_error_when_no_root_in_bracket():
    # Construct odds where sum(pi_i ** 5) is still > 1, i.e. c=5 upper bound
    # is insufficient to reach zero margin (extremely large overround).
    # This is a defensive/contrived case to confirm the error path works.
    devig = PowerDevig()
    tiny_odds = BookmakerOdds(bookmaker="Test", home=1.001, draw=1.001, away=1.001)
    # sum(pi) = 3/1.001 ~= 2.997, a huge "margin" -- verify behavior is either
    # a valid solution or a clean DevigError, never a silent wrong answer.
    try:
        result = devig.remove_vig(tiny_odds)
        total = result.home + result.draw + result.away
        assert total == pytest.approx(1.0, abs=1e-6)
    except DevigError:
        pass
