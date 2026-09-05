from football_vnext.domain.features.context import TeamContext, apply_context


def test_worse_defense_increases_opponent_lambda():
    home, away = apply_context(
        1.5, 1.0,
        home=TeamContext(defense_multiplier=0.8),
        away=None,
    )
    assert home == 1.5
    assert away > 1.0


def test_stronger_defense_decreases_opponent_lambda():
    home, away = apply_context(
        1.5, 1.0,
        home=TeamContext(defense_multiplier=1.2),
    )
    assert home == 1.5
    assert away < 1.0


def test_attack_and_opponent_defense_are_combined():
    home, away = apply_context(
        1.0, 1.0,
        home=TeamContext(attack_multiplier=1.1, defense_multiplier=0.9),
        away=TeamContext(attack_multiplier=0.8, defense_multiplier=1.1),
    )
    assert home == 1.0
    assert away < 1.0
