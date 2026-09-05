"""xG feature layer used as a bounded correction to the goal-based model.

xG never replaces Dixon-Coles.  It corrects attack/defence strength when a
team's underlying chance creation/prevention differs materially from its
observed goals.  Only matches strictly inside the model fit window are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Optional
import math

from football_vnext.domain.models.match import Match


@dataclass(frozen=True)
class XGTeamSignal:
    attack_ratio: float
    defence_ratio: float
    samples: int


class XGFeatureEngine:
    def __init__(self, max_adjustment: float = 0.15, prior_weight: float = 5.0) -> None:
        self.max_adjustment = max_adjustment
        self.prior_weight = prior_weight
        self.signals: Dict[str, XGTeamSignal] = {}
        self.league_xg_for = 1.35
        self.league_xg_against = 1.35

    def fit(self, matches: Iterable[Match], ref_date: Optional[datetime] = None) -> None:
        rows = [m for m in matches if m.result is not None and m.home_xg is not None and m.away_xg is not None]
        if ref_date is not None:
            rows = [m for m in rows if m.kickoff < ref_date]
        if not rows:
            self.signals = {}
            return

        self.league_xg_for = sum(float(m.home_xg) + float(m.away_xg) for m in rows) / (2 * len(rows))
        self.league_xg_against = self.league_xg_for
        agg: Dict[str, list[float]] = {}
        for m in rows:
            agg.setdefault(m.home_team_id, [0.0, 0.0, 0.0, 0.0])
            agg.setdefault(m.away_team_id, [0.0, 0.0, 0.0, 0.0])
            # xG for, xG against, observed goals for, observed goals against
            agg[m.home_team_id][0] += float(m.home_xg)
            agg[m.home_team_id][1] += float(m.away_xg)
            agg[m.home_team_id][2] += m.result.home_goals
            agg[m.home_team_id][3] += m.result.away_goals
            agg[m.away_team_id][0] += float(m.away_xg)
            agg[m.away_team_id][1] += float(m.home_xg)
            agg[m.away_team_id][2] += m.result.away_goals
            agg[m.away_team_id][3] += m.result.home_goals

        out: Dict[str, XGTeamSignal] = {}
        counts: Dict[str, int] = {}
        for m in rows:
            counts[m.home_team_id] = counts.get(m.home_team_id, 0) + 1
            counts[m.away_team_id] = counts.get(m.away_team_id, 0) + 1
        for team, (xgf, xga, gf, ga) in agg.items():
            n = counts.get(team, 0)
            prior = self.prior_weight
            xgf_rate = (xgf + prior * self.league_xg_for) / (n + prior)
            xga_rate = (xga + prior * self.league_xg_against) / (n + prior)
            gf_rate = (gf + prior * self.league_xg_for) / (n + prior)
            ga_rate = (ga + prior * self.league_xg_against) / (n + prior)
            attack = math.sqrt(max(0.5, min(1.5, xgf_rate / max(gf_rate, 1e-6))))
            defence = math.sqrt(max(0.5, min(1.5, xga_rate / max(ga_rate, 1e-6))))
            attack = min(1 + self.max_adjustment, max(1 - self.max_adjustment, attack))
            defence = min(1 + self.max_adjustment, max(1 - self.max_adjustment, defence))
            out[team] = XGTeamSignal(attack, defence, n)
        self.signals = out

    def multipliers(self, home_team_id: str, away_team_id: str) -> tuple[float, float]:
        h = self.signals.get(home_team_id)
        a = self.signals.get(away_team_id)
        if h is None or a is None:
            return 1.0, 1.0
        # Own attack scales own lambda directly.
        #
        # Opponent's defence_ratio scales this team's lambda DIRECTLY (not
        # inversely) -- unlike context.py's TeamContext.defense_multiplier
        # (where <1.0 always means "worse"), this ratio is defined as
        # sqrt(xG_against_rate / goals_against_rate) in fit() above, so
        # ratio > 1.0 means the team has been CONCEDING FEWER GOALS THAN
        # THEIR UNDERLYING xG-AGAINST SUGGESTS (i.e. they've been lucky /
        # their TRUE defence is WORSE than recent results show, and should
        # regress toward conceding MORE). So a HIGHER defence_ratio here
        # correctly means MORE expected goals for the opponent -- direct
        # multiplication is correct for this specific ratio's definition.
        # (Bug history: an earlier attempt "fixed" this to divide, by
        # wrongly assuming the same <1=worse convention as context.py's
        # TeamContext -- that convention does NOT apply to this ratio.)
        home_lambda = h.attack_ratio * a.defence_ratio
        away_lambda = a.attack_ratio * h.defence_ratio
        return (
            min(1 + self.max_adjustment, max(1 - self.max_adjustment, home_lambda)),
            min(1 + self.max_adjustment, max(1 - self.max_adjustment, away_lambda)),
        )
