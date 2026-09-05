"""
The GATE. Everything upstream (Dixon-Coles, calibration, de-vig, Edge/EV,
risk score, Kelly) exists to produce bet suggestions; this module is what
decides whether those suggestions have earned the right to be bet with real
money. It is deliberately conservative and deliberately NOT a single "good
enough" score — each criterion below can independently veto going live,
because each one catches a different way a backtest can look good while
being worthless (or dangerous) in production:

- Too few bets       -> any result is statistical noise, full stop.
- CLV not significant -> the core evidence of genuine edge is missing, even
                         if ROI happens to look good in this particular run.
- ROI <= 0           -> no point pretending Edge/EV meant anything.
- Drawdown too deep  -> even a profitable strategy is useless if nobody
                         (human or account) survives the ride to get there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from football_vnext.domain.backtest.clv import CLVSummary
from football_vnext.domain.backtest.metrics import BacktestMetrics


class GateError(Exception):
    """Raised on invalid gate configuration."""


@dataclass(frozen=True)
class BacktestGateConfig:
    min_bets: int = 300
    min_roi: float = 0.0
    max_acceptable_drawdown: float = 0.30
    require_significant_clv: bool = True
    require_significant_roi: bool = False  # ROI significance is a nice-to-have, CLV is the real bar

    def __post_init__(self) -> None:
        if self.min_bets < 1:
            raise GateError("min_bets must be >= 1")
        if not (0.0 < self.max_acceptable_drawdown <= 1.0):
            raise GateError("max_acceptable_drawdown must be in (0, 1]")


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons_for_failure: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.passed:
            return "GateResult(PASSED — cleared for live execution)"
        reasons = "; ".join(self.reasons_for_failure)
        return f"GateResult(FAILED — {reasons})"


class BacktestGate:
    def __init__(self, config: Optional[BacktestGateConfig] = None) -> None:
        self.config = config or BacktestGateConfig()

    def evaluate(self, metrics: BacktestMetrics, clv_summary: CLVSummary) -> GateResult:
        reasons: List[str] = []

        if metrics.n_bets < self.config.min_bets:
            reasons.append(
                f"only {metrics.n_bets} bets, need at least {self.config.min_bets} "
                f"for a statistically meaningful sample"
            )

        if self.config.require_significant_clv and not clv_summary.clv_significantly_positive:
            reasons.append(
                f"CLV not significantly positive (mean={clv_summary.mean_clv:+.2%}, "
                f"p={clv_summary.p_value:.4f}) — the core evidence of genuine edge is missing"
            )

        if metrics.roi <= self.config.min_roi:
            reasons.append(f"ROI {metrics.roi:+.2%} does not exceed minimum {self.config.min_roi:+.2%}")

        if self.config.require_significant_roi and not metrics.returns_significantly_positive:
            reasons.append(f"ROI not statistically significant (p={metrics.p_value:.4f})")

        if metrics.max_drawdown > self.config.max_acceptable_drawdown:
            reasons.append(
                f"max drawdown {metrics.max_drawdown:.2%} exceeds acceptable "
                f"{self.config.max_acceptable_drawdown:.2%}"
            )

        return GateResult(passed=(len(reasons) == 0), reasons_for_failure=reasons)
