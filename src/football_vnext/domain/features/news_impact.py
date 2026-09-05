"""
News/Injury Impact Analyzer — an OPTIONAL, AUXILIARY signal using an LLM.

WHY THIS EXISTS AND WHY IT'S BOUNDED THE WAY IT IS: earlier in this project
the question came up of whether to put an LLM at the core of the prediction
engine. The answer was no — classical statistics (Dixon-Coles, MLE fitting,
de-vig, Kelly) are more accurate, explainable, and backtestable than an LLM
for the core probability/staking math, and that hasn't changed. What an LLM
CAN legitimately help with is turning unstructured text (team news, injury
reports) into a small, structured, bounded numeric signal that a human analyst
would otherwise have to read and estimate by hand. This module does exactly
that and nothing more:

  - Input: raw text about a team's current news/injuries/suspensions.
  - Output: a small multiplicative adjustment to that team's attack/defence
    strength (clamped to +/-15% by default), a confidence score, and the
    model's stated reasoning (for a human to sanity-check, not to blindly trust).

This adjustment is applied (if at all) BEFORE the Dixon-Coles lambda
calculation, as an explicit, visible, optional multiplier — never silently,
never overriding the statistical model, and OFF BY DEFAULT everywhere it's
wired in. A malformed or nonsensical LLM response is refused, not silently
clamped into something plausible-looking, because a fabricated-but-plausible
adjustment is more dangerous than an obvious failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

try:
    import anthropic
except ImportError:  # optional dependency; only required for live LLM news analysis
    anthropic = None

logger = logging.getLogger(__name__)

# Hard bounds enforced in code regardless of what the LLM returns -- this is
# defense in depth, not a substitute for the prompt's own instructions.
_MIN_MULTIPLIER = 0.85
_MAX_MULTIPLIER = 1.15

_SYSTEM_PROMPT = """You are a football (soccer) team-news analyst. You will be given a team name and raw text describing recent news about that team (injuries, suspensions, transfers, morale, fixture congestion, etc.).

Assess how this news should adjust the team's expected attacking and defensive strength for their NEXT match, relative to their normal baseline (1.0 = no change).

Respond with ONLY a JSON object, no other text, in exactly this shape:
{"attack_multiplier": <float between 0.85 and 1.15>, "defense_multiplier": <float between 0.85 and 1.15>, "confidence": <float between 0.0 and 1.0>, "reasoning": "<one or two sentences>"}

Guidance:
- attack_multiplier < 1.0 means weaker attack than normal (e.g. key striker injured).
- defense_multiplier < 1.0 means WORSE defense than normal (e.g. key defender injured) -- lower is worse, consistent with attack_multiplier's direction (lower = worse for that team).
- If the news is neutral, minor, or you are unsure, use multipliers close to 1.0 and a LOW confidence score. Do not manufacture a strong signal from weak or ambiguous evidence.
- confidence should reflect how much the news text actually supports the adjustment, not how confident you are in your own reasoning process."""


class NewsAnalysisError(Exception):
    """Raised when the LLM response is missing, malformed, or out of bounds
    in a way that can't be safely corrected -- refused, not guessed at."""


@dataclass(frozen=True)
class NewsImpactAssessment:
    team_name: str
    attack_multiplier: float
    defense_multiplier: float
    confidence: float
    reasoning: str

    def __post_init__(self) -> None:
        for name, value in (
            ("attack_multiplier", self.attack_multiplier),
            ("defense_multiplier", self.defense_multiplier),
        ):
            if not (_MIN_MULTIPLIER <= value <= _MAX_MULTIPLIER):
                raise NewsAnalysisError(
                    f"{name}={value} outside allowed bounds "
                    f"[{_MIN_MULTIPLIER}, {_MAX_MULTIPLIER}]"
                )
        if not (0.0 <= self.confidence <= 1.0):
            raise NewsAnalysisError(f"confidence={self.confidence} outside [0.0, 1.0]")


class NewsImpactAnalyzer:
    """
    :param api_key: Anthropic API key
    :param model: model string, e.g. "claude-sonnet-5"
    :param min_confidence_to_apply: assessments below this confidence should
        be treated by the CALLER as "no adjustment" -- this class still
        returns them (so the caller can see the reasoning), it just surfaces
        confidence for the caller to act on. This class does not decide for
        the caller whether to apply the adjustment.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        min_confidence_to_apply: float = 0.5,
        max_tokens: int = 300,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("An Anthropic API key is required for NewsImpactAnalyzer.")
        if not (0.0 <= min_confidence_to_apply <= 1.0):
            raise ValueError("min_confidence_to_apply must be in [0.0, 1.0]")
        self.model = model
        self.min_confidence_to_apply = min_confidence_to_apply
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key) if anthropic is not None else None

    def assess(self, team_name: str, news_text: str) -> NewsImpactAssessment:
        if not news_text or not news_text.strip():
            raise ValueError("news_text cannot be empty")
        if anthropic is None:
            raise NewsAnalysisError("Anthropic package is not installed; install the optional anthropic dependency to enable LLM news analysis.")

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Team: {team_name}\n\nNews:\n{news_text.strip()}",
                    }
                ],
            )
        except anthropic.APIError as exc:
            raise NewsAnalysisError(f"Anthropic API call failed: {exc}") from exc

        raw_text = self._extract_text(response)
        parsed = self._parse_json(raw_text)

        try:
            return NewsImpactAssessment(
                team_name=team_name,
                attack_multiplier=float(parsed["attack_multiplier"]),
                defense_multiplier=float(parsed["defense_multiplier"]),
                confidence=float(parsed["confidence"]),
                reasoning=str(parsed.get("reasoning", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NewsAnalysisError(f"LLM response missing/invalid required field: {exc}") from exc

    @staticmethod
    def _extract_text(response) -> str:
        text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        if not text_blocks:
            raise NewsAnalysisError("Anthropic response contained no text content.")
        return "".join(text_blocks)

    @staticmethod
    def _parse_json(raw_text: str) -> dict:
        cleaned = raw_text.strip()
        # Defensive: strip markdown code fences if the model added them
        # despite instructions not to.
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise NewsAnalysisError(
                f"Could not parse LLM response as JSON: {exc}. Raw: {raw_text[:200]}"
            ) from exc


def apply_news_adjustments(
    engine,  # PoissonEngine or DixonColesEngine -- duck-typed to avoid a
             # domain.statistics -> domain.features dependency in that direction
    match_id: str,
    home_team_name: str,
    away_team_name: str,
    base_lambda_home: float,
    base_lambda_away: float,
    home_assessment=None,  # Optional[NewsImpactAssessment]
    away_assessment=None,  # Optional[NewsImpactAssessment]
    min_confidence_to_apply: float = 0.5,
):
    """
    Applies news-based attack/defence multipliers to the statistical model's
    OWN lambda estimates and rebuilds the prediction from the adjusted
    lambdas -- this is a visible, auditable adjustment applied on top of the
    real Dixon-Coles output, never a replacement for it. Assessments below
    `min_confidence_to_apply` are ignored (logged, not applied) rather than
    silently included at partial weight, since a low-confidence LLM read of
    ambiguous news is not information worth acting on.

    A team's OWN attack_multiplier scales its own lambda directly. A team's
    defense_multiplier (where <1.0 means WORSE defense) scales the OPPONENT's
    lambda inversely -- worse defense means the opponent is expected to score
    MORE, which is why this uses (1 / defense_multiplier), not the multiplier
    directly.
    """
    lambda_home = base_lambda_home
    lambda_away = base_lambda_away
    applied_adjustments = []

    if home_assessment is not None:
        if home_assessment.confidence >= min_confidence_to_apply:
            lambda_home *= home_assessment.attack_multiplier
            lambda_away *= 1.0 / home_assessment.defense_multiplier
            applied_adjustments.append(f"home attack x{home_assessment.attack_multiplier:.3f}, "
                                        f"home defense x{home_assessment.defense_multiplier:.3f}")
        else:
            logger.info(
                "Ignoring home news assessment for %s: confidence %.2f below threshold %.2f",
                home_team_name, home_assessment.confidence, min_confidence_to_apply,
            )

    if away_assessment is not None:
        if away_assessment.confidence >= min_confidence_to_apply:
            lambda_away *= away_assessment.attack_multiplier
            lambda_home *= 1.0 / away_assessment.defense_multiplier
            applied_adjustments.append(f"away attack x{away_assessment.attack_multiplier:.3f}, "
                                        f"away defense x{away_assessment.defense_multiplier:.3f}")
        else:
            logger.info(
                "Ignoring away news assessment for %s: confidence %.2f below threshold %.2f",
                away_team_name, away_assessment.confidence, min_confidence_to_apply,
            )

    if not applied_adjustments:
        logger.info("No news adjustments applied (none provided or all below confidence threshold).")
    else:
        logger.info("Applied news adjustments: %s", "; ".join(applied_adjustments))

    matrix = engine.scoreline_matrix(lambda_home, lambda_away)
    return engine._matrix_to_prediction(
        match_id, home_team_name, away_team_name, lambda_home, lambda_away, matrix,
        model_name="dixon_coles+news_adjustment",
    )
