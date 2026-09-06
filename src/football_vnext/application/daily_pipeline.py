"""Production-oriented daily prediction pipeline.

Data path:
  historical goals/odds -> Understat xG enrichment -> Dixon-Coles+xG
  -> chronological calibration -> live multi-bookmaker odds -> Shin
  -> optional structured injuries/lineups + bounded LLM news -> Edge/EV
  -> Risk -> fractional Kelly.

All auxiliary sources are fail-closed: unavailable data means the adjustment
is omitted and the record is marked as lower confidence; no synthetic data is
created.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
import re

from football_vnext.config import Settings
from football_vnext.domain.models.match import Match, MatchStatus
from football_vnext.domain.models.probability import OutcomeProbabilities
from football_vnext.domain.odds.aggregation import AggregationMethod, OddsAggregator
from football_vnext.domain.odds.devig import ShinDevig
from football_vnext.domain.odds.models import MatchOddsQuote
from football_vnext.domain.statistics.calibration import CalibrationSample, ProbabilityCalibrator
from football_vnext.domain.statistics.dixon_coles import DixonColesEngine, DixonColesFitError
from football_vnext.domain.value.edge import EdgeDetector, Outcome, build_candidates_from_predictions
from football_vnext.domain.value.kelly import PortfolioKellyStaker
from football_vnext.domain.value.market_efficiency import MarketEfficiencyClassifier
from football_vnext.domain.value.risk import RiskScoreCalculator
from football_vnext.infrastructure.data_sources.football_data_co_uk import FootballDataCoUkAdapter
from football_vnext.infrastructure.data_sources.odds_api import TheOddsApiAdapter
from football_vnext.infrastructure.data_sources.understat import UnderstatAdapter
from football_vnext.infrastructure.data_sources.news import NewsRSSAdapter
from football_vnext.infrastructure.data_sources.api_football import ApiFootballAdapter
from football_vnext.infrastructure.data_sources.team_context import TeamContextAdapter
from football_vnext.domain.features.news_impact import NewsImpactAnalyzer, NewsAnalysisError

LEAGUES={
 'Premier League': {'sport_key':'soccer_epl','history_code':'E0','api_league':39},
 'La Liga': {'sport_key':'soccer_spain_la_liga','history_code':'SP1','api_league':140},
 'Serie A': {'sport_key':'soccer_italy_serie_a','history_code':'I1','api_league':135},
 'Bundesliga': {'sport_key':'soccer_germany_bundesliga','history_code':'D1','api_league':78},
 'Ligue 1': {'sport_key':'soccer_france_ligue_one','history_code':'F1','api_league':61},
}

def norm(name:str)->str:
    s=re.sub(r"[^a-z0-9\\s]", "", name.lower())
    return ' '.join(x for x in s.split() if x not in {'fc','cf','afc','sc','ac','cd','the'})

def resolve(name:str, known:dict[str,str])->str|None:
    n=norm(name)
    if n in known:return known[n]
    aliases={'manchester united':['man utd','man united','manchester utd'],'manchester city':['man city'],'tottenham hotspur':['tottenham','spurs'],'newcastle united':['newcastle'],'wolverhampton wanderers':['wolves','wolverhampton'],'nottingham forest':["nott'm forest",'nottingham']}
    cand={n}|{norm(x) for x in aliases.get(n,[])}
    ids={tid for k,tid in known.items() if k in cand}
    return next(iter(ids)) if len(ids)==1 else None

def _season_start_year(season: str) -> int | None:
    s = str(season).strip()
    m = re.fullmatch(r'(\d{2})(\d{2})', s)
    if m:
        return 2000 + int(m.group(1))
    m = re.search(r'(20\d{2})', s)
    return int(m.group(1)) if m else None

def enrich_xg(records, league_code):
    ua=UnderstatAdapter(); by_key={}
    seasons=sorted({y for y in (_season_start_year(r.match.season) for r in records) if y is not None})
    for y in seasons:
        for row in ua.fetch_league_matches(league_code,y): by_key[(row['date'],norm(row['home_team']),norm(row['away_team']))]=row
    enriched=0
    for r in records:
        m=r.match; row=by_key.get((m.kickoff.date(),norm(m.home_team_name),norm(m.away_team_name)))
        if row:
            r.match=m.model_copy(update={'home_xg':row['home_xg'],'away_xg':row['away_xg']}); enriched += 1
    return records, enriched

def fit_calibrator(records, xi):
    if len(records)<100:return None,'insufficient history'
    split=int(len(records)*.8); early=records[:split]; hold=records[split:]
    try:
        e=DixonColesEngine(xi=xi); e.fit([r.match for r in early])
    except DixonColesFitError as exc:return None,str(exc)
    devig=ShinDevig(); samples=[]
    for r in hold:
        try:p=e.predict_match('cal',r.match.home_team_id,r.match.away_team_id)
        except ValueError:continue
        samples.append(CalibrationSample(OutcomeProbabilities(home=p.home_win_prob,draw=p.draw_prob,away=p.away_win_prob),devig.remove_vig(r.opening_odds),r.match.result.outcome))
    if len(samples)<30:return None,f'only {len(samples)} calibration observations'
    c=ProbabilityCalibrator(min_samples=30); c.fit(samples)
    return c,None

def _context(settings:Settings, league_cfg:dict, target_date, home:str, away:str):
    home_mult=away_mult=1.0; notes=[]; ids={}
    # Structured injuries/lineups. API-Football is optional and fail-closed.
    if settings.api_football_key:
        try:
            api=ApiFootballAdapter(settings.api_football_key); tc=TeamContextAdapter(api)
            season_year = target_date.year if target_date.month >= 7 else target_date.year - 1
            team_ids=tc.fetch_team_ids(league_cfg['api_league'],season_year)
            for n,tid in team_ids.items(): ids[norm(n)]=tid
            hid=resolve(home,ids); aid=resolve(away,ids)
            fixture_id=tc.find_fixture_id(league_cfg['api_league'],season_year,target_date,home,away)
            lineups=tc.fetch_lineups(fixture_id) if fixture_id else {}
            if lineups: notes.append('confirmed lineups loaded')
            for label,tid in [('home',hid),('away',aid)]:
                if tid:
                    av=tc.fetch_injuries(tid,league_cfg['api_league'],season_year)
                    starters={x.lower() for x in lineups.get(tid,[])}
                    # Do not count an injury report as an absence if the player
                    # is actually named in the confirmed starting XI.
                    effective=[a for a in av.absences if a.player_name.lower() not in starters]
                    impact=min(0.08,0.015*len(effective))
                    if label=='home': home_mult*=1-impact
                    else: away_mult*=1-impact
                    if effective: notes.append(f'{label}: {len(effective)} unconfirmed/absent players')
        except Exception as exc: notes.append(f'injury feed unavailable: {exc}')

    # Automated news -> existing bounded LLM analyzer. Never used without key.
    if settings.anthropic_api_key:
        try:
            rss=NewsRSSAdapter(); analyzer=NewsImpactAnalyzer(settings.anthropic_api_key)
            for label,name in [('home',home),('away',away)]:
                items=rss.search(name); text=rss.build_text(items)
                if not text:continue
                a=analyzer.assess(name,text)
                if a.confidence>=analyzer.min_confidence_to_apply:
                    if label=='home': home_mult*=a.attack_multiplier; away_mult*=1/a.defense_multiplier
                    else: away_mult*=a.attack_multiplier; home_mult*=1/a.defense_multiplier
                    notes.append(f'{label}: news confidence {a.confidence:.2f}')
        except (NewsAnalysisError,Exception) as exc: notes.append(f'news unavailable: {exc}')
    return max(.85,min(1.15,home_mult)),max(.85,min(1.15,away_mult)),notes

def load_daily_fixtures(settings: Settings, league_name: str, target_date, timezone_name: str = "Europe/Berlin"):
    """Return the day's fixtures without running the prediction pipeline.

    API-Football is preferred because it exposes fixtures even when no bookmaker
    quote exists. The Odds API is used as a fallback when API-Football is not
    configured. This keeps the dashboard useful as a fixture browser first.
    """
    from zoneinfo import ZoneInfo
    from datetime import timedelta
    cfg = LEAGUES[league_name]
    zone = ZoneInfo(timezone_name)
    local_start = datetime.combine(target_date, datetime.min.time(), tzinfo=zone)
    local_end = local_start + timedelta(days=1)
    fixtures = []
    if settings.api_football_key:
        api_season = target_date.year if target_date.month >= 7 else target_date.year - 1
        api = ApiFootballAdapter(settings.api_football_key)
        for m in api.fetch_fixtures(cfg["api_league"], api_season, date=target_date.isoformat()):
            kickoff = m.kickoff.astimezone(zone)
            if local_start <= kickoff < local_end:
                fixtures.append({
                    "key": f"{norm(m.home_team_name)}__{norm(m.away_team_name)}",
                    "match": f"{m.home_team_name} vs {m.away_team_name}",
                    "home": m.home_team_name,
                    "away": m.away_team_name,
                    "kickoff": m.kickoff.isoformat(),
                    "status": m.status.value,
                    "fixture_id": m.match_id,
                    "source": "API-Football",
                })
    else:
        if not settings.odds_api_key:
            raise ValueError("Configure The Odds API key or API-Football key to load fixtures.")
        adapter = TheOddsApiAdapter(settings.odds_api_key, sport_key=cfg["sport_key"], regions="eu,uk", markets="h2h")
        for q in adapter.fetch_odds():
            kickoff = q.commence_time.astimezone(zone)
            if local_start <= kickoff < local_end:
                fixtures.append({
                    "key": f"{norm(q.home_team_name)}__{norm(q.away_team_name)}",
                    "match": f"{q.home_team_name} vs {q.away_team_name}",
                    "home": q.home_team_name,
                    "away": q.away_team_name,
                    "kickoff": q.commence_time.isoformat(),
                    "status": "scheduled",
                    "fixture_id": None,
                    "source": "The Odds API",
                })
    fixtures.sort(key=lambda x: x["kickoff"])
    return fixtures


def run_daily(settings:Settings, league_name:str, target_date, seasons:tuple[str,...], xi:float=0.0018,min_edge:float=.03,min_ev:float=.03,max_risk:float=.60,kelly_fraction:float=.25,timezone_name:str='Europe/Berlin',selected_match_keys=None):
    cfg=LEAGUES[league_name]
    hist=FootballDataCoUkAdapter(timeout=20,max_retries=2); records=[]; errors=[]
    for s in seasons:
        try:records.extend(hist.fetch_historical_matches(cfg['history_code'],s,competition_name=league_name))
        except Exception as exc:errors.append(f'{s}: {exc}')
    records=[r for r in records if r.match.result is not None and r.opening_odds is not None]; records.sort(key=lambda r:r.match.kickoff)
    if len(records) < 20:
        detail = '; '.join(errors) if errors else 'historical source returned no usable settled matches'
        raise DixonColesFitError(
            f'Need at least 20 settled historical matches, got {len(records)}. {detail}'
        )
    records,xg_enriched=enrich_xg(records,cfg['history_code'])
    engine=DixonColesEngine(xi=xi); engine.fit([r.match for r in records])
    calibrator,cal_err=fit_calibrator(records,xi)
    if not settings.odds_api_key: raise ValueError('ODDS_API_KEY is required')
    quotes=TheOddsApiAdapter(settings.odds_api_key,sport_key=cfg['sport_key'],regions='eu,uk',markets='h2h').fetch_odds()
    from zoneinfo import ZoneInfo
    from datetime import timedelta
    zone=ZoneInfo(timezone_name)
    local_start=datetime.combine(target_date,datetime.min.time(),tzinfo=zone); local_end=local_start+timedelta(days=1)
    all_quotes=[q for q in quotes if local_start<=q.commence_time.astimezone(zone)<local_end]
    selected = set(selected_match_keys or [])
    quotes=[q for q in all_quotes if not selected or f"{norm(q.home_team_name)}__{norm(q.away_team_name)}" in selected]
    known={norm(r.match.home_team_name):r.match.home_team_id for r in records}; known.update({norm(r.match.away_team_name):r.match.away_team_id for r in records})
    fixture_lookup={}
    if selected and settings.api_football_key:
        try:
            for f in load_daily_fixtures(settings, league_name, target_date, timezone_name):
                fixture_lookup[f["key"]]=f
        except Exception as exc:
            errors.append(f"fixture refresh: {exc}")
    agg=OddsAggregator(AggregationMethod.MEDIAN); devig=ShinDevig(); detector=EdgeDetector(min_edge=min_edge,min_ev=min_ev); risk=RiskScoreCalculator(max_acceptable_risk_score=max_risk); staker=PortfolioKellyStaker(kelly_fraction=kelly_fraction,max_stake_per_bet=.05,max_total_exposure=.15)
    rows=[]; predictions=[]; processed_keys=set()
    for q in quotes:
        processed_keys.add(f"{norm(q.home_team_name)}__{norm(q.away_team_name)}")
        hid=resolve(q.home_team_name,known); aid=resolve(q.away_team_name,known)
        if not hid or not aid: continue
        pred=engine.predict_match(q.home_team_name+'-'+q.away_team_name,hid,aid)
        # Apply automated current-context multipliers after the statistical model.
        hm,am,notes=_context(settings,cfg,target_date,q.home_team_name,q.away_team_name)
        lh,la=pred.lambda_home*hm,pred.lambda_away*am
        matrix=engine.scoreline_matrix(lh,la); pred=engine._matrix_to_prediction(pred.match_id,hid,aid,lh,la,matrix,'dixon_coles+xg+context')
        model_p=OutcomeProbabilities(home=pred.home_win_prob,draw=pred.draw_prob,away=pred.away_win_prob)
        market=agg.aggregate(q.bookmaker_quotes); fair=devig.remove_vig(market); blended=calibrator.apply(model_p,fair) if calibrator else model_p
        predictions.append({'match':f'{q.home_team_name} vs {q.away_team_name}','kickoff':q.commence_time.isoformat(),'home_prob':blended.home,'draw_prob':blended.draw,'away_prob':blended.away,'model':'xG+context','bookmakers':len(q.bookmaker_quotes),'context':'; '.join(notes) or 'none'})
        signals=detector.detect(build_candidates_from_predictions(pred.match_id,blended,fair,home_odds=market.home,draw_odds=market.draw,away_odds=market.away))
        accepted=[]
        for sig in signals:
            a=risk.compute(q.bookmaker_quotes,sig.candidate.outcome,engine.team_match_counts.get(hid,0),engine.team_match_counts.get(aid,0),league_name)
            if a.is_acceptable: accepted.append((sig,a))
        recs=staker.compute_stakes([x[0] for x in accepted]) if accepted else []
        for rec,(_,ra) in zip(recs,accepted):
            out=rec.signal.candidate.outcome; label=q.home_team_name if out==Outcome.HOME else 'Draw' if out==Outcome.DRAW else q.away_team_name
            rows.append({'match':f'{q.home_team_name} vs {q.away_team_name}','kickoff':q.commence_time.isoformat(),'bet':label,'odds':rec.signal.candidate.decimal_odds,'probability':rec.signal.candidate.calibrated_prob,'fair_market':rec.signal.candidate.fair_market_prob,'edge':rec.signal.edge,'EV':rec.signal.expected_value,'risk':ra.risk_score,'stake':rec.stake_fraction_of_bankroll*(1-ra.risk_score),'bookmakers':len(q.bookmaker_quotes),'context':'; '.join(notes) or 'xG/statistical only'})

    # Selected fixtures with no bookmaker quote still receive a pure model
    # prediction. They simply cannot produce EV/edge/value-bet rows.
    for key, f in fixture_lookup.items():
        if key in processed_keys or (selected and key not in selected):
            continue
        hid=resolve(f["home"],known); aid=resolve(f["away"],known)
        if not hid or not aid:
            errors.append(f"unresolved selected fixture: {f['match']}")
            continue
        pred=engine.predict_match(f["home"]+'-'+f["away"],hid,aid)
        hm,am,notes=_context(settings,cfg,target_date,f["home"],f["away"])
        lh,la=pred.lambda_home*hm,pred.lambda_away*am
        matrix=engine.scoreline_matrix(lh,la); pred=engine._matrix_to_prediction(pred.match_id,hid,aid,lh,la,matrix,'dixon_coles+xg+context')
        predictions.append({'match':f["match"],'kickoff':f["kickoff"],'home_prob':pred.home_win_prob,'draw_prob':pred.draw_prob,'away_prob':pred.away_win_prob,'model':'xG+context (no odds)','bookmakers':0,'context':'; '.join(notes) or 'xG/statistical only'})

    return {'records':records,'quotes':quotes,'rows':rows,'predictions':predictions,'engine':engine,'calibrator':calibrator,'errors':errors,'calibration_error':cal_err,'xg_enriched':xg_enriched}
