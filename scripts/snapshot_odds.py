"""Capture current bookmaker prices repeatedly so the system has an auditable pre-kickoff closing snapshot."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from football_vnext.config import Settings
from football_vnext.infrastructure.data_sources.odds_api import TheOddsApiAdapter
from football_vnext.infrastructure.persistence import PredictionStore

SPORTS={"Premier League":"soccer_epl","La Liga":"soccer_spain_la_liga","Serie A":"soccer_italy_serie_a","Bundesliga":"soccer_germany_bundesliga","Ligue 1":"soccer_france_ligue_one"}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--league',default='Premier League'); args=p.parse_args(argv)
    s=Settings.from_env()
    if not s.odds_api_key: raise SystemExit('ODDS_API_KEY is required')
    now=datetime.now(timezone.utc); adapter=TheOddsApiAdapter(s.odds_api_key,sport_key=SPORTS[args.league],regions='eu,uk',markets='h2h')
    store=PredictionStore()
    n=0
    for q in adapter.fetch_odds():
        sid=f"{args.league}:{q.home_team_name}:{q.away_team_name}:{q.commence_time.isoformat()}:{now.isoformat()}"
        odds={x.bookmaker:{'home':x.home,'draw':x.draw,'away':x.away} for x in q.bookmaker_quotes}
        store.save_odds_snapshot(sid,now,q.commence_time,args.league,q.home_team_name,q.away_team_name,odds); n+=1
    print(f'saved {n} odds snapshots')
if __name__=='__main__': main()
