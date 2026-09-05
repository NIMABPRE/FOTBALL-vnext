"""Run the daily pipeline from Task Scheduler/cron.

Environment:
  ODDS_API_KEY (required)
  API_FOOTBALL_KEY (optional, injuries + confirmed lineups)
  ANTHROPIC_API_KEY (optional, automated news analysis)
  DATABASE_URL (optional, SQLite default; PostgreSQL in production)
"""
from __future__ import annotations
import argparse, json
from datetime import datetime
from football_vnext.config import Settings
from football_vnext.application.daily_pipeline import run_daily
from football_vnext.infrastructure.persistence import PredictionStore

def main():
 p=argparse.ArgumentParser(); p.add_argument('--league',default='Premier League'); p.add_argument('--date',default=datetime.now().date().isoformat()); p.add_argument('--seasons',default='2526,2425,2324'); p.add_argument('--timezone',default='Europe/Berlin'); args=p.parse_args()
 settings=Settings.from_env(); result=run_daily(settings,args.league,datetime.fromisoformat(args.date).date(),tuple(x for x in args.seasons.split(',') if x),timezone_name=args.timezone)
 store=PredictionStore()
 for i,row in enumerate(result['predictions']):
  pid=f"{args.league}:{row['match']}:{row['kickoff']}:prediction"
  store.save_prediction(pid,datetime.fromisoformat(row['kickoff']),args.league,row['match'].split(' vs ')[0],row['match'].split(' vs ')[1],row)
 for i,row in enumerate(result['rows']):
  pid=f"{args.league}:{row['match']}:{row['kickoff']}:{row['bet']}"
  store.save_prediction(pid,datetime.fromisoformat(row['kickoff']),args.league,row['match'].split(' vs ')[0],row['match'].split(' vs ')[1],row)
 print(json.dumps({'league':args.league,'date':args.date,'bets':len(result['rows']),'history':len(result['records']),'errors':result['errors'],'calibration_error':result['calibration_error']},indent=2))
if __name__=='__main__': main()
