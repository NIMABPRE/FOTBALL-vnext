"""FOOTBALL vNext — fixture-first Streamlit dashboard."""
from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
import pandas as pd
import streamlit as st
from football_vnext.config import Settings
from football_vnext.application.daily_pipeline import LEAGUES, load_daily_fixtures, run_daily

st.set_page_config(page_title="FOOTBALL vNext",page_icon="⚽",layout="wide")
st.title("⚽ FOOTBALL vNext")
st.caption("Dixon-Coles + xG + calibration + live multi-bookmaker odds + injuries/lineups/news + Risk/Kelly")

with st.sidebar:
    env=Settings.from_env(); st.header("Data sources")
    odds=st.text_input("The Odds API key",value=env.odds_api_key or "",type="password")
    api_football=st.text_input("API-Football key (optional)",value=env.api_football_key or "",type="password",help="Preferred for showing ALL fixtures, including games without bookmaker odds.")
    anthropic=st.text_input("Anthropic key (optional)",value=env.anthropic_api_key or "",type="password")
    st.header("Daily slate")
    league=st.selectbox("League",list(LEAGUES))
    target=st.date_input("Date",datetime.now().date())
    seasons=st.multiselect("Training seasons",["2526","2425","2324"],default=["2526","2425","2324"])
    xi=st.number_input("Time decay xi",0.0,.01,.0018,.0001,format="%.4f")
    min_edge=st.number_input("Min Edge",0.0,.30,.03,.01,format="%.2f")
    min_ev=st.number_input("Min EV",0.0,.30,.03,.01,format="%.2f")
    max_risk=st.number_input("Max Risk",.1,1.0,.60,.05,format="%.2f")
    kelly=st.number_input("Fractional Kelly",.05,1.0,.25,.05,format="%.2f")

settings=Settings(odds_api_key=odds or None,api_football_key=api_football or None,anthropic_api_key=anthropic or None)

# Fixture browser runs independently of the expensive model pipeline.
fixture_state_key=(league,target,bool(api_football),bool(odds))
if "fixture_date" not in st.session_state or st.session_state.fixture_date != fixture_state_key:
    st.session_state.fixture_date=fixture_state_key
    st.session_state.selected_keys=[]
    try:
        with st.spinner("Loading today's fixtures..."):
            st.session_state.fixtures=load_daily_fixtures(settings,league,target)
        st.session_state.fixture_error=None
    except Exception as exc:
        st.session_state.fixtures=[]
        st.session_state.fixture_error=str(exc)

if st.session_state.get("fixture_error"):
    st.error("Fixture loading failed: "+st.session_state.fixture_error)
    st.info("For the complete daily fixture list, configure an API-Football key. Without it, the dashboard falls back to games available from The Odds API.")
    st.stop()

fixtures=st.session_state.get("fixtures",[])
st.subheader(f"📅 {league} — {target} ({len(fixtures)} fixtures)")
if not fixtures:
    st.warning("No fixtures found for this league/date.")
    st.stop()

with st.form("fixture_selection"):
    st.write("**Step 1 — select the matches you want to analyze. No prediction runs while browsing this list.**")
    selected=[]
    for i,f in enumerate(fixtures):
        label=f"{f['match']} — {datetime.fromisoformat(f['kickoff']).strftime('%H:%M')}"
        if st.checkbox(label,key=f"fixture_{i}_{f['key']}"):
            selected.append(f["key"])
    submitted=st.form_submit_button("🚀 Predict selected matches",type="primary",use_container_width=True)

if submitted:
    if not selected:
        st.warning("Select at least one match first.")
        st.stop()
    if not odds:
        st.error("The Odds API key is required for live bookmaker odds and value-bet calculations.")
        st.stop()
    if not seasons:
        st.error("Select at least one training season.")
        st.stop()
    try:
        with st.spinner(f"Running prediction pipeline for {len(selected)} selected match(es)..."):
            st.session_state.result=run_daily(settings,league,target,tuple(seasons),float(xi),float(min_edge),float(min_ev),float(max_risk),float(kelly),selected_match_keys=selected)
        st.session_state.prediction_error=None
    except Exception as exc:
        st.session_state.prediction_error=str(exc)

if st.session_state.get("prediction_error"):
    st.error("Prediction failed: "+st.session_state.prediction_error)

result=st.session_state.get("result")
if not result:
    st.info("Fixtures are loaded. Select one or more matches above to run Dixon-Coles + xG + calibration + odds analysis.")
    st.stop()

m1,m2,m3,m4=st.columns(4)
m1.metric("Historical matches",len(result["records"]))
m2.metric("Selected priced events",len(result["quotes"]))
m3.metric("Value bets",len(result["rows"]))
m4.metric("Calibration","ON" if result["calibrator"] else "OFF")
if result["errors"]: st.warning(" | ".join(result["errors"]))
if result["calibration_error"]: st.warning("Calibration: "+str(result["calibration_error"]))

if result.get("predictions"):
    st.subheader("📊 Predictions for selected matches")
    pdf=pd.DataFrame(result["predictions"]).copy()
    for c in ["home_prob","draw_prob","away_prob"]:
        pdf[c]=pdf[c].map(lambda x:f"{x:.1%}")
    st.dataframe(pdf,use_container_width=True,hide_index=True)

if result["rows"]:
    df=pd.DataFrame(result["rows"]).sort_values(["EV","edge"],ascending=False)
    st.subheader("🟢 Value Bets")
    st.dataframe(df,use_container_width=True,hide_index=True)
else:
    st.subheader("NO BET")
    st.info("No outcome passed Edge + EV + Risk. The system does not force a bet.")

with st.expander("Model diagnostics"):
    e=result["engine"]; c=result["calibrator"]
    st.write(f"Dixon-Coles home advantage: {e.home_advantage:.4f}")
    st.write(f"Dixon-Coles rho: {e.rho:.4f}")
    st.write(f"Historical xG rows enriched: {result.get('xg_enriched', 0)}")
    st.write(f"xG signals available: {len(e.xg_features.signals)} teams")
    st.write(f"Calibration weight: {c.weight:.4f}" if c else "Calibration unavailable")
    st.write("Odds: The Odds API; median bookmaker aggregation + Shin de-vig.")
    st.write("Current injuries/lineups: API-Football when configured.")

st.caption("Gate remains mandatory before real-money use: large real-data walk-forward, CLV, significance, drawdown/Sharpe, and stability across seasons.")
