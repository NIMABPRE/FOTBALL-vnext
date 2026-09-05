# FOOTBALL vNext — Phone/Cloud Deployment

## Recommended setup

Deploy this Streamlit app to Streamlit Community Cloud and open it from the phone browser. The Python engine runs in the cloud; the phone does not need Python, Termux, or a local `.venv`.

Official deployment: https://share.streamlit.io/

## 1. Put the project on GitHub

Create a repository and upload the **contents** of this project folder (not the ZIP file itself).

The repository root should contain:

- `app.py`
- `requirements.txt`
- `src/`
- `data/`

Do not upload `.env` or API keys.

## 2. Deploy

1. Open Streamlit Community Cloud.
2. Sign in with GitHub.
3. Create app → "Yup, I have an app".
4. Select the repository and branch.
5. Entrypoint: `app.py`.
6. Open Advanced settings and select **Python 3.11**.
7. Deploy.

## 3. API key

The app currently accepts the The Odds API key in the Streamlit sidebar, so no secret needs to be committed to GitHub.

Optional keys:

- API-Football: structured injuries/lineups
- Anthropic: optional LLM news assessment

For a more permanent setup, use Streamlit Secrets rather than committing keys.

## 4. Daily use from the phone

Open the deployed `*.streamlit.app` URL in Chrome. Select league/date/seasons and press **Generate predictions**.

The phone is only the UI. The cloud instance runs Python, SciPy, pandas, the Dixon-Coles model, xG enrichment, odds ingestion, de-vig, Edge/EV, Risk and Kelly calculations.

## Important persistence note

The default local SQLite database on Community Cloud is not suitable as permanent storage because cloud app instances can restart. For persistent prediction history/settlements, configure `DATABASE_URL` with PostgreSQL.

## Important betting note

A `NO BET` result is valid behavior. The system must not be forced to produce a wager when Edge + EV + Risk filters do not pass. Real-money execution remains gated by real historical validation/CLV evidence.
