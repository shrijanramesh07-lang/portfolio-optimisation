# AI Agent Guidance for this Repository

## What this project is
- A small Streamlit app in `app.py` for portfolio optimisation using FTSE-listed equities.
- Uses `yfinance` to fetch historical prices and `scipy.optimize.minimize` for mean-variance optimisation.
- `Test.py` is a helper script for validating price downloads.
- Dependencies are listed in `requirements.txt`.

## Primary files
- `app.py` — main application logic, UI, backtesting, metrics, and explanation generation.
- `requirements.txt` — required Python packages.
- `Test.py` — simple data download/check script.

## Key guidance for AI agents
- Treat `app.py` as the source of truth. It contains both the Streamlit UI and the optimisation/backtest pipeline.
- Preserve the stock universe, sector caps, and the expected Streamlit sidebar flow.
- When changing logic, maintain the current user-facing behaviour: 3+ selected stocks, a risk slider from 1 to 10, and a 5-year historical data window.
- Keep `download_prices` and `download_benchmark` network behavior in mind. The app may fail if `yfinance` returns missing or partial columns.
- Avoid introducing heavy refactors that split the repo into multiple packages unless the user asks for it.

## Run the app
- Install dependencies: `python -m pip install -r requirements.txt`
- Start Streamlit: `streamlit run app.py`
- A quick validation script is `python Test.py`.

## Notes for future customizations
- There are no existing documentation or test suites in the repo.
- If asked, suggest adding a dedicated `.github/copilot-instructions.md` or a test suite for numeric correctness and Streamlit behaviour.
