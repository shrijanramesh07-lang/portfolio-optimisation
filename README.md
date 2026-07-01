# Global Portfolio Optimiser

**Live app: [portfolio-optimisation-nyy5h4ahxqbp25wkbkptuc.streamlit.app](https://portfolio-optimisation-nyy5h4ahxqbp25wkbkptuc.streamlit.app)**

A portfolio construction tool that automatically selects the optimal combination of global equities and tells you exactly how to split your money to maximise risk-adjusted returns at your chosen risk level. Built in Python and deployed publicly via Streamlit.

---

## What it does

You enter how much you want to invest and how much risk you are comfortable with. The tool screens a universe of 80+ global large cap stocks across the UK, US, Europe, and Asia Pacific, selects the most efficient combination, and returns a single recommended portfolio — exactly how to split your money, what to expect in terms of return and risk, and why each stock was chosen.

---

## Methodology

**Stock auto-selection**
Rather than asking the user to pick stocks, the tool screens a universe of 80+ global equities and selects the 12-15 that form the most efficient portfolio at the user's risk level. Selection is based on individual Sharpe ratio and marginal diversification benefit — stocks that improve overall portfolio efficiency by having low correlation with already-selected holdings.

**Three optimisation methods, automatically blended**

The tool runs three distinct optimisation approaches on every calculation:

- **Mean-Variance Optimisation (MVO)** — finds the portfolio that maximises return per unit of risk using historical return and covariance data. Tends to perform well in stable, trending markets but can overfit to recent history.

- **Black-Litterman** — replaces noisy historical return estimates with market-implied equilibrium returns derived from current market capitalisations. Produces more balanced, stable allocations less sensitive to estimation error.

- **Risk Parity** — allocates capital such that every stock contributes equally to total portfolio risk rather than maximising return. Does not rely on expected return estimates, making it robust during volatile or uncertain market conditions.

Each method is backtested on historical data it has never seen. The three methods are then blended via Sharpe-weighted averaging — whichever method performed best out of sample receives proportionally more influence in the final allocation.

**Dynamic risk-based constraints**
Every asset's maximum weight is calculated dynamically from its historical volatility and the user's risk setting rather than arbitrary hardcoded caps:

```
max_weight = base_cap × (1 - α × risk_score) × (1 + β × user_risk)
```

Where risk_score is each stock's annualised volatility relative to the most volatile stock in the universe. High-volatility stocks receive tighter caps at low risk settings and looser caps at high risk settings, scaling continuously with the slider.

**Rolling backtest**
The tool validates its own recommendations using a rolling out-of-sample backtest: optimise on a 2-year training window, apply weights to the following 6 months of unseen data, roll forward, repeat. The backtest compares the optimised portfolio against an equal-weight benchmark and the FTSE 100. The honest finding — that equal weighting sometimes outperforms even the optimised portfolio out of sample — is surfaced transparently and explained to the user, consistent with the well-documented 1/N puzzle in the academic literature.

---

## Stock universe

80+ global large cap equities across four regions:

- **FTSE 100 / 250** — AstraZeneca, Shell, HSBC, Unilever, Diageo, BAE Systems, Rolls-Royce, Rightmove, and others
- **US Large Cap** — Apple, Microsoft, Nvidia, Amazon, Alphabet, Meta, JPMorgan, Goldman Sachs, and others
- **European Large Cap** — ASML, LVMH, Nestlé, Novo Nordisk, Airbus, SAP, Hermès, and others
- **Asia Pacific** — TSMC, Toyota, Samsung, Sony, Keyence, and others

---

## Performance metrics

For each portfolio the tool calculates:

- **Sharpe ratio** — return above the risk-free rate per unit of total volatility
- **Sortino ratio** — return above the risk-free rate per unit of downside volatility only
- **Maximum drawdown** — largest peak-to-trough fall over the backtest period
- **Out-of-sample annual return and volatility** — calculated from the rolling backtest, not in-sample data

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| yfinance | Historical price data |
| pandas / numpy | Data manipulation and matrix operations |
| scipy | Numerical optimisation (SLSQP) |
| scikit-learn | Ledoit-Wolf covariance shrinkage |
| plotly | Interactive charts |
| streamlit | Web app and deployment |

---

## Project structure

```
portfolio-optimiser/
├── app.py                  # Main Streamlit app and page routing
├── src/
│   ├── data.py             # yfinance data fetching and returns calculation
│   ├── optimiser.py        # MVO, Black-Litterman, Risk Parity implementations
│   ├── backtest.py         # Rolling backtest logic
│   ├── metrics.py          # Sharpe, Sortino, drawdown calculations
│   └── screener.py         # Auto stock selection logic
├── requirements.txt
└── README.md
```

---

## Key findings

- The Sharpe-weighted ensemble of three methods consistently outperforms any single method across most stock combinations and time periods
- Equal weighting frequently matches or beats mean-variance optimisation at the individual stock level, consistent with the 1/N puzzle documented by DeMiguel, Garlappi and Uppal (2009)
- Black-Litterman produces more stable out-of-sample allocations than MVO when recent historical returns diverge significantly from long-run market expectations
- Risk Parity produces the lowest maximum drawdown across most tested periods, making it most relevant for conservative risk settings

---

## Limitations

- All return and risk estimates are historical — past performance does not guarantee future results
- Returns for non-UK stocks are calculated in local currency terms — currency movements are not accounted for
- The model assumes liquid markets and ignores transaction costs and bid-ask spreads
- This tool is for educational and research purposes only and does not constitute financial advice. Capital is at risk.
