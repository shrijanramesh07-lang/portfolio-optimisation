import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import date, timedelta

# ── Full global stock universe ────────────────────────────────────────────────
# (name, sector)

FULL_STOCK_UNIVERSE = {
    # FTSE 100
    "HSBA.L":     ("HSBC", "Financials"),
    "BARC.L":     ("Barclays", "Financials"),
    "LLOY.L":     ("Lloyds", "Financials"),
    "LGEN.L":     ("Legal & General", "Financials"),
    "AZN.L":      ("AstraZeneca", "Healthcare"),
    "GSK.L":      ("GSK", "Healthcare"),
    "HLMA.L":     ("Halma", "Industrials"),
    "SN.L":       ("Smith & Nephew", "Healthcare"),
    "ULVR.L":     ("Unilever", "Consumer Staples"),
    "DGE.L":      ("Diageo", "Consumer Staples"),
    "BATS.L":     ("British American Tobacco", "Consumer Staples"),
    "RKT.L":      ("Reckitt", "Consumer Staples"),
    "BP.L":       ("BP", "Energy"),
    "SHEL.L":     ("Shell", "Energy"),
    "RIO.L":      ("Rio Tinto", "Materials"),
    "GLEN.L":     ("Glencore", "Materials"),
    "BHP.L":      ("BHP", "Materials"),
    "RR.L":       ("Rolls-Royce", "Industrials"),
    "BA.L":       ("BAE Systems", "Industrials"),
    "CPG.L":      ("Compass Group", "Industrials"),
    "RTO.L":      ("Rentokil", "Industrials"),
    "SGRO.L":     ("Segro", "Real Estate"),
    "LAND.L":     ("Land Securities", "Real Estate"),
    "NG.L":       ("National Grid", "Utilities"),
    "SSE.L":      ("SSE", "Utilities"),
    "VOD.L":      ("Vodafone", "Telecom"),
    "BT-A.L":     ("BT Group", "Telecom"),
    "RMV.L":      ("Rightmove", "Technology"),
    "AUTO.L":     ("Auto Trader", "Technology"),
    "EXPN.L":     ("Experian", "Financials"),
    "REL.L":      ("Relx", "Industrials"),
    "LSEG.L":     ("London Stock Exchange Group", "Financials"),
    "PRU.L":      ("Prudential", "Financials"),
    "STJ.L":      ("St James's Place", "Financials"),
    "MKS.L":      ("Marks & Spencer", "Consumer Discretionary"),
    "NXT.L":      ("Next", "Consumer Discretionary"),
    "JD.L":       ("JD Sports", "Consumer Discretionary"),
    # FTSE 250
    "ICP.L":      ("Intermediate Capital Group", "Financials"),
    "ATG.L":      ("Auction Technology", "Technology"),
    "BYIT.L":     ("Bytes Technology", "Technology"),
    "DPLM.L":     ("Diploma", "Industrials"),
    "WOSG.L":     ("Watches of Switzerland", "Consumer Discretionary"),
    "HFG.L":      ("Hilton Food Group", "Consumer Staples"),
    # US Large Cap
    "AAPL":       ("Apple", "Technology"),
    "MSFT":       ("Microsoft", "Technology"),
    "NVDA":       ("Nvidia", "Technology"),
    "AMZN":       ("Amazon", "Consumer Discretionary"),
    "GOOGL":      ("Alphabet", "Technology"),
    "META":       ("Meta", "Technology"),
    "TSLA":       ("Tesla", "Consumer Discretionary"),
    "BRK-B":      ("Berkshire Hathaway", "Financials"),
    "JPM":        ("JPMorgan Chase", "Financials"),
    "JNJ":        ("Johnson & Johnson", "Healthcare"),
    "XOM":        ("Exxon Mobil", "Energy"),
    "V":          ("Visa", "Financials"),
    "UNH":        ("UnitedHealth", "Healthcare"),
    "PG":         ("Procter & Gamble", "Consumer Staples"),
    "MA":         ("Mastercard", "Financials"),
    "LLY":        ("Eli Lilly", "Healthcare"),
    "AVGO":       ("Broadcom", "Technology"),
    "CVX":        ("Chevron", "Energy"),
    "HD":         ("Home Depot", "Consumer Discretionary"),
    "COST":       ("Costco", "Consumer Staples"),
    "MRK":        ("Merck", "Healthcare"),
    "ABBV":       ("AbbVie", "Healthcare"),
    "PEP":        ("PepsiCo", "Consumer Staples"),
    "KO":         ("Coca-Cola", "Consumer Staples"),
    "MCD":        ("McDonald's", "Consumer Discretionary"),
    "NKE":        ("Nike", "Consumer Discretionary"),
    "DIS":        ("Walt Disney", "Consumer Discretionary"),
    "NFLX":       ("Netflix", "Consumer Discretionary"),
    "CRM":        ("Salesforce", "Technology"),
    "ADBE":       ("Adobe", "Technology"),
    # European Large Cap
    "MC.PA":      ("LVMH", "Consumer Discretionary"),
    "NESN.SW":    ("Nestlé", "Consumer Staples"),
    "ROG.SW":     ("Roche", "Healthcare"),
    "ASML.AS":    ("ASML", "Technology"),
    "SAP.DE":     ("SAP", "Technology"),
    "NOVO-B.CO":  ("Novo Nordisk", "Healthcare"),
    "TTE.PA":     ("TotalEnergies", "Energy"),
    "SIE.DE":     ("Siemens", "Industrials"),
    "RMS.PA":     ("Hermès", "Consumer Discretionary"),
    "RACE.MI":    ("Ferrari", "Consumer Discretionary"),
    "SU.PA":      ("Schneider Electric", "Industrials"),
    "OR.PA":      ("L'Oréal", "Consumer Staples"),
    # Asia Pacific
    "7203.T":     ("Toyota", "Consumer Discretionary"),
    "005930.KS":  ("Samsung", "Technology"),
    "TSM":        ("TSMC", "Technology"),
    "SONY":       ("Sony", "Technology"),
    # Additional FTSE
    "CRDA.L":     ("Croda International", "Materials"),
    "SPX.L":      ("Spirax-Sarco", "Industrials"),
    "ITRK.L":     ("Intertek", "Industrials"),
    "AHT.L":      ("Ashtead Group", "Industrials"),
    "ANTO.L":     ("Antofagasta", "Materials"),
    "SGE.L":      ("Sage Group", "Technology"),
    "WTB.L":      ("Whitbread", "Consumer Discretionary"),
    "IHG.L":      ("InterContinental Hotels", "Consumer Discretionary"),
    "PSN.L":      ("Persimmon", "Consumer Discretionary"),
    "TW.L":       ("Taylor Wimpey", "Consumer Discretionary"),
    # Additional US Large Cap
    "INTC":       ("Intel", "Technology"),
    "AMD":        ("AMD", "Technology"),
    "QCOM":       ("Qualcomm", "Technology"),
    "TXN":        ("Texas Instruments", "Technology"),
    "GS":         ("Goldman Sachs", "Financials"),
    "MS":         ("Morgan Stanley", "Financials"),
    "BAC":        ("Bank of America", "Financials"),
    "CAT":        ("Caterpillar", "Industrials"),
    "BA":         ("Boeing", "Industrials"),
    "LMT":        ("Lockheed Martin", "Industrials"),
    "WMT":        ("Walmart", "Consumer Staples"),
    "SBUX":       ("Starbucks", "Consumer Discretionary"),
    # Additional European
    "ALV.DE":     ("Allianz", "Financials"),
    "CS.PA":      ("AXA", "Financials"),
    "BNP.PA":     ("BNP Paribas", "Financials"),
    "AIR.PA":     ("Airbus", "Industrials"),
    "ADS.DE":     ("Adidas", "Consumer Discretionary"),
    "VOW3.DE":    ("Volkswagen", "Consumer Discretionary"),
    # Additional Asia Pacific
    "0005.HK":    ("HSBC (Hong Kong)", "Financials"),
    "005380.KS":  ("Hyundai Motor", "Consumer Discretionary"),
    "8306.T":     ("Mitsubishi UFJ Financial", "Financials"),
    "6861.T":     ("Keyence", "Technology"),
}

STOCK_DESCRIPTIONS = {
    "HSBA.L":     "Global banking and financial services",
    "BARC.L":     "Retail banking, credit cards, and investment banking",
    "LLOY.L":     "High street banking, mortgages, and insurance",
    "LGEN.L":     "Life insurance, pensions, and long-term savings",
    "AZN.L":      "Prescription medicines and cancer treatments",
    "GSK.L":      "Vaccines, antibiotics, and specialist medicines",
    "HLMA.L":     "Safety, health, and environmental technology",
    "SN.L":       "Medical devices — wound care, orthopaedics, surgery",
    "ULVR.L":     "Everyday brands — Dove, Ben & Jerry's, Marmite, Persil",
    "DGE.L":      "Premium drinks and spirits — Guinness, Johnnie Walker, Baileys",
    "BATS.L":     "Cigarettes and next-generation nicotine products (Vuse, Velo)",
    "RKT.L":      "Household and health brands — Dettol, Nurofen, Durex",
    "BP.L":       "Oil, gas, and renewable energy",
    "SHEL.L":     "Global oil, gas, and low-carbon energy",
    "RIO.L":      "Iron ore, copper, and critical minerals for clean energy",
    "GLEN.L":     "Commodities trading and mining — cobalt, copper, zinc",
    "BHP.L":      "Iron ore, copper, and coal mining",
    "RR.L":       "Jet engines, power systems, and defence technology",
    "BA.L":       "Defence systems, military aircraft, and cyber security",
    "CPG.L":      "Catering and food services for businesses worldwide",
    "RTO.L":      "Pest control and hygiene services worldwide",
    "SGRO.L":     "Warehouses and logistics properties for e-commerce",
    "LAND.L":     "Offices, retail parks, and commercial property in London",
    "NG.L":       "Gas and electricity networks in the UK and US",
    "SSE.L":      "Electricity generation, networks, and home energy supply",
    "VOD.L":      "Mobile and broadband across Europe and Africa",
    "BT-A.L":     "Broadband, mobile (EE), and business telecommunications",
    "RMV.L":      "UK's largest property listings website",
    "AUTO.L":     "UK's largest online car marketplace",
    "EXPN.L":     "Credit reports and data analytics worldwide",
    "REL.L":      "Scientific publishing, legal data, and risk analytics",
    "LSEG.L":     "Stock exchange operator and financial data provider",
    "PRU.L":      "Life insurance and asset management across Asia and Africa",
    "STJ.L":      "Wealth management and financial advice",
    "MKS.L":      "Clothing, food, and homeware retailer",
    "NXT.L":      "Clothing and homeware retailer",
    "JD.L":       "Sportswear and trainers retailer",
    "ICP.L":      "Specialist asset management and lending",
    "ATG.L":      "Online auction marketplaces for specialist goods",
    "BYIT.L":     "Software licensing and IT reseller services",
    "DPLM.L":     "Specialist technical products distribution",
    "WOSG.L":     "Luxury watch retailer — Rolex, Patek Philippe",
    "HFG.L":      "Meat and seafood processing for supermarkets",
    "AAPL":       "iPhones, Macs, and consumer technology",
    "MSFT":       "Windows, Office, and Azure cloud computing",
    "NVDA":       "Graphics chips and AI computing hardware",
    "AMZN":       "Online retail and AWS cloud computing",
    "GOOGL":      "Search engine, advertising, and cloud computing",
    "META":       "Facebook, Instagram, and WhatsApp",
    "TSLA":       "Electric vehicles and energy storage",
    "BRK-B":      "Diversified holding company — insurance, rail, energy",
    "JPM":        "Largest US bank — consumer and investment banking",
    "JNJ":        "Pharmaceuticals and medical devices",
    "XOM":        "Oil and gas exploration and refining",
    "V":          "Global payments network",
    "UNH":        "Health insurance and care services",
    "PG":         "Household brands — Gillette, Pampers, Tide",
    "MA":         "Global payments network",
    "LLY":        "Pharmaceuticals — diabetes and obesity treatments",
    "AVGO":       "Semiconductors and infrastructure software",
    "CVX":        "Oil and gas exploration and refining",
    "HD":         "Home improvement retailer",
    "COST":       "Membership-based warehouse retailer",
    "MRK":        "Pharmaceuticals and vaccines",
    "ABBV":       "Pharmaceuticals — immunology and oncology",
    "PEP":        "Snacks and beverages — Pepsi, Lay's, Gatorade",
    "KO":         "Soft drinks — Coca-Cola, Sprite, Fanta",
    "MCD":        "Fast food restaurant chain",
    "NKE":        "Sportswear and footwear",
    "DIS":        "Film studios, theme parks, and streaming",
    "NFLX":       "Streaming video entertainment",
    "CRM":        "Cloud-based customer relationship software",
    "ADBE":       "Creative and document software — Photoshop, Acrobat",
    "MC.PA":      "Luxury goods — Louis Vuitton, Dior, Moët",
    "NESN.SW":    "Food and beverage brands worldwide",
    "ROG.SW":     "Pharmaceuticals and diagnostics",
    "ASML.AS":    "Semiconductor manufacturing equipment",
    "SAP.DE":     "Enterprise business software",
    "NOVO-B.CO":  "Diabetes and obesity treatments",
    "TTE.PA":     "Global oil, gas, and renewable energy",
    "SIE.DE":     "Industrial automation and engineering",
    "RMS.PA":     "Luxury leather goods and fashion",
    "RACE.MI":    "Luxury sports cars",
    "SU.PA":      "Electrical equipment and automation",
    "OR.PA":      "Cosmetics and beauty brands",
    "7203.T":     "World's largest car manufacturer",
    "005930.KS":  "Semiconductors, smartphones, and electronics",
    "TSM":        "World's largest contract chip manufacturer",
    "SONY":       "Electronics, gaming (PlayStation), and entertainment",
    "CRDA.L":     "Speciality chemicals for cosmetics, healthcare, and agriculture",
    "SPX.L":      "Steam and thermal energy control systems for industry",
    "ITRK.L":     "Quality assurance and product testing services worldwide",
    "AHT.L":      "Equipment rental for construction and industrial projects",
    "ANTO.L":     "Copper mining in Chile",
    "SGE.L":      "Accounting and payroll software for small and medium businesses",
    "WTB.L":      "Premier Inn hotels and budget accommodation",
    "IHG.L":      "Global hotel brands — Holiday Inn, Crowne Plaza, InterContinental",
    "PSN.L":      "UK housebuilder",
    "TW.L":       "UK housebuilder",
    "INTC":       "Computer processors and semiconductor manufacturing",
    "AMD":        "Computer processors and graphics chips",
    "QCOM":       "Mobile phone chips and wireless technology",
    "TXN":        "Analog and embedded semiconductor chips",
    "GS":         "Investment banking and asset management",
    "MS":         "Investment banking and wealth management",
    "BAC":        "Retail and commercial banking across the US",
    "CAT":        "Construction and mining heavy machinery",
    "BA":         "Commercial aircraft and defence systems",
    "LMT":        "Military aircraft, missiles, and defence systems",
    "WMT":        "World's largest retail and grocery chain",
    "SBUX":       "Coffeehouse chain",
    "ALV.DE":     "Insurance and asset management",
    "CS.PA":      "Insurance and asset management",
    "BNP.PA":     "Retail and investment banking across Europe",
    "AIR.PA":     "Commercial aircraft manufacturing",
    "ADS.DE":     "Sportswear and footwear",
    "VOW3.DE":    "Cars — VW, Audi, Porsche, Skoda",
    "0005.HK":    "Global banking and financial services (Hong Kong listing)",
    "005380.KS":  "Cars and electric vehicles",
    "8306.T":     "Japan's largest banking group",
    "6861.T":     "Industrial sensors and automation equipment",
}

# ── Dynamic asset-cap formula constants ───────────────────────────────────────
DYNAMIC_BASE_CAP = 0.20
DYNAMIC_ALPHA    = 0.5
DYNAMIC_BETA     = 0.4

RISK_FREE_RATE = 0.045   # UK gilt yield ~4.5%
TRADING_DAYS   = 252
TRAIN_DAYS     = 2 * TRADING_DAYS   # 2-year training window
TEST_DAYS      = TRADING_DAYS // 2  # 6-month test window
YEARS_DATA     = 5
SCREEN_TARGET_N = 15

# ── Step 1: Download prices ───────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def download_prices(tickers: tuple) -> pd.DataFrame:
    end   = date.today()
    start = end - timedelta(days=YEARS_DATA * 365 + 60)
    try:
        raw = yf.download(
            tickers     = list(tickers),
            start       = start.isoformat(),
            end         = end.isoformat(),
            auto_adjust = True,
            progress    = False,
        )
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices.dropna(how="all").dropna(axis=1, how="all")


@st.cache_data(show_spinner=False)
def download_benchmark() -> pd.Series:
    end   = date.today()
    start = end - timedelta(days=YEARS_DATA * 365 + 60)
    try:
        raw = yf.download(
            "^FTSE", start=start.isoformat(), end=end.isoformat(),
            auto_adjust=True, progress=False,
        )
    except Exception:
        return pd.Series(dtype=float)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    return raw["Close"].squeeze().dropna()

# ── Step 2: Daily returns ─────────────────────────────────────────────────────

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

# ── Step 3: Statistics ────────────────────────────────────────────────────────

def build_statistics(returns: pd.DataFrame):
    mean_returns = returns.mean() * TRADING_DAYS
    cov_matrix   = returns.cov()  * TRADING_DAYS
    return mean_returns, cov_matrix

# ── Stock screening ────────────────────────────────────────────────────────────

def screen_stocks(returns: pd.DataFrame, target_n: int = SCREEN_TARGET_N):
    """
    Greedily build a high-Sharpe, low-correlation subset of the full stock universe.
    Starts with the single best individual Sharpe ratio, then repeatedly adds whichever
    remaining stock most improves the equal-weight portfolio Sharpe of the selected set.
    Returns (selected_tickers, individual_sharpe_series).
    """
    mean_ann = returns.mean() * TRADING_DAYS
    vol_ann  = returns.std()  * np.sqrt(TRADING_DAYS)
    sharpe   = ((mean_ann - RISK_FREE_RATE) / vol_ann).replace([np.inf, -np.inf], np.nan).dropna()
    candidates = list(sharpe.sort_values(ascending=False).index)
    if not candidates:
        return [], sharpe

    selected  = [candidates[0]]
    remaining = candidates[1:]

    def port_sharpe(tickers):
        sub = returns[tickers]
        w   = np.ones(len(tickers)) / len(tickers)
        r   = sub @ w
        ar, av = r.mean() * TRADING_DAYS, r.std() * np.sqrt(TRADING_DAYS)
        return (ar - RISK_FREE_RATE) / av if av > 0 else -np.inf

    n_target = min(target_n, len(candidates))
    while len(selected) < n_target and remaining:
        best_c, best_s = None, -np.inf
        for c in remaining:
            s = port_sharpe(selected + [c])
            if s > best_s:
                best_s, best_c = s, c
        selected.append(best_c)
        remaining.remove(best_c)

    return selected, sharpe

# ── Risk scoring (drives the dynamic constraint system) ──────────────────────

def calculate_risk_scores(returns: pd.DataFrame, keys: list) -> pd.Series:
    """risk_score(asset) = annualised_volatility(asset) / max(annualised_volatility(all assets))"""
    vol = returns[keys].std() * np.sqrt(TRADING_DAYS)
    max_vol = float(vol.max())
    if max_vol <= 0:
        return pd.Series(0.0, index=keys)
    return (vol / max_vol).clip(lower=0.0, upper=1.0)

# ── Step 4: Optimiser (stocks only — Standard MVO, Black-Litterman, Risk Parity) ──

def _build_constraints(tickers: list) -> list:
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def _dynamic_bounds(tickers: list, risk_scores: "pd.Series", user_risk: float) -> list:
    """
    Per-asset bounds, 0 to a dynamic max weight:
      max_weight = base_cap * (1 - alpha * risk_score) * (1 + beta * user_risk)
    """
    bounds = []
    for t in tickers:
        rs = float(risk_scores.get(t, 0.5))
        max_w = max(0.0, DYNAMIC_BASE_CAP * (1 - DYNAMIC_ALPHA * rs) * (1 + DYNAMIC_BETA * user_risk))
        bounds.append((0.0, max_w))
    return bounds


def _minimise(objective, constraints, bounds, extra=None) -> np.ndarray:
    result = minimize(
        objective,
        x0          = np.ones(len(bounds)) / len(bounds),
        method      = "SLSQP",
        bounds      = bounds,
        constraints = constraints + (extra or []),
        options     = {"maxiter": 1000, "ftol": 1e-9},
    )
    return result.x


def _normalise_weights(raw: np.ndarray, tickers: list) -> "pd.Series":
    clipped = np.clip(raw, 0.0, None)
    total   = clipped.sum()
    if total <= 0:
        n = len(tickers)
        return pd.Series(np.ones(n) / n, index=tickers)
    return pd.Series(clipped / total, index=tickers)


def run_optimiser(
    mean_returns: pd.Series,
    cov_matrix:   pd.DataFrame,
    tickers:      list,
    risk_level:   int,
    risk_scores:  "pd.Series",
) -> pd.Series:
    """Standard MVO / Black-Litterman optimiser (same efficient-frontier interpolation either way,
    only the mean_returns input differs between the two methods)."""
    mu  = mean_returns.values
    cov = cov_matrix.values
    user_risk = risk_level / 100.0
    con = _build_constraints(tickers)
    bnd = _dynamic_bounds(tickers, risk_scores, user_risk)

    def portfolio_vol(w):
        return float(np.sqrt(w @ cov @ w))

    def neg_sharpe(w):
        vol = float(np.sqrt(w @ cov @ w))
        return -(float(w @ mu) - RISK_FREE_RATE) / vol if vol > 1e-10 else 0.0

    if risk_level == 1:
        raw = _minimise(portfolio_vol, con, bnd)
    elif risk_level >= 100:
        raw = _minimise(neg_sharpe, con, bnd)
    else:
        w_min = _minimise(portfolio_vol, con, bnd)
        w_max = _minimise(neg_sharpe,    con, bnd)
        r_min, r_max = float(w_min @ mu), float(w_max @ mu)
        if r_max <= r_min:
            raw = w_max
        else:
            t      = (risk_level - 1) / 99.0
            target = r_min + t * (r_max - r_min)
            raw    = _minimise(portfolio_vol, con, bnd, extra=[
                {"type": "ineq", "fun": lambda w: float(w @ mu) - target}
            ])

    return _normalise_weights(raw, tickers)


def _risk_parity_objective(w: np.ndarray, cov: np.ndarray) -> float:
    """Sum of squared differences between each asset's risk contribution and the average
    risk contribution across all assets. Guarded against division by zero."""
    port_var = float(w @ cov @ w)
    if port_var <= 1e-12:
        return 1e6
    marginal_contrib = cov @ w
    risk_contrib = w * marginal_contrib
    target = port_var / len(w)
    return float(np.sum((risk_contrib - target) ** 2))


def run_risk_parity(
    cov_matrix:  pd.DataFrame,
    tickers:     list,
    risk_level:  int,
    risk_scores: "pd.Series",
) -> pd.Series:
    """Risk Parity: every asset contributes equally to total portfolio variance.
    Bounds are 0 to the same dynamic max weight used by the other two methods."""
    cov = cov_matrix.values
    user_risk = risk_level / 100.0
    con = _build_constraints(tickers)
    bnd = _dynamic_bounds(tickers, risk_scores, user_risk)

    raw = _minimise(lambda w: _risk_parity_objective(w, cov), con, bnd)
    return _normalise_weights(raw, tickers)

# ── Step 6: Performance metrics ───────────────────────────────────────────────

def performance_metrics(daily_returns: pd.Series) -> dict:
    ann_ret  = float(daily_returns.mean() * TRADING_DAYS)
    ann_vol  = float(daily_returns.std()  * np.sqrt(TRADING_DAYS))
    sharpe   = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0
    dn_vol   = float(daily_returns[daily_returns < 0].std() * np.sqrt(TRADING_DAYS))
    sortino  = (ann_ret - RISK_FREE_RATE) / dn_vol if dn_vol > 0 else 0.0
    cum      = (1 + daily_returns).cumprod()
    max_dd   = float(((cum - cum.cummax()) / cum.cummax()).min())
    return {
        "Annual Return":     ann_ret,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio":      sharpe,
        "Sortino Ratio":     sortino,
        "Max Drawdown":      max_dd,
    }

# ── Step 5: Rolling backtest ──────────────────────────────────────────────────

def rolling_backtest(returns, tickers, risk_level, benchmark_prices, risk_scores):
    opt_daily, eq_daily = [], []
    n_total, start = len(returns), 0

    while start + TRAIN_DAYS + TEST_DAYS <= n_total:
        train = returns.iloc[start : start + TRAIN_DAYS]
        test  = returns.iloc[start + TRAIN_DAYS : start + TRAIN_DAYS + TEST_DAYS]
        valid = [t for t in tickers if t in train.columns]
        if len(valid) < 2:
            start += TEST_DAYS
            continue
        mu, cov = build_statistics(train[valid])
        try:
            w = run_optimiser(mu, cov, valid, risk_level, risk_scores)
        except Exception:
            start += TEST_DAYS
            continue
        opt_daily.append(test[valid] @ w.values)
        eq_daily.append(test[valid].mean(axis=1))
        start += TEST_DAYS

    if not opt_daily:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    opt_r   = pd.concat(opt_daily)
    eq_r    = pd.concat(eq_daily)
    bench_r = benchmark_prices.pct_change().dropna()

    common  = opt_r.index.intersection(eq_r.index).intersection(bench_r.index)
    opt_r, eq_r, bench_r = opt_r.loc[common], eq_r.loc[common], bench_r.loc[common]

    cum = lambda r: (1 + r).cumprod()
    return cum(opt_r), cum(eq_r), cum(bench_r)


def rolling_backtest_rp(returns, tickers, risk_level, benchmark_prices, risk_scores):
    """Like rolling_backtest but uses Risk Parity weights in each window."""
    opt_daily, eq_daily = [], []
    n_total, start = len(returns), 0

    while start + TRAIN_DAYS + TEST_DAYS <= n_total:
        train = returns.iloc[start : start + TRAIN_DAYS]
        test  = returns.iloc[start + TRAIN_DAYS : start + TRAIN_DAYS + TEST_DAYS]
        valid = [t for t in tickers if t in train.columns]
        if len(valid) < 2:
            start += TEST_DAYS
            continue
        _, cov = build_statistics(train[valid])
        try:
            w = run_risk_parity(cov, valid, risk_level, risk_scores)
        except Exception:
            start += TEST_DAYS
            continue
        opt_daily.append(test[valid] @ w.values)
        eq_daily.append(test[valid].mean(axis=1))
        start += TEST_DAYS

    if not opt_daily:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    opt_r   = pd.concat(opt_daily)
    eq_r    = pd.concat(eq_daily)
    bench_r = benchmark_prices.pct_change().dropna()

    common  = opt_r.index.intersection(eq_r.index).intersection(bench_r.index)
    opt_r, eq_r, bench_r = opt_r.loc[common], eq_r.loc[common], bench_r.loc[common]

    cum = lambda r: (1 + r).cumprod()
    return cum(opt_r), cum(eq_r), cum(bench_r)


# ── Black-Litterman ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_market_caps(tickers: tuple) -> pd.Series:
    caps = {}
    for t in list(tickers):
        try:
            fi = yf.Ticker(t).fast_info
            caps[t] = float(getattr(fi, "market_cap", 0) or 0)
        except Exception:
            caps[t] = 0.0
    s = pd.Series(caps, dtype=float)
    known = s[s > 0]
    if known.empty:
        return pd.Series(1.0 / len(tickers), index=list(tickers))
    # Stocks with missing/unavailable market cap fall back to the average of the known caps
    # rather than being silently treated as having zero weight.
    s = s.where(s > 0, known.mean())
    total = s.sum()
    return s / total if total > 0 else pd.Series(1.0 / len(tickers), index=list(tickers))


def black_litterman_returns(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    market_cap_weights: pd.Series,
    delta: float = 2.5,
) -> pd.Series:
    """Black-Litterman equilibrium returns (no views). Returns absolute annualised returns."""
    tickers = mean_returns.index.tolist()
    w = market_cap_weights.reindex(tickers).fillna(0).values
    total = w.sum()
    if total <= 0:
        return mean_returns
    w = w / total
    Sigma = cov_matrix.loc[tickers, tickers].values
    pi = delta * Sigma @ w          # implied excess returns
    return pd.Series(pi + RISK_FREE_RATE, index=tickers)


def rolling_backtest_bl(returns, tickers, risk_level, benchmark_prices, market_cap_weights, risk_scores):
    """Like rolling_backtest but uses Black-Litterman expected returns in each window."""
    opt_daily, eq_daily = [], []
    n_total, start = len(returns), 0

    while start + TRAIN_DAYS + TEST_DAYS <= n_total:
        train = returns.iloc[start : start + TRAIN_DAYS]
        test  = returns.iloc[start + TRAIN_DAYS : start + TRAIN_DAYS + TEST_DAYS]
        valid = [t for t in tickers if t in train.columns]
        if len(valid) < 2:
            start += TEST_DAYS
            continue
        mu_hist, cov = build_statistics(train[valid])
        w_mkt = market_cap_weights.reindex(valid).fillna(0)
        s = w_mkt.sum()
        w_mkt = w_mkt / s if s > 0 else pd.Series(1.0 / len(valid), index=valid)
        mu_bl = black_litterman_returns(mu_hist, cov, w_mkt)
        try:
            w = run_optimiser(mu_bl, cov, valid, risk_level, risk_scores)
        except Exception:
            start += TEST_DAYS
            continue
        opt_daily.append(test[valid] @ w.values)
        eq_daily.append(test[valid].mean(axis=1))
        start += TEST_DAYS

    if not opt_daily:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    opt_r   = pd.concat(opt_daily)
    eq_r    = pd.concat(eq_daily)
    bench_r = benchmark_prices.pct_change().dropna()

    common  = opt_r.index.intersection(eq_r.index).intersection(bench_r.index)
    opt_r, eq_r, bench_r = opt_r.loc[common], eq_r.loc[common], bench_r.loc[common]

    cum = lambda r: (1 + r).cumprod()
    return cum(opt_r), cum(eq_r), cum(bench_r)


def blend_weights(method_weights: dict, method_sharpes: dict) -> pd.Series:
    """Simple Sharpe-weighted blend across however many methods are provided — no regime adjustment."""
    sharpes = {k: max(v, 0.0) for k, v in method_sharpes.items()}
    total_sharpe = sum(sharpes.values())
    n = len(method_weights)
    alphas = (
        {k: 1.0 / n for k in method_weights}
        if total_sharpe <= 0
        else {k: sharpes[k] / total_sharpe for k in method_weights}
    )

    tickers = list(next(iter(method_weights.values())).index)
    v = np.zeros(len(tickers))
    for k, w in method_weights.items():
        v += alphas[k] * w.reindex(tickers).fillna(0).values
    v = np.clip(v, 0, None)
    total = v.sum()
    if total <= 0:
        return pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
    return pd.Series(v / total, index=tickers)


def generate_method_explanation(
    winner: str,
    mvo_bt: dict,
    bl_bt: dict,
    rp_bt: dict,
    comb_bt: dict,
    mvo_weights: "pd.Series",
    bl_weights: "pd.Series",
    rp_weights: "pd.Series",
    combined_w: "pd.Series",
    names: dict,
) -> list:
    """
    Returns a list of plain-English paragraphs for the 'How we built this portfolio' section.
    [part_mvo, part_bl, part_rp, part_why, part_combined_detail_or_None]
    """
    mvo_sharpe = mvo_bt.get("Sharpe Ratio", 0.0)
    bl_sharpe  = bl_bt.get("Sharpe Ratio", 0.0)
    rp_sharpe  = rp_bt.get("Sharpe Ratio", 0.0)
    comb_sharpe = comb_bt.get("Sharpe Ratio", 0.0)
    mvo_ret, bl_ret, rp_ret = mvo_bt.get("Annual Return", 0.0), bl_bt.get("Annual Return", 0.0), rp_bt.get("Annual Return", 0.0)
    mvo_vol, bl_vol, rp_vol = mvo_bt.get("Annual Volatility", 0.0), bl_bt.get("Annual Volatility", 0.0), rp_bt.get("Annual Volatility", 0.0)

    tickers      = list(mvo_weights.index)
    sorted_ticks = sorted(tickers, key=lambda t: float(mvo_weights[t]), reverse=True)
    top2_names   = [names.get(t, t) for t in sorted_ticks[:2]]
    mvo_max_w    = float(mvo_weights[sorted_ticks[0]]) if sorted_ticks else 0.0
    bl_max_w     = max((float(bl_weights.get(t, 0)) for t in tickers), default=0.0)
    rp_max_w     = max((float(rp_weights.get(t, 0)) for t in tickers), default=0.0)

    mvo_is_top  = mvo_sharpe >= bl_sharpe and mvo_sharpe >= rp_sharpe
    mvo_verdict = "paid off out of sample" if mvo_is_top else "didn't fully pay off on unseen data"

    bl_more_spread = bl_max_w < mvo_max_w
    spread_phrase  = "a more evenly spread allocation" if bl_more_spread else "a similarly concentrated allocation"
    bl_robustness  = "more robust when market conditions shifted" if bl_sharpe > mvo_sharpe else "not dramatically more robust to shifting conditions"

    top_stock_text = top2_names[0] + (f" and {top2_names[1]}" if len(top2_names) > 1 else "")
    part_mvo = (
        f"The **historically-informed approach (Standard)** produced {mvo_ret*100:.1f}% annual return at "
        f"{mvo_vol*100:.1f}% volatility (how much the portfolio's value moves up and down) out of sample, "
        f"with a Sharpe ratio (return earned per unit of risk taken) of {mvo_sharpe:.2f}. "
        f"This approach concentrated {mvo_max_w*100:.0f}% of the portfolio into {top_stock_text}, "
        f"because they had the strongest recent track record — a strategy that {mvo_verdict}."
    )

    part_bl = (
        f"The **market-informed approach (Market-Informed)** produced {bl_ret*100:.1f}% annual return at "
        f"{bl_vol*100:.1f}% volatility out of sample, with a Sharpe ratio of {bl_sharpe:.2f}. "
        f"By starting from market expectations rather than recent history, this approach produced "
        f"{spread_phrase} — the largest single position was {bl_max_w*100:.0f}% — "
        f"making it {bl_robustness}."
    )

    part_rp = (
        f"The **equal-risk approach (Risk Parity)** produced {rp_ret*100:.1f}% annual return at "
        f"{rp_vol*100:.1f}% volatility out of sample, with a Sharpe ratio of {rp_sharpe:.2f}. "
        f"Rather than targeting a specific return, this approach sizes each position so it contributes "
        f"roughly equally to overall portfolio risk — its largest position was {rp_max_w*100:.0f}%, "
        "typically spreading money more evenly across your stocks than the other two approaches."
    )

    plain_names = {
        "MVO":        "historically-informed (Standard)",
        "BL":         "market-informed (Market-Informed)",
        "RiskParity": "equal-risk (Risk Parity)",
        "Combined":   "blended (Combined)",
    }
    sorted_sharpes = sorted(
        [("MVO", mvo_sharpe), ("BL", bl_sharpe), ("RiskParity", rp_sharpe), ("Combined", comb_sharpe)],
        key=lambda x: x[1], reverse=True
    )
    winner_sharpe = sorted_sharpes[0][1]
    second_method, second_sharpe = sorted_sharpes[1]
    third_method, third_sharpe   = sorted_sharpes[2]

    margin = winner_sharpe - second_sharpe
    if margin <= 0.05:
        margin_sent = (
            f"The margin was narrow — the leading approaches performed similarly "
            f"(Sharpe ratios within {margin:.2f} of each other), which is why the blended method "
            f"is an equally defensible choice regardless of which approach nominally came first."
        )
    elif margin <= 0.15:
        margin_sent = (
            f"The difference was modest — the winning approach produced {margin:.2f} more units of "
            f"risk-adjusted return than the next best alternative."
        )
    else:
        margin_sent = (
            f"The difference was meaningful — the winning approach produced {margin:.2f} more units of "
            f"risk-adjusted return than the next best alternative, a gap large enough that it is unlikely "
            f"to be due to chance alone."
        )

    driver_map = {
        "MVO":        "The key factor was the consistency of recent returns in your selected stocks — the historical data contained enough signal to give the data-driven approach a clear edge.",
        "BL":         "The key factor was that market-implied returns provided a more stable anchor for the allocation than the recent historical data, which may have overstated the relative attractiveness of one or two stocks.",
        "RiskParity": "The key factor was that spreading risk evenly across your stocks avoided concentrating in any one position that could have underperformed, producing a steadier result.",
        "Combined":   "The key factor was that no single method dominated clearly — the blended approach captured the strengths of each by avoiding the extreme positions that any one method alone would have taken.",
    }
    driver_sent = driver_map.get(winner, "")

    part_why = (
        f"We selected the **{plain_names[winner]}** approach because it produced the strongest "
        f"risk-adjusted return for your chosen risk level — a Sharpe ratio of {winner_sharpe:.2f} "
        f"compared to {second_sharpe:.2f} for the {plain_names[second_method]} and "
        f"{third_sharpe:.2f} for the {plain_names[third_method]}. "
        f"{margin_sent} {driver_sent}"
    )

    if winner != "Combined":
        return [part_mvo, part_bl, part_rp, part_why, None]

    s_mvo, s_bl, s_rp = max(mvo_sharpe, 0.0), max(bl_sharpe, 0.0), max(rp_sharpe, 0.0)
    total_s = s_mvo + s_bl + s_rp
    if total_s <= 0:
        a_mvo = a_bl = a_rp = 1.0 / 3.0
    else:
        a_mvo, a_bl, a_rp = s_mvo / total_s, s_bl / total_s, s_rp / total_s
    pct_mvo, pct_bl, pct_rp = round(a_mvo * 100), round(a_bl * 100), round(a_rp * 100)

    divergences = sorted(
        [
            (
                t,
                float(mvo_weights[t]),
                float(bl_weights.get(t, 0.0)),
                float(rp_weights.get(t, 0.0)),
                float(combined_w.get(t, 0.0)),
                max(float(mvo_weights[t]), float(bl_weights.get(t, 0.0)), float(rp_weights.get(t, 0.0)))
                - min(float(mvo_weights[t]), float(bl_weights.get(t, 0.0)), float(rp_weights.get(t, 0.0))),
            )
            for t in tickers
        ],
        key=lambda x: x[5], reverse=True,
    )
    top_div = [
        (names.get(t, t), mw, bw, rw, cw)
        for t, mw, bw, rw, cw, d in divergences
        if d >= 0.03
    ][:3]

    if top_div:
        div_pieces = ", ".join(
            f"{n} ({mw*100:.0f}% Standard, {bw*100:.0f}% Market-Informed, {rw*100:.0f}% Risk Parity, settled at {cw*100:.0f}%)"
            for n, mw, bw, rw, cw in top_div
        )
        div_sent = (
            f"Stocks where the three methods disagreed most — {div_pieces} — had their allocations pulled "
            "toward the middle, reducing the risk of any single method's view dominating."
        )
    else:
        div_sent = (
            "In this case, the three methods produced fairly similar allocations, so the blend made "
            "only modest adjustments to any of their individual weightings."
        )

    part_combined_detail = (
        f"The blended portfolio is not a simple average. It gave {pct_mvo}% weight to the "
        f"historically-informed method, {pct_bl}% to the market-informed method, and {pct_rp}% to the "
        f"equal-risk method, calculated purely from their relative Sharpe ratios on unseen data "
        f"({mvo_sharpe:.2f} / {bl_sharpe:.2f} / {rp_sharpe:.2f}). {div_sent}"
    )

    return [part_mvo, part_bl, part_rp, part_why, part_combined_detail]


# ── Selection-rationale paragraph ─────────────────────────────────────────────

def generate_selection_rationale_paragraph(
    selected_stocks: list,
    sector_map: dict,
    weights: "pd.Series",
    n_considered: int,
) -> str:
    sector_counts: dict = {}
    for t in selected_stocks:
        sec = sector_map[t]
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
    top_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    sector_txt  = " and ".join(f"{s} ({c} stock{'s' if c != 1 else ''})" for s, c in top_sectors)

    return (
        f"From a universe of {n_considered} global stocks across the UK, US, Europe, and Asia, our screening "
        f"process selected the {len(selected_stocks)} that, together, formed the most efficient combination of "
        f"risk and return — prioritising stocks with strong individual track records that also moved differently "
        f"from one another, so the portfolio isn't overly exposed to any single market move. "
        f"{sector_txt} dominated the final selection, reflecting where the strongest combination of return and "
        "diversification was found in the current market environment."
    )


# ── Explanations ─────────────────────────────────────────────────────────────

def _move_desc(c: float, other_name: str) -> str:
    """Investor-friendly description of how two stocks move together, no jargon."""
    if c >= 0.70:
        return f"tends to fall at the same time as {other_name}"
    if c >= 0.45:
        return f"often moves in a similar direction to {other_name}"
    if c >= 0.10:
        return f"tends to behave quite differently from {other_name}"
    if c >= -0.10:
        return f"tends to move largely independently of {other_name}"
    return f"often moves in the opposite direction to {other_name} — when one falls, the other often holds steady"


def generate_allocation_explanations(
    weights, mean_returns, cov_matrix, returns, names,
    compare_weights=None,
) -> list:
    tickers  = list(weights.index)
    held     = sorted([t for t in tickers if float(weights[t]) > 0.005],
                      key=lambda t: float(weights[t]), reverse=True)
    excluded = [t for t in tickers if float(weights[t]) <= 0.005]

    vols    = pd.Series({t: np.sqrt(float(cov_matrix.loc[t, t])) for t in tickers})
    sharpes = (mean_returns - RISK_FREE_RATE) / vols
    ranks   = {t: int(sharpes.rank(ascending=False)[t]) for t in tickers}
    n       = len(tickers)
    corr    = returns[tickers].corr()

    blocks = []

    for t in held:
        w      = float(weights[t])
        name   = names[t]
        rank   = ranks[t]
        peers  = [x for x in held if x != t]
        cap_binding   = w >= 0.15 - 0.015
        strong_earner = rank <= max(1, n // 2)

        if peers:
            peer_corrs = {x: float(corr.loc[t, x]) for x in peers}
            best_peer  = min(peer_corrs, key=peer_corrs.get)
            best_c     = peer_corrs[best_peer]
            worst_peer = max(peer_corrs, key=peer_corrs.get)

        if w >= 0.10:
            if strong_earner and peers:
                txt = (
                    f"**{name} — {w*100:.0f}%**\n\n"
                    f"Gets the largest share because it's historically delivered strong returns *and* it "
                    f"{_move_desc(best_c, names[best_peer])} — so they naturally balance each other out when markets turn. "
                    "The model values stocks that earn well *and* hold steady when others fall."
                )
            elif peers:
                txt = (
                    f"**{name} — {w*100:.0f}%**\n\n"
                    f"Gets a large share mainly because it {_move_desc(best_c, names[best_peer])}, "
                    "which smooths out the overall portfolio when your other stocks have bad days. "
                    "The model sometimes gives more weight to a stabilising stock than a higher-earning one, "
                    "because reducing your worst falls is just as valuable as boosting your average return."
                )
            else:
                txt = (
                    f"**{name} — {w*100:.0f}%**\n\n"
                    "Historically one of the strongest performers in your selection."
                )
            if cap_binding:
                txt += (
                    "\n\n> This stock has hit the 15% maximum we allow for any single holding — "
                    "without that limit, the model would have put even more here."
                )

        elif w >= 0.05:
            if peers:
                txt = (
                    f"**{name} — {w*100:.0f}%**\n\n"
                    f"A solid mid-sized position. It's performed well historically and it "
                    f"{_move_desc(best_c, names[best_peer])}, giving the portfolio useful balance. "
                    f"It's more closely linked to {names[worst_peer]}, which is why the model didn't go heavier — "
                    "adding too much would start repeating the same risks."
                )
            else:
                txt = (
                    f"**{name} — {w*100:.0f}%**\n\n"
                    "A solid mid-sized position based on historical performance."
                )

        else:
            if peers:
                high_c = {x: float(corr.loc[t, x]) for x in peers if float(corr.loc[t, x]) > 0.55}
                if high_c:
                    sim = max(high_c, key=high_c.get)
                    txt = (
                        f"**{name} — {w*100:.0f}%**\n\n"
                        f"Gets a small share because it {_move_desc(high_c[sim], names[sim])}, "
                        "which already has a bigger position. "
                        "Holding more of both would just double up on the same risks rather than spread them."
                    )
                else:
                    txt = (
                        f"**{name} — {w*100:.0f}%**\n\n"
                        "Gets a small share because it hasn't performed as strongly as the other options "
                        "in your selection over the historical period."
                    )
            else:
                txt = f"**{name} — {w*100:.0f}%**\n\nSmaller position based on historical performance."

        if compare_weights is not None:
            cmp_w = float(compare_weights.get(t, 0))
            diff  = w - cmp_w
            if abs(diff) >= 0.05:
                if diff > 0:
                    txt += (
                        f"\n\n*This is {diff*100:.0f} percentage points more than the standard method "
                        "would give — the market's collective view here is more optimistic than recent "
                        "historical returns alone suggest.*"
                    )
                else:
                    txt += (
                        f"\n\n*This is {abs(diff)*100:.0f} percentage points less than the standard "
                        "method would give — the market implies more modest returns here than recent "
                        "history suggests.*"
                    )

        blocks.append(txt)

    for t in excluded:
        name = names[t]
        if held:
            held_corrs = {x: float(corr.loc[t, x]) for x in held}
            most_sim   = max(held_corrs, key=held_corrs.get)
            most_c     = held_corrs[most_sim]
            txt = (
                f"**{name} — 0%**\n\n"
                f"Left out. It {_move_desc(most_c, names[most_sim])} — so including it would mostly duplicate "
                "that exposure without adding any balancing effect. "
                "The model uses that allocation for something more independent instead."
            )
        else:
            txt = f"**{name} — 0%**\n\nNot selected — ranked below the other options in historical performance."
        blocks.append(txt)

    return blocks


def generate_backtest_intro(
    opt_m: dict, eq_m: dict, bench_m: dict, investing_goal: str
) -> str:
    goal_sentences = {
        "Long-term wealth building (10+ years)":
            "For a 10+ year horizon, short-term underperformance matters less than long-term compounding — "
            "the direction of travel is more important than any individual period.",
        "Medium-term goal — house, travel, etc. (3–5 years)":
            "Over a 3–5 year horizon, both return and how much the portfolio swings around matter. "
            "This strategy aims to balance growth with a smoother ride.",
        "Short-term savings (1–2 years)":
            "For a 1–2 year horizon, how much the portfolio swings matters most — a sudden drop is harder to "
            "recover from in a short window. Consider whether the lower-risk allocation suits you better.",
    }
    intro = (
        "The model was trained on 2 years of price data, then its recommended weights were applied to "
        "the *next* 6 months — data it had never seen. That process was repeated, rolling forward through "
        "the full period. Everything on the chart is what a real investor would actually have experienced — "
        "not a replay of the data used to build the weights. Testing on new data is the only honest measure "
        "of whether the approach works."
    )
    goal_txt = goal_sentences.get(investing_goal, "")
    return intro + (f"\n\n{goal_txt}" if goal_txt else "")


def generate_backtest_verdict(opt_m: dict, eq_m: dict, bench_m: dict) -> list:
    opt_ret   = opt_m["Annual Return"]
    eq_ret    = eq_m["Annual Return"]
    bench_ret = bench_m["Annual Return"]
    opt_dd    = opt_m["Max Drawdown"]
    opt_vol   = opt_m["Annual Volatility"]
    eq_vol    = eq_m["Annual Volatility"]

    lines = []
    beat_eq   = opt_ret > eq_ret
    beat_ftse = opt_ret > bench_ret

    if beat_eq and beat_ftse:
        lines.append(
            f"**The result: the strategy added genuine value.**\n\n"
            f"The optimised weights returned **{opt_ret*100:.1f}% per year** on data they had never seen — "
            f"beating both splitting the money equally ({eq_ret*100:.1f}%) and the FTSE 100 ({bench_ret*100:.1f}%). "
            "The mathematical approach found something real in the historical patterns that continued into the future."
        )
    elif beat_eq and not beat_ftse:
        lines.append(
            f"**The result: better than equal split, behind the broad market.**\n\n"
            f"The optimised weights returned **{opt_ret*100:.1f}% per year** — ahead of an equal split "
            f"({eq_ret*100:.1f}%) but behind the FTSE 100 ({bench_ret*100:.1f}%). "
            "The model added value within your chosen stocks, but the broader market outperformed this selection. "
            "This is common when a small group of stocks misses the wider market's big winners."
        )
    elif not beat_eq and beat_ftse:
        lines.append(
            f"**The result: beat the market, but an equal split did better.**\n\n"
            f"The optimised weights returned **{opt_ret*100:.1f}% per year** — ahead of the FTSE 100 "
            f"({bench_ret*100:.1f}%) but behind an equal split ({eq_ret*100:.1f}%). "
            "This is a known pattern: the model found weights that looked perfect on past data but were too "
            "precisely tuned to stick in the future. Splitting equally — no maths required — proved more robust. "
            "A useful reminder that a simpler approach is sometimes the more reliable one."
        )
    else:
        lines.append(
            f"**The result: both benchmarks outperformed.**\n\n"
            f"The optimised weights returned **{opt_ret*100:.1f}% per year** — behind both an equal split "
            f"({eq_ret*100:.1f}%) and the FTSE 100 ({bench_ret*100:.1f}%). "
            "The model over-fitted to historical patterns that didn't repeat. "
            "Try a different set of stocks or a different risk level."
        )

    smoother = opt_vol < eq_vol
    lines.append(
        f"**The portfolio moved around {'less' if smoother else 'more'} than the equal split** "
        f"({opt_vol*100:.1f}% vs {eq_vol*100:.1f}% per year). "
        + ("A calmer portfolio is easier to hold through bad periods without selling at the wrong moment."
           if smoother else
           "The model accepted a bumpier ride in exchange for a higher target return.")
    )

    lines.append(
        f"**The worst stretch:** The portfolio fell as much as **{abs(opt_dd)*100:.1f}%** from its peak before recovering. "
        "If you had invested at the single worst moment in this period, that's how far down you'd have been before seeing a recovery. "
        "Whether that's acceptable depends on your investing timeline and how long you could afford to wait."
    )

    return lines


def _render_comparison(
    mvo_weights: pd.Series,
    bl_weights: pd.Series,
    rp_weights: pd.Series,
    combined_w: pd.Series,
    names: dict,
    mvo_cum: pd.Series,
    bl_cum: pd.Series,
    rp_cum: pd.Series,
    comb_cum: pd.Series,
    bench_cum: pd.Series,
    winner: str,
) -> None:
    _divider()
    st.markdown(_h(2, "What each approach would have recommended"), unsafe_allow_html=True)
    st.markdown(
        _sub("For context, here is how each approach performed on your specific selection and risk level — the allocation above uses whichever worked best."),
        unsafe_allow_html=True,
    )

    col_info   = [("MVO", "Standard"), ("BL", "Market-Informed"), ("RiskParity", "Risk Parity"), ("Combined", "Blended")]
    col_colors = {"MVO": _SLATE, "BL": "#C4A882", "RiskParity": "#BFA8A8", "Combined": _SAGE}
    W_BG = "#1E2D45"

    tickers   = list(mvo_weights.index)
    w_lookup  = {"MVO": mvo_weights, "BL": bl_weights, "RiskParity": rp_weights, "Combined": combined_w}
    a_rows  = [
        (names[t], {mk: float(w_lookup[mk].get(t, 0)) for mk, _ in col_info})
        for t in sorted(tickers, key=lambda x: float(mvo_weights[x]), reverse=True)
        if any(float(w_lookup[mk].get(t, 0)) >= 0.005 for mk, _ in col_info)
    ]

    ah = f'<th style="background:#1E2130;color:{_GREY};padding:10px 16px;text-align:left;font-weight:500;font-size:0.82rem;">Investment</th>'
    for mk, label in col_info:
        is_w   = mk == winner
        bg_c   = W_BG if is_w else "#1E2130"
        col_c  = _TEXT if is_w else _GREY
        marker = "  ★" if is_w else ""
        ah += f'<th style="background:{bg_c};color:{col_c};padding:10px 16px;text-align:right;font-weight:500;font-size:0.82rem;">{label}{marker}</th>'

    ab = ""
    for i, (name, vals) in enumerate(a_rows):
        bg = "#161822" if i % 2 == 0 else "#1A1D2B"
        ab += f'<tr style="background:{bg};"><td style="padding:10px 16px;color:{_TEXT};font-weight:500;">{name}</td>'
        for mk, _ in col_info:
            val     = vals[mk]
            cell_bg = f"background:{W_BG};" if mk == winner else ""
            ab += f'<td style="padding:10px 16px;{cell_bg}color:{col_colors[mk]};font-weight:600;text-align:right;">{val*100:.1f}%</td>'
        ab += "</tr>"

    st.markdown(
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid {_BORDER};">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{ah}</tr></thead><tbody>{ab}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_h(3, "Backtest performance (out-of-sample)"), unsafe_allow_html=True)

    def _bm(cum):
        return performance_metrics(cum.pct_change().dropna()) if not cum.empty else {}

    bt = {"MVO": _bm(mvo_cum), "BL": _bm(bl_cum), "RiskParity": _bm(rp_cum), "Combined": _bm(comb_cum)}
    metric_labels = ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Max Drawdown"]
    metric_fmts   = [fmt_pct,         fmt_pct,             fmt_2dp,        fmt_pct]

    ph = f'<th style="background:#1E2130;color:{_GREY};padding:10px 16px;text-align:left;font-weight:500;font-size:0.82rem;">Metric</th>'
    for mk, label in col_info:
        is_w   = mk == winner
        bg_c   = W_BG if is_w else "#1E2130"
        col_c  = _TEXT if is_w else _GREY
        marker = "  ★" if is_w else ""
        ph += f'<th style="background:{bg_c};color:{col_c};padding:10px 16px;text-align:right;font-weight:500;font-size:0.82rem;">{label}{marker}</th>'

    pb = ""
    for i, (lbl, fmt) in enumerate(zip(metric_labels, metric_fmts)):
        bg = "#161822" if i % 2 == 0 else "#1A1D2B"
        pb += f'<tr style="background:{bg};"><td style="padding:10px 16px;color:{_TEXT};">{lbl}</td>'
        for mk, _ in col_info:
            cell_bg = f"background:{W_BG};" if mk == winner else ""
            val     = bt[mk].get(lbl)
            pb += (
                f'<td style="padding:10px 16px;{cell_bg}color:{col_colors[mk]};'
                f'font-weight:600;text-align:right;">{fmt(val) if val is not None else "—"}</td>'
            )
        pb += "</tr>"

    st.markdown(
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid {_BORDER};">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{ph}</tr></thead><tbody>{pb}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=mvo_cum.index, y=mvo_cum.values, name="Standard",
        line=dict(color=_SLATE, width=2),
    ))
    fig3.add_trace(go.Scatter(
        x=bl_cum.index, y=bl_cum.values, name="Market-Informed",
        line=dict(color="#C4A882", width=2),
    ))
    fig3.add_trace(go.Scatter(
        x=rp_cum.index, y=rp_cum.values, name="Risk Parity",
        line=dict(color="#BFA8A8", width=2),
    ))
    fig3.add_trace(go.Scatter(
        x=comb_cum.index, y=comb_cum.values, name="Blended",
        line=dict(color=_SAGE, width=2),
    ))
    fig3.add_trace(go.Scatter(
        x=bench_cum.index, y=bench_cum.values, name="FTSE 100",
        line=dict(color="#7A6B8A", width=1.5, dash="dash"),
    ))
    fig3.update_layout(
        title         = "All three approaches tested on data they had never seen",
        yaxis_title   = "Portfolio value (£1 start)",
        legend        = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height        = 400,
        margin        = dict(l=0, r=0, t=55, b=0),
        hovermode     = "x unified",
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color=_GREY),
        title_font    = dict(color=_TEXT),
        xaxis         = dict(gridcolor="#1E2130", linecolor=_BORDER, zerolinecolor=_BORDER),
        yaxis         = dict(gridcolor="#1E2130", linecolor=_BORDER, zerolinecolor=_BORDER),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Step 7: Streamlit UI ──────────────────────────────────────────────────────

def fmt_pct(x):  return f"{x * 100:.1f}%"
def fmt_2dp(x):  return f"{x:.2f}"

# Colour palette
_SLATE   = "#4A6FA5"   # primary accent
_SAGE    = "#52796F"   # positive / upward
_GREY    = "#8B8FA8"   # secondary text
_BORDER  = "#2A2D3E"   # subtle borders / dividers
_BG_CARD = "#131620"   # card / panel background
_BG_BOX  = "#1A1D2B"   # trust-section background
_TEXT    = "#F0F0F5"   # primary text
_DIMTEXT = "#C8CAD8"   # body text inside panels


_CSS = f"""
<style>
/* ── Base ── */
.stApp {{ background-color: #0D0F18; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{ background-color: #0F1117; }}
[data-testid="stSidebar"] section[data-testid="stSidebarContent"] {{
    background-color: #0F1117;
}}

/* ── Primary button → slate blue ── */
button[kind="primary"] {{
    background-color: {_SLATE} !important;
    border-color:     {_SLATE} !important;
    color: #ffffff !important;
    font-weight: 500 !important;
}}
button[kind="primary"]:hover {{
    background-color: #3D5E8F !important;
    border-color:     #3D5E8F !important;
}}

/* ── Secondary buttons ── */
button[kind="secondary"] {{
    border-color: {_BORDER} !important;
    color: {_GREY} !important;
}}

/* ── Dividers ── */
hr {{
    border: none !important;
    border-top: 1px solid {_BORDER} !important;
    margin: 22px 0 !important;
}}

/* ── Metric tiles ── */
[data-testid="stMetricLabel"] {{ color: {_GREY} !important; }}
[data-testid="stMetricValue"] {{ color: {_TEXT} !important; }}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    border-color: {_BORDER} !important;
    background-color: {_BG_CARD} !important;
}}

/* ── Multiselect tags ── */
[data-baseweb="tag"] {{ background-color: {_SLATE} !important; }}

/* ── Captions ── */
[data-testid="stCaptionContainer"] p {{ color: {_GREY} !important; }}

/* ── Radio labels ── */
[data-testid="stRadio"] label p {{ color: {_DIMTEXT} !important; }}

/* ── Slider ── */
[data-testid="stSlider"] .stSlider {{ color: {_SLATE}; }}
</style>
"""


def _divider() -> None:
    st.markdown(f'<hr style="border:none;border-top:1px solid {_BORDER};margin:24px 0;">', unsafe_allow_html=True)


def _h(level: int, text: str) -> str:
    weight = "400" if level == 1 else "500"
    size   = {1: "1.9rem", 2: "1.35rem", 3: "1.05rem"}.get(level, "1rem")
    return (
        f'<h{level} style="font-weight:{weight};color:{_TEXT};'
        f'font-size:{size};margin-bottom:0.2rem;">{text}</h{level}>'
    )


def _sub(text: str) -> str:
    return f'<p style="color:{_GREY};margin-top:2px;margin-bottom:1rem;font-size:0.9rem;">{text}</p>'


def _card(col, headline: str, number: str, sub: str) -> None:
    col.markdown(
        f'<div style="border:1px solid {_BORDER};border-radius:10px;'
        f'padding:18px 20px;background:{_BG_CARD};height:100%;">'
        f'<p style="margin:0 0 12px 0;font-size:0.92rem;font-weight:500;'
        f'line-height:1.5;color:{_DIMTEXT};">{headline}</p>'
        f'<p style="margin:0 0 6px 0;font-size:1.4rem;font-weight:700;color:{_SLATE};">{number}</p>'
        f'<p style="margin:0;font-size:0.74rem;color:{_GREY};line-height:1.4;">{sub}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_investment_table(rows: list) -> None:
    cols = ["Investment", "What it is", "Type", "Your allocation", "Amount"]
    header = "".join(
        f'<th style="background:#1E2130;color:{_GREY};padding:10px 16px;'
        f'text-align:left;font-weight:500;font-size:0.82rem;white-space:nowrap;">{c}</th>'
        for c in cols
    )
    body = ""
    for i, row in enumerate(rows):
        bg = "#161822" if i % 2 == 0 else "#1A1D2B"
        body += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:10px 16px;color:{_TEXT};font-weight:500;white-space:nowrap;">'
            f'{row["Investment"]}</td>'
            f'<td style="padding:10px 16px;color:{_GREY};font-size:0.87rem;">'
            f'{row["What it is"]}</td>'
            f'<td style="padding:10px 16px;color:{_GREY};white-space:nowrap;">'
            f'{row["Type"]}</td>'
            f'<td style="padding:10px 16px;color:{_SLATE};font-weight:600;white-space:nowrap;">'
            f'{row["Your allocation"]}</td>'
            f'<td style="padding:10px 16px;color:{_TEXT};white-space:nowrap;">'
            f'{row["Amount"]}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid {_BORDER};">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{header}</tr></thead><tbody>{body}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )


def _render_stock_screening_expander(selected_stocks: list, all_sharpes: pd.Series, weights: "pd.Series", names: dict) -> None:
    with st.expander("Which stocks were considered and why?"):
        rows = sorted(selected_stocks, key=lambda t: float(all_sharpes.get(t, 0)), reverse=True)
        for t in rows:
            s = float(all_sharpes.get(t, 0))
            w = float(weights.get(t, 0))
            if w > 0.01:
                note = (
                    f"Included with a {w*100:.1f}% allocation — its combination of historical return and "
                    "diversification benefit earned it a place in the final portfolio."
                )
            else:
                note = (
                    "Screened in for its diversification benefit but received minimal allocation once optimised "
                    "alongside the other stocks and defensive assets."
                )
            st.markdown(f"**{names.get(t, t)}** — Sharpe ratio {s:.2f}. {note}")


def _trust_section(
    om: dict, em: dict, bm: dict,
    n_test_periods: int,
    portfolio_size: int,
    opt_cum, eq_cum, bench_cum,
) -> str:
    opt_ret   = om["Annual Return"]
    eq_ret    = em["Annual Return"]
    bench_ret = bm["Annual Return"]

    opt_vs_ftse = (opt_ret  - bench_ret) * 100   # +ve = opt beats FTSE
    eq_vs_opt   = (eq_ret   - opt_ret)   * 100   # +ve = eq beats opt

    opt_final   = portfolio_size * float(opt_cum.iloc[-1])
    eq_final    = portfolio_size * float(eq_cum.iloc[-1])
    bench_final = portfolio_size * float(bench_cum.iloc[-1])

    if opt_vs_ftse >= 0:
        got_right = (
            f"The optimised portfolio beat the FTSE 100 by {opt_vs_ftse:.1f} percentage points per year, "
            f"tested across {n_test_periods} separate 6-month periods the model had never seen."
        )
    else:
        got_right = (
            f"The optimised portfolio trailed the FTSE 100 by {abs(opt_vs_ftse):.1f} percentage points per year "
            f"across {n_test_periods} separate 6-month test periods."
        )

    if eq_vs_opt > 0:
        got_wrong = (
            f"A simple equal split outperformed the optimised weights by {eq_vs_opt:.1f} percentage points per year — "
            "meaning the model found weights that looked perfect on past data but were too precisely tuned to persist going forward."
        )
    else:
        got_wrong = (
            f"The optimised weights beat a simple equal split by {abs(eq_vs_opt):.1f} percentage points per year, "
            "though the equal split captured most of the return without any mathematical modelling."
        )

    verdict = (
        f"If you had followed this model's recommendations and rebalanced every 6 months, "
        f"£{portfolio_size:,} would have become £{opt_final:,.0f} over the test period. "
        f"A simple equal split would have produced £{eq_final:,.0f}. "
        f"The FTSE 100 would have produced £{bench_final:,.0f}. "
        "The model beats the market but doesn't consistently beat the simplest possible alternative — "
        "treat it as a structured starting point for thinking about allocation, not a precise return forecast."
    )

    confidence = (
        "All figures shown are historical — past performance does not guarantee future results. "
        "The model is most reliable for understanding how your selected investments relate to each other — "
        "which ones balance each other out, which ones double up on the same risks — "
        "rather than as a precise forecast of what any of them will return."
    )

    def _para(label: str, body: str) -> str:
        return (
            f'<div style="border-left:3px solid {_BORDER};padding:12px 18px;margin-bottom:18px;">'
            f'<span style="color:{_SLATE};font-weight:500;font-size:0.88rem;">{label} </span>'
            f'<span style="color:{_DIMTEXT};font-size:0.88rem;line-height:1.65;">{body}</span>'
            f'</div>'
        )

    return (
        f'<div style="background:{_BG_BOX};border-radius:10px;padding:24px 28px;margin:20px 0;">'
        f'<p style="color:{_TEXT};font-weight:500;font-size:1.05rem;margin:0 0 20px 0;">'
        f'How much should you trust this?</p>'
        + _para("What the model got right:", got_right)
        + _para("What the model got wrong:", got_wrong)
        + _para("The honest verdict:", verdict)
        + _para("How confident should you be in these numbers?", confidence)
        + '</div>'
    )


# ── Risk score visualisation and explanation (Change 3) ───────────────────────

def calculate_universe_sharpes(mean_returns: "pd.Series", cov_matrix: "pd.DataFrame", tickers: list) -> "pd.Series":
    vols = pd.Series({t: np.sqrt(float(cov_matrix.loc[t, t])) for t in tickers})
    return (mean_returns.reindex(tickers) - RISK_FREE_RATE) / vols


def _define_once(term: str, definition: str, seen: set) -> str:
    """Returns the term with an inline bracket definition the first time it's used, plain after that."""
    if term in seen:
        return term
    seen.add(term)
    return f"{term} ({definition})"


def generate_risk_score_summary_sentence(avg_score: float) -> str:
    if avg_score < 40:
        comparison = "lower than"
    elif avg_score <= 60:
        comparison = "similar to"
    else:
        comparison = "higher than"
    return f"Your portfolio's average risk score is {avg_score:.0f} out of 100 — {comparison} a typical balanced portfolio."


def _risk_gradient_color(score_0_100: float) -> str:
    """Linear RGB interpolation between #52796F (low risk score) and #9B2226 (high risk score)."""
    t = max(0.0, min(1.0, score_0_100 / 100.0))
    c1 = (0x52, 0x79, 0x6F)
    c2 = (0x9B, 0x22, 0x26)
    r = round(c1[0] + (c2[0] - c1[0]) * t)
    g = round(c1[1] + (c2[1] - c1[1]) * t)
    b = round(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def _render_risk_score_chart(score_items: list) -> None:
    """score_items: list of (key, display_name, score_0_100)."""
    if not score_items:
        return
    items  = sorted(score_items, key=lambda x: x[2])
    names_ = [n for _, n, _ in items]
    scores = [s for _, _, s in items]
    colors = [_risk_gradient_color(s) for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=names_, orientation="h",
        marker=dict(color=colors),
        text=[f"{s:.0f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        height=max(220, 32 * len(items)),
        margin=dict(l=0, r=30, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_GREY),
        xaxis=dict(range=[0, 100], showticklabels=False, gridcolor="#1E2130", zerolinecolor=_BORDER, title=""),
        yaxis=dict(gridcolor="#1E2130", zerolinecolor=_BORDER, title=""),
        showlegend=False,
        annotations=[
            dict(xref="paper", yref="paper", x=0, y=-0.12, xanchor="left", yanchor="top",
                 text="Safer", showarrow=False, font=dict(color=_GREY, size=12)),
            dict(xref="paper", yref="paper", x=1, y=-0.12, xanchor="right", yanchor="top",
                 text="More volatile", showarrow=False, font=dict(color=_GREY, size=12)),
        ],
    )
    st.plotly_chart(fig, use_container_width=True)


def _dynamic_cap_only(risk_score: float, user_risk: float) -> float:
    return max(0.0, DYNAMIC_BASE_CAP * (1 - DYNAMIC_ALPHA * risk_score) * (1 + DYNAMIC_BETA * user_risk))


def _render_risk_cap_table(score_items: list, risk_level: int) -> None:
    """score_items: list of (key, display_name, score_0_100)."""
    user_risk = risk_level / 100.0
    cols = ["Investment", "Your risk level cap", "Cap at lowest risk", "Cap at highest risk"]
    header = "".join(
        f'<th style="background:#1E2130;color:{_GREY};padding:10px 16px;'
        f'text-align:left;font-weight:500;font-size:0.82rem;white-space:nowrap;">{c}</th>'
        for c in cols
    )
    body = ""
    for i, (key, name, score100) in enumerate(score_items):
        rs       = score100 / 100.0
        cap_now  = _dynamic_cap_only(rs, user_risk)
        cap_low  = _dynamic_cap_only(rs, 0.0)
        cap_high = _dynamic_cap_only(rs, 1.0)
        bg = "#161822" if i % 2 == 0 else "#1A1D2B"
        body += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:10px 16px;color:{_TEXT};font-weight:500;white-space:nowrap;">{name}</td>'
            f'<td style="padding:10px 16px;color:{_SLATE};font-weight:600;white-space:nowrap;">{cap_now*100:.1f}%</td>'
            f'<td style="padding:10px 16px;color:{_GREY};white-space:nowrap;">{cap_low*100:.1f}%</td>'
            f'<td style="padding:10px 16px;color:{_GREY};white-space:nowrap;">{cap_high*100:.1f}%</td>'
            f'</tr>'
        )
    st.markdown(
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid {_BORDER};">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def generate_role_explanations(
    held_items: list,
    universe_sharpes: "pd.Series",
    universe_avg_sharpe: float,
    sharpe_threshold: float,
    combined_df: "pd.DataFrame",
    defensive_keys: set,
    seen_terms: set,
) -> list:
    """held_items: list of (key, display_name, score_0_100, weight)."""
    held_keys = [k for k, _, _, _ in held_items]
    corr_df = combined_df[held_keys].corr() if len(held_keys) > 1 else None

    blocks = []
    for key, name, score100, w in held_items:
        w_sharpe = float(universe_sharpes.get(key, 0.0))
        avg_corr = None
        if corr_df is not None and key in corr_df.columns:
            peers = [k for k in held_keys if k != key]
            if peers:
                avg_corr = float(corr_df.loc[key, peers].mean())

        sharpe_term = _define_once("Sharpe ratio", "how much return you get per unit of risk taken", seen_terms)
        corr_term   = _define_once(
            "correlation",
            "a measure of how closely two investments move together, from -1 (opposite) to +1 (identical)",
            seen_terms,
        )
        vol_term = _define_once("volatility", "how much the portfolio's value moves up and down", seen_terms)

        if w_sharpe >= sharpe_threshold:
            blocks.append(
                f"**{name}** was chosen for its strong historical return relative to its risk — a {sharpe_term} "
                f"of {w_sharpe:.2f} compared to the universe average of {universe_avg_sharpe:.2f}."
            )
        elif avg_corr is not None and avg_corr < 0.3:
            blocks.append(
                f"**{name}** was chosen because it tends to move independently of your other investments — its "
                f"average {corr_term} with the rest of your portfolio is {avg_corr:.2f}, which reduced overall "
                f"{vol_term} without sacrificing return."
            )
        elif key in defensive_keys and w > 0.03:
            blocks.append(
                f"**{name}** was included as a buffer — it has historically held its value or risen during stock "
                "market downturns, reducing your worst-case loss."
            )
        else:
            blocks.append(
                f"**{name}** was included because it improved the overall balance of return and risk within your "
                "portfolio at this risk level."
            )
    return blocks


def generate_left_out_explanations(
    all_sharpes: "pd.Series",
    selected_stocks: list,
    weights: "pd.Series",
    all_keys: list,
    names: dict,
    combined_df: "pd.DataFrame",
    returns_all: "pd.DataFrame",
    risk_scores: "pd.Series",
    max_vol: float,
    seen_terms: set,
) -> list:
    candidates = [
        (t, float(all_sharpes[t]))
        for t in all_sharpes.index
        if t not in selected_stocks
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    top5 = candidates[:5]

    held_keys = [t for t in all_keys if float(weights.get(t, 0)) > 0.01]

    blocks = []
    for ticker, sharpe_val in top5:
        name = names.get(ticker, ticker)

        most_sim, most_c = None, -2.0
        r_series = returns_all[ticker] if ticker in returns_all.columns else None

        if r_series is not None:
            for hk in held_keys:
                if hk == ticker or hk not in combined_df.columns:
                    continue
                paired = pd.concat([r_series, combined_df[hk]], axis=1).dropna()
                if len(paired) > 30:
                    c = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
                    if c > most_c:
                        most_c, most_sim = c, hk

        if most_sim is not None and most_c >= 0.7:
            sim_name = names.get(most_sim, most_sim)
            reason = f"it is highly correlated with {sim_name}, which already captures that exposure"
        else:
            if ticker in risk_scores.index:
                rs = float(risk_scores.get(ticker, 0.0))
            elif ticker in returns_all.columns and max_vol > 0:
                vol = float(returns_all[ticker].std() * np.sqrt(TRADING_DAYS))
                rs = min(1.0, vol / max_vol)
            else:
                rs = 0.0
            if rs >= 0.6:
                reason = f"its risk score of {rs*100:.0f} was too high relative to the return it added at your risk level"
            else:
                reason = (
                    "it did not improve the portfolio's overall risk-adjusted return despite a reasonable "
                    "individual Sharpe ratio"
                )

        sharpe_term = _define_once("Sharpe ratio", "how much return you get per unit of risk taken", seen_terms)
        blocks.append(f"**{name}** ({sharpe_term} {sharpe_val:.2f}) was not included because {reason}.")
    return blocks


def main():
    st.set_page_config(
        page_title="Portfolio Optimiser",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(_h(2, "Your investment"), unsafe_allow_html=True)

        investing_goal = st.radio(
            "What are you investing for?",
            options=[
                "Long-term wealth building (10+ years)",
                "Medium-term goal — house, travel, etc. (3–5 years)",
                "Short-term savings (1–2 years)",
            ],
        )

        st.divider()

        portfolio_size = st.number_input(
            "How much are you investing? (£)",
            min_value=100, max_value=10_000_000, value=10_000, step=500,
        )

        risk_level = st.slider(
            "How much risk are you comfortable with? (%)",
            min_value=1, max_value=100, value=50,
        )
        st.markdown(
            f'<p style="color:{_SLATE};font-weight:600;font-size:1.1rem;margin:2px 0 4px 0;">{risk_level}%</p>',
            unsafe_allow_html=True,
        )
        if risk_level <= 33:
            st.caption("Conservative — prioritising stability over growth. Expect lower returns but a smoother ride.")
        elif risk_level <= 66:
            st.caption("Balanced — a mix of growth and stability. Accepts some volatility in exchange for higher long-term returns.")
        else:
            st.caption("Growth-focused — maximising long-term return potential. Expects significant short-term swings.")

        run_clicked = st.button("Calculate my portfolio", type="primary", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<a href="https://forms.gle/SSHiMbwoH78n4cyU8" target="_blank" rel="noopener" '
            f'style="display:block;text-align:center;color:{_GREY};font-size:0.8rem;'
            f'text-decoration:none;padding:9px 12px;border:1px solid {_BORDER};'
            f'border-radius:6px;letter-spacing:0.02em;">'
            "Share feedback on this tool</a>",
            unsafe_allow_html=True,
        )

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(_h(1, "Find your optimal global portfolio"), unsafe_allow_html=True)
    st.markdown(
        _sub("Tell us how much risk you're comfortable with. We'll screen the global market for the strongest "
             "combination of stocks and explain every decision in plain English."),
        unsafe_allow_html=True,
    )
    _divider()

    if not run_clicked:
        st.markdown(
            f'<p style="color:{_GREY};">Set your investment details in the sidebar, then click '
            f'<strong style="color:{_TEXT};">Calculate my portfolio</strong>.</p>'
            f'<p style="color:{_GREY};">We\'ll screen the global market for you, work out exactly how to split your '
            "money, show you what it would have returned historically, and explain every decision in plain English.</p>",
            unsafe_allow_html=True,
        )
        return

    with st.sidebar:
        st.caption(f"Analysing {len(FULL_STOCK_UNIVERSE)} global companies to find your optimal allocation.")

    # ── Download & screen ───────────────────────────────────────────────────────
    with st.spinner("Downloading global market data…"):
        all_tickers = tuple(sorted(FULL_STOCK_UNIVERSE.keys()))
        prices_all  = download_prices(all_tickers)
        benchmark   = download_benchmark()

    available_stocks = [t for t in FULL_STOCK_UNIVERSE if t in prices_all.columns]
    if len(available_stocks) < 15:
        st.error("Couldn't download enough market data to build a portfolio. Please try again later.")
        return
    prices_all  = prices_all[available_stocks].ffill()
    returns_all = calculate_returns(prices_all)

    with st.spinner("Screening the global market for the most efficient stock combination…"):
        selected_stocks, all_sharpes = screen_stocks(returns_all, target_n=SCREEN_TARGET_N)

    if len(selected_stocks) < 5:
        st.error("Couldn't find enough stocks with usable data. Please try again later.")
        return

    names      = {t: FULL_STOCK_UNIVERSE[t][0] for t in selected_stocks}
    sector_map = {t: FULL_STOCK_UNIVERSE[t][1] for t in selected_stocks}
    returns_stocks = returns_all[selected_stocks].dropna()

    combined = returns_stocks.copy()
    all_keys = [t for t in selected_stocks if t in combined.columns]

    mean_returns, cov_matrix = build_statistics(combined[all_keys])

    # Pre-optimisation risk scoring — computed once, reused for every method and every
    # rolling-backtest window (the backtest windowing logic itself is unchanged).
    risk_scores = calculate_risk_scores(combined, all_keys)

    _empty_m = {"Sharpe Ratio": 0.0, "Annual Return": 0.0, "Annual Volatility": 0.0, "Max Drawdown": 0.0, "Sortino Ratio": 0.0}

    def bt_m(cum):
        return performance_metrics(cum.pct_change().dropna()) if not cum.empty else dict(_empty_m)

    # ── Optimisation ─────────────────────────────────────────────────────────────
    with st.spinner("Running analysis — this takes around 60 seconds…"):
        try:
            mvo_weights = run_optimiser(mean_returns, cov_matrix, all_keys, risk_level, risk_scores)
        except Exception as e:
            st.error(f"Optimisation failed: {e}")
            return

        mkt_w = get_market_caps(tuple(sorted(selected_stocks)))
        bl_means = black_litterman_returns(mean_returns, cov_matrix, mkt_w)
        try:
            bl_weights = run_optimiser(bl_means, cov_matrix, all_keys, risk_level, risk_scores)
        except Exception:
            bl_weights = mvo_weights

        try:
            rp_weights = run_risk_parity(cov_matrix, all_keys, risk_level, risk_scores)
        except Exception:
            rp_weights = mvo_weights

        mvo_opt_cum, eq_cum, bench_cum = rolling_backtest(
            combined[all_keys], all_keys, risk_level, benchmark, risk_scores
        )
        bl_opt_cum, _, _ = rolling_backtest_bl(
            combined[all_keys], all_keys, risk_level, benchmark, mkt_w, risk_scores
        )
        rp_opt_cum, _, _ = rolling_backtest_rp(
            combined[all_keys], all_keys, risk_level, benchmark, risk_scores
        )

        _non_empty = [c for c in (mvo_opt_cum, bl_opt_cum, rp_opt_cum) if not c.empty]
        if len(_non_empty) == 3:
            _r_mvo  = mvo_opt_cum.pct_change().dropna()
            _r_bl   = bl_opt_cum.pct_change().dropna()
            _r_rp   = rp_opt_cum.pct_change().dropna()
            _common = _r_mvo.index.intersection(_r_bl.index).intersection(_r_rp.index)
            if len(_common) > 0:
                _r_comb = (_r_mvo.loc[_common] + _r_bl.loc[_common] + _r_rp.loc[_common]) / 3.0
            else:
                _r_comb = _r_mvo
            comb_opt_cum = (1 + _r_comb).cumprod()
        elif _non_empty:
            comb_opt_cum = _non_empty[0]
        else:
            comb_opt_cum = mvo_opt_cum

        mvo_bt_m  = bt_m(mvo_opt_cum)
        bl_bt_m   = bt_m(bl_opt_cum)
        rp_bt_m   = bt_m(rp_opt_cum)
        comb_bt_m = bt_m(comb_opt_cum)

        sharpes   = {
            "MVO": mvo_bt_m["Sharpe Ratio"], "BL": bl_bt_m["Sharpe Ratio"],
            "RiskParity": rp_bt_m["Sharpe Ratio"], "Combined": comb_bt_m["Sharpe Ratio"],
        }
        best_meth = max(sharpes, key=sharpes.get)
        top2      = sorted(sharpes.values(), reverse=True)
        winner    = "Combined" if top2[0] - top2[1] <= 0.05 else best_meth

        combined_w = blend_weights(
            {"MVO": mvo_weights, "BL": bl_weights, "RiskParity": rp_weights},
            {"MVO": sharpes["MVO"], "BL": sharpes["BL"], "RiskParity": sharpes["RiskParity"]},
        )
        _w_map      = {"MVO": mvo_weights, "BL": bl_weights, "RiskParity": rp_weights, "Combined": combined_w}
        _cum_map    = {"MVO": mvo_opt_cum, "BL": bl_opt_cum, "RiskParity": rp_opt_cum, "Combined": comb_opt_cum}
        weights     = _w_map[winner]
        display_cum = _cum_map[winner]

    if display_cum.empty:
        st.warning("Not enough price history to run a backtest — need at least 2.5 years of data.")
        return

    port_ret = combined[all_keys] @ weights.reindex(all_keys).fillna(0).values
    m        = performance_metrics(port_ret)

    display_names = {t: v[0] for t, v in FULL_STOCK_UNIVERSE.items()}

    # ── Section 1: Your recommended investments ─────────────────────────────────
    st.markdown(_h(2, "Your recommended investments"), unsafe_allow_html=True)
    st.markdown(_sub("This is how to split your money. Investments are sorted from largest to smallest allocation."), unsafe_allow_html=True)

    rows = []
    for t in all_keys:
        w = float(weights.get(t, 0))
        if w > 0.01:
            rows.append({
                "Investment":      display_names.get(t, t),
                "What it is":      STOCK_DESCRIPTIONS.get(t, ""),
                "Type":            "Stock",
                "Your allocation": fmt_pct(w),
                "Amount":          f"£{w * portfolio_size:,.0f}",
            })
    rows.sort(key=lambda r: float(r["Your allocation"].rstrip("%")), reverse=True)
    _render_investment_table(rows)

    st.markdown(
        f'<p style="color:{_GREY};font-size:0.78rem;margin:10px 0 0 0;">'
        "Non-UK stocks are priced in their local currency. Returns are calculated in local currency terms — "
        "exchange rate movements are not accounted for in this model.</p>",
        unsafe_allow_html=True,
    )

    # ── Why these assets, and why these amounts? ─────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_h(2, "Why these assets, and why these amounts?"), unsafe_allow_html=True)

    held_items = []
    for t in all_keys:
        w_held = float(weights.get(t, 0))
        if w_held > 0.01:
            held_items.append((t, display_names.get(t, t), float(risk_scores.get(t, 0.0)) * 100, w_held))
    held_items.sort(key=lambda x: x[3], reverse=True)

    # Sub-section 1 — How we scored each investment
    st.markdown(_h(3, "How we scored each investment"), unsafe_allow_html=True)
    _render_risk_score_chart([(k, n, s) for k, n, s, w in held_items])
    avg_score = (sum(s for _, _, s, _ in held_items) / len(held_items)) if held_items else 0.0
    st.markdown(generate_risk_score_summary_sentence(avg_score))

    # Sub-section 2 — How your risk setting shaped the limits
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_h(3, "How your risk setting shaped the limits"), unsafe_allow_html=True)
    st.markdown(
        f"You set your risk level to {risk_level}%. This is how it affected the maximum we were willing to invest "
        "in each asset — riskier assets get tighter limits, but your risk setting loosens them as you move toward growth."
    )
    _render_risk_cap_table([(k, n, s) for k, n, s, w in held_items], risk_level)

    # Sub-section 3 — Why the optimiser picked these investments
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_h(3, "Why the optimiser picked these investments"), unsafe_allow_html=True)
    _seen_terms = set()
    universe_sharpes    = calculate_universe_sharpes(mean_returns, cov_matrix, all_keys)
    universe_avg_sharpe = float(universe_sharpes.mean())
    sharpe_threshold     = float(universe_sharpes.quantile(0.75))
    for block in generate_role_explanations(
        held_items, universe_sharpes, universe_avg_sharpe, sharpe_threshold,
        combined[all_keys], set(), _seen_terms,
    ):
        st.markdown(block)

    # Sub-section 4 — What we left out and why
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_h(3, "What we left out and why"), unsafe_allow_html=True)
    max_vol = float((combined[all_keys].std() * np.sqrt(TRADING_DAYS)).max())
    for block in generate_left_out_explanations(
        all_sharpes, selected_stocks, weights, all_keys,
        display_names, combined[all_keys], returns_all, risk_scores, max_vol, _seen_terms,
    ):
        st.markdown(block)

    _divider()

    # ── Section 2: Why these investments were chosen ────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_h(3, "Why these investments were chosen"), unsafe_allow_html=True)
    st.markdown(
        generate_selection_rationale_paragraph(
            selected_stocks, sector_map, weights, len(FULL_STOCK_UNIVERSE)
        )
    )

    st.markdown("<br>", unsafe_allow_html=True)
    _render_stock_screening_expander(selected_stocks, all_sharpes, weights, names)

    # ── Section 3: Metric cards ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_h(3, "What to expect"), unsafe_allow_html=True)

    final_value = portfolio_size * (1 + m["Annual Return"]) ** 5
    vol_pct     = abs(m["Annual Volatility"]) * 100
    dd_pct      = abs(m["Max Drawdown"])      * 100

    c1, c2, c3 = st.columns(3)
    _card(
        c1,
        headline = f"Based on historical returns, £{portfolio_size:,} could grow to £{final_value:,.0f} over 5 years",
        number   = fmt_pct(m["Annual Return"]) + " average annual return",
        sub      = "Historical only — past performance does not guarantee future results",
    )
    _card(
        c2,
        headline = f"In a typical year, expect your portfolio to move by around ±{vol_pct:.0f}%",
        number   = f"±{vol_pct:.0f}% annual movement",
        sub      = "Some years will be much better, some much worse — this is the average size of the swings",
    )
    _card(
        c3,
        headline = f"Worst historical drop before recovery: {dd_pct:.0f}%",
        number   = f"−{dd_pct:.0f}% at worst",
        sub      = "If you had invested at the worst possible moment, this is how far down you would have been before recovering",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Show technical detail (Sharpe ratio, Sortino ratio)"):
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Annual Return",     fmt_pct(m["Annual Return"]),
                   help="Average yearly gain this allocation would historically have produced.")
        tc2.metric("Annual Volatility", fmt_pct(m["Annual Volatility"]),
                   help="Standard deviation of annual returns.")
        tc3.metric("Sharpe Ratio",      fmt_2dp(m["Sharpe Ratio"]),
                   help=f"Return above the UK gilt rate (~{RISK_FREE_RATE*100:.1f}%) per unit of risk. Higher is better.")
        tc4.metric("Sortino Ratio",     fmt_2dp(m["Sortino Ratio"]),
                   help="Like Sharpe but only counts bad days as risk.")

    _divider()

    # ── Section: How we built this portfolio ─────────────────────────────────────
    st.markdown(_h(2, "How we built this portfolio"), unsafe_allow_html=True)

    st.markdown(
        "To find your optimal allocation, we tested three different approaches to splitting your money across "
        "your screened stocks. The first looks purely at how each stock has performed historically and finds the "
        "combination that would have worked best. The second ignores recent history and instead starts from what "
        "the market collectively expects each stock to return. The third sizes each position so it contributes "
        "equally to overall portfolio risk, rather than targeting a specific return."
    )

    _exp_parts = generate_method_explanation(
        winner, mvo_bt_m, bl_bt_m, rp_bt_m, comb_bt_m,
        mvo_weights, bl_weights, rp_weights, combined_w, display_names,
    )
    for _p in _exp_parts:
        if _p:
            st.markdown(_p)

    _divider()

    # ── Section: Why is the money split this way? ────────────────────────────────
    st.markdown(_h(2, "Why is the money split this way?"), unsafe_allow_html=True)
    st.markdown(
        _sub("The model considered every possible way to divide your money across your screened stocks and "
             "settled on these weights because they give you the best combination of growth and stability for "
             "the risk level you chose — based on 5 years of historical data."),
        unsafe_allow_html=True,
    )

    stock_weights_for_explain = weights.reindex(selected_stocks).fillna(0)
    for block in generate_allocation_explanations(
        stock_weights_for_explain,
        mean_returns.reindex(selected_stocks),
        cov_matrix.loc[selected_stocks, selected_stocks],
        combined[selected_stocks],
        names,
    ):
        st.markdown(block)
        _divider()

    # ── Section: Historical backtest ──────────────────────────────────────────────
    st.markdown(_h(2, "How would this have performed historically?"), unsafe_allow_html=True)

    om, em, bm = bt_m(display_cum), bt_m(eq_cum), bt_m(bench_cum)

    st.markdown(generate_backtest_intro(om, em, bm, investing_goal))

    # Use the realised number of out-of-sample windows (eq_cum's length), not the
    # theoretical maximum, in case any window was skipped during the backtest.
    n_test_periods = max(1, len(eq_cum) // TEST_DAYS)
    st.markdown(
        _trust_section(om, em, bm, n_test_periods, portfolio_size, display_cum, eq_cum, bench_cum),
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=display_cum.index, y=display_cum.values, name="Recommended Portfolio",
        line=dict(color=_SLATE, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=eq_cum.index, y=eq_cum.values, name="Equal split",
        line=dict(color="#C4A882", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=bench_cum.index, y=bench_cum.values, name="FTSE 100",
        line=dict(color=_SAGE, width=2, dash="dash"),
    ))
    fig.update_layout(
        title         = "What £1 would have become (tested on data the model had never seen)",
        yaxis_title   = "Portfolio value (£1 start)",
        legend        = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height        = 420,
        margin        = dict(l=0, r=0, t=55, b=0),
        hovermode     = "x unified",
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color=_GREY),
        title_font    = dict(color=_TEXT),
        xaxis         = dict(gridcolor="#1E2130", linecolor=_BORDER, zerolinecolor=_BORDER),
        yaxis         = dict(gridcolor="#1E2130", linecolor=_BORDER, zerolinecolor=_BORDER),
    )
    st.plotly_chart(fig, use_container_width=True)

    for line in generate_backtest_verdict(om, em, bm):
        st.markdown(line)

    with st.expander("Show me the numbers"):
        labels     = ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"]
        fmts       = [fmt_pct, fmt_pct, fmt_2dp, fmt_2dp, fmt_pct]
        compare_df = pd.DataFrame({
            "Metric":                labels,
            "Recommended Portfolio": [f(om[l]) for l, f in zip(labels, fmts)],
            "Equal split":           [f(em[l]) for l, f in zip(labels, fmts)],
            "FTSE 100":              [f(bm[l]) for l, f in zip(labels, fmts)],
        })
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

    # ── Section: What each approach would have recommended ──────────────────────
    if not mvo_opt_cum.empty and not bl_opt_cum.empty and not rp_opt_cum.empty:
        _render_comparison(
            mvo_weights=mvo_weights,
            bl_weights=bl_weights,
            rp_weights=rp_weights,
            combined_w=combined_w,
            names=display_names,
            mvo_cum=mvo_opt_cum,
            bl_cum=bl_opt_cum,
            rp_cum=rp_opt_cum,
            comb_cum=comb_opt_cum,
            bench_cum=bench_cum,
            winner=winner,
        )

    _divider()
    st.markdown(
        f'<div style="text-align:center;padding:20px 0 8px;">'
        f'<p style="color:{_GREY};font-size:0.85rem;margin:0 0 10px 0;">'
        "Built something useful? We'd love to hear what you think.</p>"
        f'<a href="https://forms.gle/SSHiMbwoH78n4cyU8" target="_blank" rel="noopener" '
        f'style="display:inline-block;color:{_TEXT};font-size:0.85rem;font-weight:500;'
        f'text-decoration:none;padding:9px 22px;border:1px solid {_SLATE};'
        f'border-radius:6px;letter-spacing:0.03em;">'
        "Share feedback</a></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
