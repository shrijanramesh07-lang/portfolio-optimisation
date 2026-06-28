import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import date, timedelta

# ── Stock universe ────────────────────────────────────────────────────────────

STOCK_UNIVERSE = {
    "AZN.L":   ("AstraZeneca",               "Healthcare"),
    "GSK.L":   ("GSK",                       "Healthcare"),
    "ULVR.L":  ("Unilever",                  "Consumer Staples"),
    "DGE.L":   ("Diageo",                    "Consumer Staples"),
    "BATS.L":  ("British American Tobacco",  "Consumer Staples"),
    "TSCO.L":  ("Tesco",                     "Consumer Staples"),
    "IMB.L":   ("Imperial Brands",           "Consumer Staples"),
    "HSBA.L":  ("HSBC",                      "Financials"),
    "LLOY.L":  ("Lloyds",                    "Financials"),
    "BARC.L":  ("Barclays",                  "Financials"),
    "STAN.L":  ("Standard Chartered",        "Financials"),
    "LGEN.L":  ("Legal & General",           "Financials"),
    "BP.L":    ("BP",                        "Energy"),
    "SHEL.L":  ("Shell",                     "Energy"),
    "RIO.L":   ("Rio Tinto",                 "Materials"),
    "AAL.L":   ("Anglo American",            "Materials"),
    "GLEN.L":  ("Glencore",                  "Materials"),
    "NG.L":    ("National Grid",             "Utilities"),
    "SSE.L":   ("SSE",                       "Utilities"),
    "BAE.L":   ("BAE Systems",               "Industrials"),
    "RR.L":    ("Rolls-Royce",               "Industrials"),
    "BT-A.L":  ("BT Group",                  "Telecom"),
    "VOD.L":   ("Vodafone",                  "Telecom"),
    "SGRO.L":  ("Segro",                     "Real Estate"),
    "LAND.L":  ("Land Securities",           "Real Estate"),
}

RISK_FREE_RATE = 0.045   # UK gilt yield ~4.5%
TRADING_DAYS   = 252
SECTOR_CAP     = 0.30
TRAIN_DAYS     = 2 * TRADING_DAYS   # 2-year training window
TEST_DAYS      = TRADING_DAYS // 2  # 6-month test window
YEARS_DATA     = 5

# ── Step 1: Download prices ───────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def download_prices(tickers: tuple) -> pd.DataFrame:
    end   = date.today()
    start = end - timedelta(days=YEARS_DATA * 365 + 60)
    raw   = yf.download(
        tickers     = list(tickers),
        start       = start.isoformat(),
        end         = end.isoformat(),
        auto_adjust = True,
        progress    = False,
    )
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices.dropna(how="all").dropna(axis=1, how="all")


@st.cache_data(show_spinner=False)
def download_benchmark() -> pd.Series:
    end   = date.today()
    start = end - timedelta(days=YEARS_DATA * 365 + 60)
    raw   = yf.download(
        "^FTSE", start=start.isoformat(), end=end.isoformat(),
        auto_adjust=True, progress=False,
    )
    return raw["Close"].squeeze().dropna()

# ── Step 2: Daily returns ─────────────────────────────────────────────────────

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

# ── Step 3: Statistics ────────────────────────────────────────────────────────

def build_statistics(returns: pd.DataFrame):
    mean_returns = returns.mean() * TRADING_DAYS
    cov_matrix   = returns.cov()  * TRADING_DAYS
    return mean_returns, cov_matrix

# ── Step 4: Optimiser ─────────────────────────────────────────────────────────

def _build_constraints(tickers: list, sector_map: dict) -> list:
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    sector_indices = {}
    for i, t in enumerate(tickers):
        sector_indices.setdefault(sector_map[t], []).append(i)
    for idx_list in sector_indices.values():
        constraints.append({
            "type": "ineq",
            "fun": lambda w, il=idx_list: SECTOR_CAP - np.sum(w[il]),
        })
    return constraints


def _minimise(objective, constraints, n: int, extra=None) -> np.ndarray:
    result = minimize(
        objective,
        x0          = np.ones(n) / n,
        method      = "SLSQP",
        bounds      = [(0.0, 1.0)] * n,
        constraints = constraints + (extra or []),
        options     = {"maxiter": 1000, "ftol": 1e-9},
    )
    return result.x


def run_optimiser(
    mean_returns: pd.Series,
    cov_matrix:   pd.DataFrame,
    tickers:      list,
    sector_map:   dict,
    risk_level:   int,
) -> pd.Series:
    n   = len(tickers)
    mu  = mean_returns.values
    cov = cov_matrix.values
    con = _build_constraints(tickers, sector_map)

    def portfolio_vol(w):
        return float(np.sqrt(w @ cov @ w))

    def neg_sharpe(w):
        vol = float(np.sqrt(w @ cov @ w))
        return -(float(w @ mu) - RISK_FREE_RATE) / vol if vol > 1e-10 else 0.0

    if risk_level == 1:
        raw = _minimise(portfolio_vol, con, n)
    elif risk_level >= 100:
        raw = _minimise(neg_sharpe, con, n)
    else:
        w_min = _minimise(portfolio_vol, con, n)
        w_max = _minimise(neg_sharpe,    con, n)
        r_min, r_max = float(w_min @ mu), float(w_max @ mu)
        if r_max <= r_min:
            raw = w_max
        else:
            t      = (risk_level - 1) / 99.0
            target = r_min + t * (r_max - r_min)
            raw    = _minimise(portfolio_vol, con, n, extra=[
                {"type": "ineq", "fun": lambda w: float(w @ mu) - target}
            ])

    clipped = np.clip(raw, 0.0, None)
    return pd.Series(clipped / clipped.sum(), index=tickers)

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

def rolling_backtest(returns, tickers, sector_map, risk_level, benchmark_prices):
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
            w = run_optimiser(mu, cov, valid, sector_map, risk_level)
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
    total = s.sum()
    if total <= 0:
        return pd.Series(1.0 / len(tickers), index=list(tickers))
    return s / total


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


def rolling_backtest_bl(returns, tickers, sector_map, risk_level, benchmark_prices, market_cap_weights):
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
            w = run_optimiser(mu_bl, cov, valid, sector_map, risk_level)
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


def volatility_regime(benchmark_prices: pd.Series) -> str:
    r = benchmark_prices.pct_change().dropna()
    if len(r) < 60:
        return "Normal"
    recent_vol = r.iloc[-21:].std() * np.sqrt(TRADING_DAYS)
    hist_vol   = r.iloc[-252:].std() * np.sqrt(TRADING_DAYS)
    if hist_vol <= 0:
        return "Normal"
    ratio = recent_vol / hist_vol
    if ratio > 1.35:
        return "High"
    if ratio < 0.75:
        return "Low"
    return "Normal"


def blend_weights(
    mvo_w: pd.Series,
    bl_w: pd.Series,
    mvo_sharpe: float,
    bl_sharpe: float,
    regime: str,
) -> pd.Series:
    """Sharpe-weighted blend of MVO and BL adjusted for current volatility regime."""
    s_mvo = max(mvo_sharpe, 0.0)
    s_bl  = max(bl_sharpe,  0.0)
    alpha = 0.5 if (s_mvo + s_bl) <= 0 else s_mvo / (s_mvo + s_bl)
    if regime == "High":
        alpha -= 0.15   # shift toward BL in turbulent markets
    elif regime == "Low":
        alpha += 0.15   # shift toward MVO when recent history is more predictive
    alpha = float(np.clip(alpha, 0.2, 0.8))
    tickers = list(mvo_w.index)
    v = alpha * mvo_w.values + (1 - alpha) * bl_w.reindex(tickers).fillna(0).values
    v = np.clip(v, 0, None)
    return pd.Series(v / v.sum(), index=tickers)


def generate_method_explanation(
    winner: str,
    mvo_bt: dict,
    bl_bt: dict,
    comb_bt: dict,
    regime: str,
) -> tuple:
    """Returns (para2, para3) for the 'How we built this portfolio' section."""
    all_bt      = {"MVO": mvo_bt, "BL": bl_bt, "Combined": comb_bt}
    sorted_bt   = sorted(all_bt.items(), key=lambda x: x[1].get("Annual Return", 0), reverse=True)
    second_key  = sorted_bt[1][0]
    third_key   = sorted_bt[2][0]
    plain_names = {"MVO": "historically-informed", "BL": "market-expectations", "Combined": "blended"}

    w_ret = all_bt[winner].get("Annual Return", 0)
    w_vol = all_bt[winner].get("Annual Volatility", 0)
    s_ret = all_bt[second_key].get("Annual Return", 0)
    t_ret = all_bt[third_key].get("Annual Return", 0)

    why_map = {
        ("BL",       "High"):   "In the current volatile market environment, starting from market expectations rather than recent history produced more stable allocations that held up better out of sample.",
        ("MVO",      "Low"):    "Markets have been relatively calm recently, which means recent historical returns are a reasonable guide, and the historically-informed approach captured this effectively.",
        ("Combined", "High"):   "The blended approach outperformed either method alone — in volatile conditions neither model is consistently right, so combining them reduces the damage when one is wrong.",
        ("Combined", "Low"):    "The blended approach outperformed either method alone — even in calm markets, combining approaches reduces the risk of over-fitting to recent data.",
        ("Combined", "Normal"): "The blended approach outperformed either method alone — neither model is consistently right, so combining them reduces the damage when one is wrong.",
        ("MVO",      "High"):   "Even in a volatile environment, recent historical patterns proved more predictive than market-implied estimates for this particular selection of stocks.",
        ("MVO",      "Normal"): "Recent historical performance was a reliable predictor for this selection of stocks at this risk level.",
        ("BL",       "Normal"): "Starting from market expectations rather than recent history produced more stable weights that held up better in out-of-sample testing.",
        ("BL",       "Low"):    "Market-implied returns proved more reliable than recent history — the market may be discounting some stocks' recent strong runs as unlikely to persist.",
    }
    why = why_map.get((winner, regime)) or why_map.get((winner, "Normal"), "")

    para2 = (
        f"Tested on historical data the model had never seen, the {plain_names[winner]} approach produced "
        f"the strongest result for your risk level — a {w_ret*100:.1f}% annual return at "
        f"{w_vol*100:.1f}% volatility, compared to {s_ret*100:.1f}% and {t_ret*100:.1f}% return "
        f"for the other approaches. {why}"
    )

    para3_map = {
        ("BL",       "High"):   "Markets have been more volatile than usual recently, which tends to make historical return estimates less reliable. Starting from market expectations produced a more balanced portfolio that is less dependent on recent trends continuing.",
        ("MVO",      "Low"):    "Markets have been relatively calm recently, which means recent historical returns are a reasonable guide. The historically-informed approach captured this momentum effectively.",
        ("Combined", ""):       "Neither approach alone was clearly superior, so the blended method — which automatically adjusts based on market conditions — produced the most reliable result. This is the most common outcome and the most defensible one.",
        ("BL",       ""):       "The market's collective expectations for these stocks proved a better guide than their recent track record alone. This often happens when a stock has had an unusually strong or weak recent period that the broader market does not expect to continue.",
        ("MVO",      ""):       "Historical return patterns for this selection of stocks proved consistent enough that the data-driven approach outperformed market-implied estimates. This is most likely when the selected stocks have stable, predictable return relationships.",
    }
    para3 = (
        para3_map.get((winner, regime))
        or para3_map.get((winner, ""))
        or para3_map["Combined", ""]
    )
    return para2, para3


# ── Plain-English stock descriptions ─────────────────────────────────────────

STOCK_DESCRIPTIONS = {
    "AZN.L":   "Prescription medicines and cancer treatments",
    "GSK.L":   "Vaccines, antibiotics, and specialist medicines",
    "ULVR.L":  "Everyday brands — Dove, Ben & Jerry's, Marmite, Persil",
    "DGE.L":   "Premium drinks and spirits — Guinness, Johnnie Walker, Baileys",
    "BATS.L":  "Cigarettes and next-generation nicotine products (Vuse, Velo)",
    "TSCO.L":  "UK's largest supermarket chain",
    "IMB.L":   "Cigarettes and heated tobacco — Winston, Davidoff, Blu",
    "HSBA.L":  "Global banking and financial services",
    "LLOY.L":  "High street banking, mortgages, and insurance",
    "BARC.L":  "Retail banking, credit cards, and investment banking",
    "STAN.L":  "Banking across Asia, Africa, and the Middle East",
    "LGEN.L":  "Life insurance, pensions, and long-term savings",
    "BP.L":    "Oil, gas, and renewable energy",
    "SHEL.L":  "Global oil, gas, and low-carbon energy",
    "RIO.L":   "Iron ore, copper, and critical minerals for clean energy",
    "AAL.L":   "Mining — diamonds, platinum, coal, and copper",
    "GLEN.L":  "Commodities trading and mining — cobalt, copper, zinc",
    "NG.L":    "Gas and electricity networks in the UK and US",
    "SSE.L":   "Electricity generation, networks, and home energy supply",
    "BAE.L":   "Defence systems, military aircraft, and cyber security",
    "RR.L":    "Jet engines, power systems, and defence technology",
    "BT-A.L":  "Broadband, mobile (EE), and business telecommunications",
    "VOD.L":   "Mobile and broadband across Europe and Africa",
    "SGRO.L":  "Warehouses and logistics properties for e-commerce",
    "LAND.L":  "Offices, retail parks, and commercial property in London",
}

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
    weights, mean_returns, cov_matrix, returns, sector_map, names,
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

    sector_totals: dict = {}
    for t in tickers:
        sector_totals[sector_map[t]] = sector_totals.get(sector_map[t], 0.0) + float(weights[t])

    blocks = []

    for t in held:
        w      = float(weights[t])
        name   = names[t]
        rank   = ranks[t]
        sector = sector_map[t]
        peers  = [x for x in held if x != t]
        cap_binding     = sector_totals.get(sector, 0.0) >= SECTOR_CAP - 0.015
        strong_earner   = rank <= max(1, n // 2)

        if peers:
            peer_corrs = {x: float(corr.loc[t, x]) for x in peers}
            best_peer  = min(peer_corrs, key=peer_corrs.get)
            best_c     = peer_corrs[best_peer]
            worst_peer = max(peer_corrs, key=peer_corrs.get)
            worst_c    = peer_corrs[worst_peer]

        if w >= 0.18:
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
                    f"\n\n> Your {sector} picks have hit the 30% safety limit — without it, "
                    "the model would have put even more into this sector."
                )

        elif w >= 0.08:
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

        # Note where the comparison method differs meaningfully
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

    for sector, sw in sector_totals.items():
        if sw >= SECTOR_CAP - 0.015:
            held_in = [names[t] for t in held if sector_map[t] == sector]
            if not held_in:
                continue
            txt = (
                f"**Safety rule: {sector} capped at 30%**\n\n"
                f"Your {sector} picks ({', '.join(held_in)}) have hit the 30% limit. "
                f"Without this rule the model would put more into {sector} because it performed well historically. "
                "But concentrating too heavily in one sector means that if something goes wrong across the whole area "
                "— new regulation, falling prices, a downturn — it hits most of your money at once. "
                "The cap keeps things properly spread."
            )
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


def generate_comparison_paragraph(
    mvo_weights: pd.Series,
    bl_weights: pd.Series,
    names: dict,
    mvo_m: dict,
    bl_m: dict,
) -> str:
    tickers = list(mvo_weights.index)
    diffs   = {t: float(bl_weights.get(t, 0)) - float(mvo_weights[t]) for t in tickers}
    max_t   = max(diffs, key=lambda t: abs(diffs[t]))
    max_diff = diffs[max_t]

    both_high = [t for t in tickers if float(mvo_weights[t]) > 0.12 and float(bl_weights.get(t, 0)) > 0.12]

    parts = []
    if both_high:
        agree_names = ", ".join(names[t] for t in both_high[:3])
        parts.append(
            f"Both methods agree on holding significant positions in {agree_names}, "
            "suggesting these are reliably attractive within your selected universe — "
            "their appeal holds whether you use historical returns or market-implied estimates."
        )

    if abs(max_diff) >= 0.05:
        dir_w = "higher" if max_diff > 0 else "lower"
        parts.append(
            f"The biggest divergence is in {names[max_t]}: the market-informed method gives it "
            f"{abs(max_diff)*100:.0f} percentage points {dir_w} than the standard approach. "
            + (
                "This suggests the market's collective estimate of its future returns is more optimistic "
                "than its recent historical average — the market may be pricing in something the historical data doesn't capture."
                if max_diff > 0 else
                "This suggests the market sees its recent strong returns as less likely to persist — "
                "the model moderates the allocation accordingly."
            )
        )

    ret_diff = (mvo_m["Annual Return"] - bl_m["Annual Return"]) * 100
    vol_diff = (mvo_m["Annual Volatility"] - bl_m["Annual Volatility"]) * 100
    if abs(ret_diff) > 0.3:
        if ret_diff > 0:
            vol_note = (
                "with similar volatility" if abs(vol_diff) < 0.5
                else f"but with {abs(vol_diff):.1f}pp more volatility" if vol_diff > 0
                else f"and {abs(vol_diff):.1f}pp less volatility"
            )
            parts.append(
                f"The standard method targets {abs(ret_diff):.1f} percentage points more return per year "
                f"({vol_note}). The market-informed method trades some return potential for a more balanced allocation."
            )
        else:
            parts.append(
                f"The market-informed method actually targets {abs(ret_diff):.1f} percentage points more "
                "return per year — suggesting the market's equilibrium view is more optimistic than the "
                "historical average for this particular selection."
            )

    parts.append(
        "Where both methods agree on a stock's weight, you can be more confident the allocation reflects "
        "something real rather than historical noise. Where they diverge significantly, treat the weighting "
        "as more uncertain — neither past performance nor market consensus guarantees future returns."
    )
    return " ".join(parts)


def _render_comparison(
    mvo_weights: pd.Series,
    bl_weights: pd.Series,
    combined_w: pd.Series,
    names: dict,
    mvo_cum: pd.Series,
    bl_cum: pd.Series,
    comb_cum: pd.Series,
    bench_cum: pd.Series,
    winner: str,
) -> None:
    _divider()
    st.markdown(_h(2, "What each approach would have recommended"), unsafe_allow_html=True)
    st.markdown(
        _sub("For context, here is how each approach performed on your specific stock selection and risk level — the allocation above uses whichever worked best."),
        unsafe_allow_html=True,
    )

    col_info   = [("MVO", "Standard"), ("BL", "Market-Informed"), ("Combined", "Blended")]
    col_colors = {"MVO": _SLATE, "BL": "#C4A882", "Combined": _SAGE}
    W_BG = "#1E2D45"

    # ── Allocation table (3 methods, winner column highlighted) ───────────────
    tickers = list(mvo_weights.index)
    a_rows  = [
        (names[t], float(mvo_weights[t]), float(bl_weights.get(t, 0)), float(combined_w.get(t, 0)))
        for t in sorted(tickers, key=lambda x: float(mvo_weights[x]), reverse=True)
        if float(mvo_weights[t]) >= 0.005 or float(bl_weights.get(t, 0)) >= 0.005 or float(combined_w.get(t, 0)) >= 0.005
    ]

    ah = f'<th style="background:#1E2130;color:{_GREY};padding:10px 16px;text-align:left;font-weight:500;font-size:0.82rem;">Company</th>'
    for mk, label in col_info:
        is_w   = mk == winner
        bg_c   = W_BG if is_w else "#1E2130"
        col_c  = _TEXT if is_w else _GREY
        marker = "  ★" if is_w else ""
        ah += f'<th style="background:{bg_c};color:{col_c};padding:10px 16px;text-align:right;font-weight:500;font-size:0.82rem;">{label}{marker}</th>'

    ab = ""
    idxmap = {"MVO": 1, "BL": 2, "Combined": 3}
    for i, row in enumerate(a_rows):
        bg = "#161822" if i % 2 == 0 else "#1A1D2B"
        ab += f'<tr style="background:{bg};"><td style="padding:10px 16px;color:{_TEXT};font-weight:500;">{row[0]}</td>'
        for mk, _ in col_info:
            val     = row[idxmap[mk]]
            cell_bg = f"background:{W_BG};" if mk == winner else ""
            ab += f'<td style="padding:10px 16px;{cell_bg}color:{col_colors[mk]};font-weight:600;text-align:right;">{val*100:.1f}%</td>'
        ab += "</tr>"

    st.markdown(
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid {_BORDER};">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{ah}</tr></thead><tbody>{ab}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    # ── Backtest performance table ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_h(3, "Backtest performance (out-of-sample)"), unsafe_allow_html=True)

    def _bm(cum):
        return performance_metrics(cum.pct_change().dropna()) if not cum.empty else {}

    bt = {"MVO": _bm(mvo_cum), "BL": _bm(bl_cum), "Combined": _bm(comb_cum)}
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

    # ── Chart: all three approaches + FTSE ───────────────────────────────────
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
_ROSE    = "#9B6B6B"   # negative / downward
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


def _render_table(rows: list) -> None:
    cols = ["Company", "What they do", "Sector", "Your allocation", "How much to invest"]
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
            f'{row["Company"]}</td>'
            f'<td style="padding:10px 16px;color:{_GREY};font-size:0.87rem;">'
            f'{row["What they do"]}</td>'
            f'<td style="padding:10px 16px;color:{_GREY};white-space:nowrap;">'
            f'{row["Sector"]}</td>'
            f'<td style="padding:10px 16px;color:{_SLATE};font-weight:600;white-space:nowrap;">'
            f'{row["Your allocation"]}</td>'
            f'<td style="padding:10px 16px;color:{_TEXT};white-space:nowrap;">'
            f'{row["How much to invest"]}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid {_BORDER};">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{header}</tr></thead><tbody>{body}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )


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
        "The model is most reliable for understanding how your selected stocks relate to each other — "
        "which stocks balance each other out, which ones double up on the same risks — "
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

        label_to_ticker = {f"{v[0]} ({k})": k for k, v in STOCK_UNIVERSE.items()}
        defaults = [
            "AstraZeneca (AZN.L)", "HSBC (HSBA.L)", "BP (BP.L)",
            "Unilever (ULVR.L)", "GSK (GSK.L)", "Barclays (BARC.L)",
        ]
        selected_labels = st.multiselect(
            "Which stocks do you want to include? (pick at least 3)",
            options=sorted(label_to_ticker),
            default=defaults,
        )
        selected_tickers = [label_to_ticker[l] for l in selected_labels]

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
    st.markdown(_h(1, "Find the best way to split your investment across UK stocks"), unsafe_allow_html=True)
    st.markdown(_sub("Tell us what you want to invest in and how much risk you're comfortable with. We'll do the maths."), unsafe_allow_html=True)
    _divider()

    if not run_clicked:
        st.markdown(
            f'<p style="color:{_GREY};">Choose your stocks and preferences in the sidebar, then click '
            f'<strong style="color:{_TEXT};">Calculate my portfolio</strong>.</p>'
            f'<p style="color:{_GREY};">We\'ll work out exactly how to split your money, show you what it would '
            "have returned historically, and explain every decision in plain English.</p>",
            unsafe_allow_html=True,
        )
        return

    if len(selected_tickers) < 3:
        st.error("Please select at least 3 stocks to build a portfolio.")
        return

    sector_map = {t: STOCK_UNIVERSE[t][1] for t in selected_tickers}
    names      = {t: STOCK_UNIVERSE[t][0] for t in selected_tickers}

    # ── Calculations ──────────────────────────────────────────────────────────
    with st.spinner("Downloading 5 years of price data…"):
        prices    = download_prices(tuple(sorted(selected_tickers)))
        benchmark = download_benchmark()

    available = [t for t in selected_tickers if t in prices.columns]
    if len(available) < 3:
        st.error("Couldn't download data for enough stocks. Try a different selection.")
        return
    prices     = prices[available]
    sector_map = {t: sector_map[t] for t in available}
    names      = {t: names[t] for t in available}

    returns                  = calculate_returns(prices)
    mean_returns, cov_matrix = build_statistics(returns)

    with st.spinner("Running analysis — this takes around 60 seconds…"):
        try:
            mvo_weights = run_optimiser(mean_returns, cov_matrix, available, sector_map, risk_level)
        except Exception as e:
            st.error(f"Optimisation failed: {e}")
            return

        market_caps = get_market_caps(tuple(sorted(available)))
        bl_means    = black_litterman_returns(mean_returns, cov_matrix, market_caps)
        try:
            bl_weights = run_optimiser(bl_means, cov_matrix, available, sector_map, risk_level)
        except Exception:
            bl_weights = mvo_weights

        regime = volatility_regime(benchmark)

        mvo_opt_cum, eq_cum, bench_cum = rolling_backtest(
            returns, available, sector_map, risk_level, benchmark
        )
        bl_opt_cum, _, _ = rolling_backtest_bl(
            returns, available, sector_map, risk_level, benchmark, market_caps
        )

        # Combined backtest: 50/50 blend of MVO and BL daily returns
        if not mvo_opt_cum.empty and not bl_opt_cum.empty:
            _r_mvo   = mvo_opt_cum.pct_change().dropna()
            _r_bl    = bl_opt_cum.pct_change().dropna()
            _common  = _r_mvo.index.intersection(_r_bl.index)
            _r_comb  = 0.5 * _r_mvo.loc[_common] + 0.5 * _r_bl.loc[_common] if len(_common) > 0 else _r_mvo
            comb_opt_cum = (1 + _r_comb).cumprod()
        else:
            comb_opt_cum = mvo_opt_cum if not mvo_opt_cum.empty else bl_opt_cum

        _empty_m = {"Sharpe Ratio": 0.0, "Annual Return": 0.0, "Annual Volatility": 0.0, "Max Drawdown": 0.0, "Sortino Ratio": 0.0}

        def bt_m(cum):
            return performance_metrics(cum.pct_change().dropna()) if not cum.empty else dict(_empty_m)

        mvo_bt_m   = bt_m(mvo_opt_cum)
        bl_bt_m    = bt_m(bl_opt_cum)
        comb_bt_m  = bt_m(comb_opt_cum)

        sharpes   = {"MVO": mvo_bt_m["Sharpe Ratio"], "BL": bl_bt_m["Sharpe Ratio"], "Combined": comb_bt_m["Sharpe Ratio"]}
        best_meth = max(sharpes, key=sharpes.get)
        top2      = sorted(sharpes.values(), reverse=True)
        winner    = "Combined" if top2[0] - top2[1] <= 0.05 else best_meth

        combined_w  = blend_weights(mvo_weights, bl_weights, sharpes["MVO"], sharpes["BL"], regime)
        _w_map      = {"MVO": mvo_weights, "BL": bl_weights, "Combined": combined_w}
        _cum_map    = {"MVO": mvo_opt_cum, "BL": bl_opt_cum, "Combined": comb_opt_cum}
        weights     = _w_map[winner]
        display_cum = _cum_map[winner]

    if display_cum.empty:
        st.warning("Not enough price history to run a backtest — need at least 2.5 years of data.")
        return

    port_ret = returns[available] @ weights
    m        = performance_metrics(port_ret)

    # ── Section 1: Allocation table ───────────────────────────────────────────
    st.markdown(_h(2, "Your recommended portfolio"), unsafe_allow_html=True)
    st.markdown(_sub("This is how to split your money. Stocks are sorted from largest to smallest allocation."), unsafe_allow_html=True)

    rows = []
    for t in available:
        w = float(weights[t])
        if w > 0.0005:
            rows.append({
                "Company":            names[t],
                "What they do":       STOCK_DESCRIPTIONS.get(t, ""),
                "Sector":             sector_map[t],
                "Your allocation":    fmt_pct(w),
                "How much to invest": f"£{w * portfolio_size:,.0f}",
            })
    rows.sort(key=lambda r: float(r["Your allocation"].rstrip("%")), reverse=True)
    _render_table(rows)

    # ── Section 2: Metric cards ───────────────────────────────────────────────
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
                   help="Return above the UK gilt rate (~4.5%) per unit of risk. Higher is better.")
        tc4.metric("Sortino Ratio",     fmt_2dp(m["Sortino Ratio"]),
                   help="Like Sharpe but only counts bad days as risk.")

    _divider()

    # ── Section 2.5: How we built this portfolio ──────────────────────────────
    st.markdown(_h(2, "How we built this portfolio"), unsafe_allow_html=True)

    _regime_desc = {
        "High":   "Markets have been more volatile than normal recently, so the model gives more weight to market-implied expectations over recent historical returns.",
        "Normal": "Market conditions are within normal ranges, so all three approaches are given equal consideration.",
        "Low":    "Markets have been calmer than normal recently, which makes recent historical data a more reliable guide.",
    }[regime]
    st.markdown(
        f'<p style="color:{_DIMTEXT};font-size:0.88rem;margin:0 0 1.2rem 0;">'
        f'<strong style="color:{_GREY};">Current market conditions:</strong> '
        f'<strong style="color:{_TEXT};">{regime} volatility</strong> — {_regime_desc}</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "To find your optimal allocation, we tested three different approaches to splitting your money. "
        "The first looks purely at how each stock has performed historically and finds the combination "
        "that would have worked best. The second ignores recent history and instead starts from what the "
        "market collectively expects each stock to return — based on how large each company is relative "
        "to the others. The third blends both approaches, giving more weight to whichever has been more "
        "reliable recently."
    )

    _para2, _para3 = generate_method_explanation(winner, mvo_bt_m, bl_bt_m, comb_bt_m, regime)
    st.markdown(_para2)
    st.markdown(_para3)

    _divider()

    # ── Section 3: Why these weights ─────────────────────────────────────────
    st.markdown(_h(2, "Why is the money split this way?"), unsafe_allow_html=True)
    st.markdown(
        _sub("The model considered every possible way to divide your money across these stocks and settled on "
             "these weights because they give you the best combination of growth and stability for the risk "
             "level you chose — based on 5 years of historical data."),
        unsafe_allow_html=True,
    )
    for block in generate_allocation_explanations(
        weights, mean_returns, cov_matrix, returns, sector_map, names,
    ):
        st.markdown(block)
        _divider()

    # ── Section 4: Historical backtest ────────────────────────────────────────
    st.markdown(_h(2, "How would this have performed historically?"), unsafe_allow_html=True)

    om, em, bm = bt_m(display_cum), bt_m(eq_cum), bt_m(bench_cum)

    # Intro paragraph
    st.markdown(generate_backtest_intro(om, em, bm, investing_goal))

    # Trust section — above the chart
    n_test_periods = max(1, (len(returns) - TRAIN_DAYS) // TEST_DAYS)
    st.markdown(
        _trust_section(om, em, bm, n_test_periods, portfolio_size, display_cum, eq_cum, bench_cum),
        unsafe_allow_html=True,
    )

    # Chart
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

    # Verdict
    for line in generate_backtest_verdict(om, em, bm):
        st.markdown(line)

    # Numbers in expander
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

    # ── Section 5: What each approach would have recommended ──────────────────
    if not mvo_opt_cum.empty and not bl_opt_cum.empty:
        _render_comparison(
            mvo_weights=mvo_weights,
            bl_weights=bl_weights,
            combined_w=combined_w,
            names=names,
            mvo_cum=mvo_opt_cum,
            bl_cum=bl_opt_cum,
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
