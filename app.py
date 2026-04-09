"""
Interactive Portfolio Analytics Application
FDA2 Project — converted from notebook to Streamlit
"""

import math
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ── Must be FIRST Streamlit call ──────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Analytics", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
TRADING_DAYS    = 252
DEFAULT_TICKERS = "AAPL, MSFT, AMZN, PG, JNJ, JPM"
BENCH           = "^GSPC"


# ═══════════════════════════════════════════════════════════════════════════════
# FINANCIAL HELPER FUNCTIONS  (exact notebook logic)
# ═══════════════════════════════════════════════════════════════════════════════

def sharpe_ratio(r: pd.Series, rf_daily: float) -> float:
    excess = r - rf_daily
    denom  = excess.std() * math.sqrt(TRADING_DAYS)
    return (excess.mean() * TRADING_DAYS) / denom if denom != 0 else np.nan


def sortino_ratio(r: pd.Series, rf_daily: float) -> float:
    excess   = r - rf_daily
    downside = excess[excess < 0]
    ds_std   = downside.std()
    return (excess.mean() * TRADING_DAYS) / (ds_std * math.sqrt(TRADING_DAYS)) if ds_std != 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    wealth      = (1 + r).cumprod()
    running_max = wealth.cummax()
    return (wealth / running_max - 1).min()


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth      = (1 + r).cumprod()
    running_max = wealth.cummax()
    return wealth / running_max - 1


def perf_summary(r: pd.Series, rf_daily: float) -> pd.Series:
    return pd.Series({
        "Annual Return": r.mean() * TRADING_DAYS,
        "Annual Vol":    r.std()  * math.sqrt(TRADING_DAYS),
        "Sharpe":        sharpe_ratio(r,  rf_daily),
        "Sortino":       sortino_ratio(r, rf_daily),
        "Max Drawdown":  max_drawdown(r),
    })


# ── Optimisation ──────────────────────────────────────────────────────────────

def _pret(w, mu):  return w @ mu
def _pvar(w, S):   return w @ S @ w
def _pvol(w, S):   return math.sqrt(_pvar(w, S))


def _gmv(mu, S):
    n      = len(mu)
    w0     = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons   = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    res    = minimize(lambda w: _pvar(w, S), w0,
                      bounds=bounds, constraints=cons, method="SLSQP")
    return res.x if res.success else w0


def _tangency(mu, S, rf_annual):
    n      = len(mu)
    w0     = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons   = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    def neg_sharpe(w):
        vol = _pvol(w, S)
        return -((_pret(w, mu) - rf_annual) / vol) if vol > 0 else np.inf

    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons, method="SLSQP")
    return res.x if res.success else w0


def _efficient_frontier(mu, S, n_points=40):
    n        = len(mu)
    w0       = np.ones(n) / n
    bounds   = [(0, 1)] * n
    cons_sum = {"type": "eq", "fun": lambda w: w.sum() - 1}
    w_gmv_   = _gmv(mu, S)
    gmv_ret  = _pret(w_gmv_, mu)
    targets  = np.linspace(gmv_ret, mu.max(), n_points)
    vols, rets = [], []
    for tr in targets:
        c = [cons_sum, {"type": "eq", "fun": lambda w, t=tr: _pret(w, mu) - t}]
        r = minimize(lambda w: _pvar(w, S), w0, bounds=bounds, constraints=c, method="SLSQP")
        vols.append(_pvol(r.x, S))
        rets.append(_pret(r.x, mu))
    return np.array(vols), np.array(rets)


def _risk_contribution(w, S):
    pv = w @ S @ w
    return (w * (S @ w)) / pv if pv > 0 else np.zeros(len(w))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LAYER  (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers: tuple, start: str, end: str):
    """
    Download adjusted close prices, compute returns.
    Returns (r_stocks, r_bench, warnings_list) or raises on hard error.
    """
    all_tix = list(tickers) + [BENCH]
    raw = yf.download(all_tix, start=start, end=end,
                      auto_adjust=False, progress=False)

    if raw.empty:
        raise ValueError("No data returned. Check your tickers and date range.")

    level0 = raw.columns.get_level_values(0)
    px_df  = raw["Adj Close"] if "Adj Close" in level0 else raw["Close"]

    # Check for tickers that returned nothing
    missing = [t for t in all_tix if t not in px_df.columns or px_df[t].dropna().empty]
    if missing:
        raise ValueError(f"No data found for: {', '.join(missing)}")

    warn_msgs = []

    # Drop stock tickers with >5% missing values
    dropped = []
    for t in list(tickers):
        if px_df[t].isna().mean() > 0.05:
            px_df  = px_df.drop(columns=[t])
            dropped.append(t)
    if dropped:
        warn_msgs.append(f"Dropped (>5% missing data): {', '.join(dropped)}")

    px_df = px_df.dropna()

    remaining = [t for t in tickers if t in px_df.columns]
    if len(remaining) < 3:
        raise ValueError(
            "Fewer than 3 valid stocks remain after cleaning. "
            "Try different tickers or a wider date range."
        )

    rets     = px_df.pct_change().dropna()
    r_stocks = rets[remaining]
    r_bench  = rets[BENCH]
    return r_stocks, r_bench, warn_msgs


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ Configuration")

    ticker_input = st.text_input(
        "Tickers (comma-separated, 3–10)",
        value=DEFAULT_TICKERS,
    )

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start", value=pd.Timestamp("2014-01-01"))
    with c2:
        end_date = st.date_input("End", value=pd.Timestamp("2024-12-31"))

    rf_pct = st.number_input(
        "Risk-Free Rate (annual %)",
        min_value=0.0, max_value=20.0, value=2.0, step=0.1,
    )
    RF_ANNUAL = rf_pct / 100
    RF_DAILY  = RF_ANNUAL / TRADING_DAYS

    run = st.button("🚀 Run Analysis", use_container_width=True)

    st.markdown("---")
    with st.expander("📖 About & Methodology"):
        st.markdown("""
**Data:** Yahoo Finance adjusted close prices

**Returns:** Simple daily returns `r_t = P_t/P_{t-1} − 1`

**Annualisation:** Return × 252; Vol × √252

**Sharpe:** `(mean_excess × 252) / (σ_excess × √252)`

**Sortino:** Same numerator; denominator uses std of *negative* excess returns only

**Portfolio variance:** Full quadratic form `wᵀΣw`

**Efficient Frontier:** Constrained optimisation at each target return (not simulation)

**Risk Contribution:** `PRC_i = w_i·(Σw)_i / σ²_p` — sums to 1

**Benchmark:** S&P 500 (`^GSPC`) — comparison only, not in optimisation
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

st.title("📊 Interactive Portfolio Analytics")

raw_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

errors = []
if len(raw_tickers) < 3:
    errors.append("Enter at least **3** ticker symbols.")
if len(raw_tickers) > 10:
    errors.append("Enter no more than **10** ticker symbols.")
delta_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
if delta_days < 730:
    errors.append("Date range must be at least **2 years**.")
if end_date <= start_date:
    errors.append("End date must be after start date.")

if errors:
    for e in errors:
        st.error(e)
    st.stop()

# ── Show landing screen until first run ──────────────────────────────────────
if not run and "r_stocks" not in st.session_state:
    st.info("Configure your settings in the sidebar and click **Run Analysis** to begin.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

if run:
    with st.spinner("Downloading market data from Yahoo Finance…"):
        try:
            r_stocks, r_bench, warn_msgs = load_data(
                tuple(raw_tickers), str(start_date), str(end_date)
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error downloading data: {e}")
            st.stop()

    for w in warn_msgs:
        st.warning(w)

    st.session_state.update(
        r_stocks=r_stocks, r_bench=r_bench,
        RF_ANNUAL=RF_ANNUAL, RF_DAILY=RF_DAILY,
    )
else:
    r_stocks  = st.session_state["r_stocks"]
    r_bench   = st.session_state["r_bench"]
    RF_ANNUAL = st.session_state["RF_ANNUAL"]
    RF_DAILY  = st.session_state["RF_DAILY"]

STOCKS        = list(r_stocks.columns)
n             = len(STOCKS)
bench_aligned = r_bench.reindex(r_stocks.index).dropna()

# Annual optimisation inputs (full sample)
mu_full = (r_stocks.mean() * TRADING_DAYS).values
S_full  = (r_stocks.cov()  * TRADING_DAYS).values

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Returns & Exploration",
    "⚠️  Risk Analysis",
    "🔗 Correlation & Covariance",
    "💼 Portfolio Optimisation",
    "🔍 Estimation Sensitivity",
])


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 1  –  Returns & Exploration                     ║
# ╚═══════════════════════════════════════════════════════╝
with tab1:
    st.header("Returns & Exploratory Analysis")

    # ── Summary statistics (daily) ───────────────────────
    st.subheader("Summary Statistics (Daily Returns)")
    summary = pd.DataFrame({
        "Mean":     r_stocks.mean(),
        "Std":      r_stocks.std(),
        "Skew":     r_stocks.skew(),
        "Kurtosis": r_stocks.kurt(),
        "Min":      r_stocks.min(),
        "Max":      r_stocks.max(),
    }).round(4)
    bench_row = pd.DataFrame({
        "Mean":     [bench_aligned.mean()],
        "Std":      [bench_aligned.std()],
        "Skew":     [bench_aligned.skew()],
        "Kurtosis": [bench_aligned.kurt()],
        "Min":      [bench_aligned.min()],
        "Max":      [bench_aligned.max()],
    }, index=["S&P 500"]).round(4)
    st.dataframe(pd.concat([summary, bench_row]), use_container_width=True)

    # ── Annualised return & vol ───────────────────────────
    st.subheader("Annualised Return & Volatility")
    annual_df = pd.DataFrame({
        "Annual Return": (r_stocks.mean() * TRADING_DAYS).round(4),
        "Annual Vol":    (r_stocks.std()  * math.sqrt(TRADING_DAYS)).round(4),
    })
    st.dataframe(annual_df, use_container_width=True)

    # ── Cumulative Wealth Index ───────────────────────────
    st.subheader("Cumulative Wealth Index ($10,000 invested)")
    all_series = pd.concat(
        [r_stocks, bench_aligned.rename("S&P 500")], axis=1
    )
    sel = st.multiselect(
        "Select series to display",
        options=list(all_series.columns),
        default=list(all_series.columns),
        key="wealth_sel",
    )
    wealth = 10_000 * (1 + all_series).cumprod()
    fig_w  = go.Figure()
    for col in sel:
        if col in wealth.columns:
            fig_w.add_trace(go.Scatter(x=wealth.index, y=wealth[col],
                                       mode="lines", name=col))
    fig_w.update_layout(
        title="Cumulative Wealth Index ($10,000)",
        xaxis_title="Date", yaxis_title="Value ($)",
        hovermode="x unified", template="plotly_white",
    )
    st.plotly_chart(fig_w, use_container_width=True)

    # ── Return Distribution ───────────────────────────────
    st.subheader("Return Distribution")
    dist_col1, dist_col2 = st.columns([1, 2])
    with dist_col1:
        dist_stock  = st.selectbox("Stock", STOCKS, key="dist_stk")
        dist_view   = st.radio("View", ["Histogram + Normal Fit", "Q-Q Plot"],
                                key="dist_view")
    r_sel = r_stocks[dist_stock].dropna()

    with dist_col2:
        if dist_view == "Histogram + Normal Fit":
            mu_d, sig_d = r_sel.mean(), r_sel.std()
            x_rng       = np.linspace(r_sel.min(), r_sel.max(), 300)
            fig_d = go.Figure()
            fig_d.add_trace(go.Histogram(
                x=r_sel, histnorm="probability density",
                nbinsx=80, opacity=0.65,
                marker_color="mediumpurple", name="Daily Returns",
            ))
            fig_d.add_trace(go.Scatter(
                x=x_rng, y=stats.norm.pdf(x_rng, mu_d, sig_d),
                mode="lines", name="Normal Fit",
                line=dict(color="red", width=2),
            ))
            fig_d.update_layout(
                title=f"{dist_stock} — Daily Return Distribution",
                xaxis_title="Daily Return", yaxis_title="Density",
                template="plotly_white",
            )
            jb_stat, jb_p = stats.jarque_bera(r_sel)
            st.plotly_chart(fig_d, use_container_width=True)
            st.caption(
                f"**Jarque-Bera:** stat={jb_stat:.2f}, p={jb_p:.4f} — "
                + ("Fail to reject normality (p > 0.05)" if jb_p > 0.05
                   else "Reject normality (p ≤ 0.05)")
            )
        else:
            (osm, osr), (slope_qq, intercept_qq, _) = stats.probplot(r_sel)
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=osm, y=osr, mode="markers",
                marker=dict(size=4, color="mediumpurple"), name="Observed",
            ))
            fig_qq.add_trace(go.Scatter(
                x=osm, y=slope_qq * np.array(osm) + intercept_qq,
                mode="lines", name="Normal line",
                line=dict(color="red", width=2),
            ))
            fig_qq.update_layout(
                title=f"{dist_stock} — Q-Q Plot vs Normal",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                template="plotly_white",
            )
            st.plotly_chart(fig_qq, use_container_width=True)
            st.caption(
                "Points departing from the red line in the tails indicate "
                "fat tails (excess kurtosis), common in stock returns."
            )


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 2  –  Risk Analysis                             ║
# ╚═══════════════════════════════════════════════════════╝
with tab2:
    st.header("Risk Analysis")

    # ── Rolling Volatility ────────────────────────────────
    st.subheader("Rolling Annualised Volatility")
    vol_window = st.select_slider(
        "Window (days)", options=[30, 60, 90, 120], value=60, key="vol_win"
    )
    rolling_vol = r_stocks.rolling(vol_window).std() * math.sqrt(TRADING_DAYS)
    fig_rv = go.Figure()
    for col in STOCKS:
        fig_rv.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol[col],
            mode="lines", name=col,
        ))
    fig_rv.update_layout(
        title=f"Rolling {vol_window}-Day Annualised Volatility",
        xaxis_title="Date", yaxis_title="Annualised Volatility",
        hovermode="x unified", template="plotly_white",
    )
    st.plotly_chart(fig_rv, use_container_width=True)

    # ── Drawdown Analysis ─────────────────────────────────
    st.subheader("Drawdown Analysis")
    dd_stock = st.selectbox("Select stock", STOCKS, key="dd_stk")
    dd_s     = drawdown_series(r_stocks[dd_stock])
    mdd_val  = dd_s.min()
    st.metric(f"Maximum Drawdown — {dd_stock}", f"{mdd_val:.2%}")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_s.index, y=dd_s.values,
        fill="tozeroy", mode="lines",
        line=dict(color="crimson"), name="Drawdown",
    ))
    fig_dd.update_layout(
        title=f"{dd_stock} — Drawdown from Running Peak",
        xaxis_title="Date", yaxis_title="Drawdown",
        template="plotly_white",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Risk-Adjusted Metrics ─────────────────────────────
    st.subheader("Sharpe & Sortino Ratios")
    ratios = pd.DataFrame({
        "Sharpe":  r_stocks.apply(lambda r: sharpe_ratio(r,  RF_DAILY)),
        "Sortino": r_stocks.apply(lambda r: sortino_ratio(r, RF_DAILY)),
    }).round(4)
    bench_ratios = pd.DataFrame({
        "Sharpe":  [sharpe_ratio(bench_aligned,  RF_DAILY)],
        "Sortino": [sortino_ratio(bench_aligned, RF_DAILY)],
    }, index=["S&P 500"]).round(4)
    st.dataframe(pd.concat([ratios, bench_ratios]), use_container_width=True)


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 3  –  Correlation & Covariance                  ║
# ╚═══════════════════════════════════════════════════════╝
with tab3:
    st.header("Correlation & Covariance Analysis")

    corr = r_stocks.corr().round(4)
    cov  = r_stocks.cov().round(6)

    # ── Correlation Heatmap ───────────────────────────────
    st.subheader("Correlation Heatmap")
    fig_heat = px.imshow(
        corr, text_auto=True,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Pairwise Correlation Matrix (Daily Returns)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Rolling Correlation ───────────────────────────────
    st.subheader("Rolling Pairwise Correlation")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        stk_a = st.selectbox("Stock A", STOCKS, index=0, key="rca")
    with rc2:
        stk_b = st.selectbox("Stock B", STOCKS, index=min(1, n - 1), key="rcb")
    with rc3:
        rc_win = st.select_slider(
            "Window (days)", options=[30, 60, 90, 120], value=120, key="rcw"
        )
    if stk_a == stk_b:
        st.info("Select two different stocks.")
    else:
        roll_c = r_stocks[stk_a].rolling(rc_win).corr(r_stocks[stk_b])
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(
            x=roll_c.index, y=roll_c.values,
            mode="lines", line=dict(color="steelblue"),
        ))
        fig_rc.update_layout(
            title=f"Rolling {rc_win}-Day Correlation: {stk_a} vs {stk_b}",
            xaxis_title="Date", yaxis_title="Correlation",
            template="plotly_white",
        )
        st.plotly_chart(fig_rc, use_container_width=True)

    # ── Covariance Matrix ─────────────────────────────────
    with st.expander("View Covariance Matrix (Daily Returns)"):
        st.dataframe(cov, use_container_width=True)


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 4  –  Portfolio Optimisation                    ║
# ╚═══════════════════════════════════════════════════════╝
with tab4:
    st.header("Portfolio Construction & Optimisation")

    with st.spinner("Running optimisations…"):
        w_eq  = np.ones(n) / n
        w_gmv = _gmv(mu_full, S_full)
        w_tan = _tangency(mu_full, S_full, RF_ANNUAL)

        eq_ret,  eq_vol  = _pret(w_eq,  mu_full), _pvol(w_eq,  S_full)
        gmv_ret, gmv_vol = _pret(w_gmv, mu_full), _pvol(w_gmv, S_full)
        tan_ret, tan_vol = _pret(w_tan, mu_full), _pvol(w_tan, S_full)

        r_eq_s  = pd.Series(r_stocks.values @ w_eq,  index=r_stocks.index)
        r_gmv_s = pd.Series(r_stocks.values @ w_gmv, index=r_stocks.index)
        r_tan_s = pd.Series(r_stocks.values @ w_tan, index=r_stocks.index)

    # ── Custom Portfolio Sliders ──────────────────────────
    st.subheader("Custom Portfolio")
    st.caption("Sliders are independent — weights are automatically normalised to sum to 1.")
    slider_cols = st.columns(n)
    raw_w = {}
    for i, s in enumerate(STOCKS):
        with slider_cols[i]:
            raw_w[s] = st.slider(s, 0.0, 1.0,
                                  value=round(1.0 / n, 2),
                                  step=0.05, key=f"cw_{s}")
    total_raw = sum(raw_w.values())
    w_custom  = (np.array([raw_w[s] for s in STOCKS]) / total_raw
                 if total_raw > 0 else w_eq.copy())
    norm_df   = pd.DataFrame({"Normalised Weight": w_custom.round(4)}, index=STOCKS)
    st.dataframe(norm_df.T, use_container_width=True)

    cust_ret  = _pret(w_custom, mu_full)
    cust_vol  = _pvol(w_custom, S_full)
    r_cust_s  = pd.Series(r_stocks.values @ w_custom, index=r_stocks.index)

    # Custom portfolio live metrics
    c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
    c_m1.metric("Annual Return",  f"{cust_ret:.2%}")
    c_m2.metric("Annual Vol",     f"{cust_vol:.2%}")
    c_m3.metric("Sharpe",         f"{sharpe_ratio(r_cust_s, RF_DAILY):.4f}")
    c_m4.metric("Sortino",        f"{sortino_ratio(r_cust_s, RF_DAILY):.4f}")
    c_m5.metric("Max Drawdown",   f"{max_drawdown(r_cust_s):.2%}")

    st.markdown("---")

    # ── Portfolio Weights Bar Chart ───────────────────────
    st.subheader("Portfolio Weights")
    wts_df = pd.DataFrame({
        "Equal":    w_eq.round(4),
        "GMV":      w_gmv.round(4),
        "Tangency": w_tan.round(4),
        "Custom":   w_custom.round(4),
    }, index=STOCKS)
    fig_wts = go.Figure()
    colours = ["steelblue", "seagreen", "darkorange", "mediumpurple"]
    for col, clr in zip(wts_df.columns, colours):
        fig_wts.add_trace(go.Bar(name=col, x=wts_df.index, y=wts_df[col],
                                  marker_color=clr))
    fig_wts.update_layout(
        barmode="group",
        title="Portfolio Weights by Strategy",
        xaxis_title="Stock", yaxis_title="Weight",
        template="plotly_white",
    )
    st.plotly_chart(fig_wts, use_container_width=True)

    # ── Risk Contribution ─────────────────────────────────
    st.subheader("Risk Contribution (% of Portfolio Variance)")
    st.info(
        "**What is risk contribution?**  "
        "A stock with a 10% weight can contribute 25%+ of total portfolio variance if it "
        "is highly volatile or correlated with others. "
        "Formula: `PRC_i = w_i · (Σw)_i / σ²_p` — values sum to 1."
    )

    S_daily = r_stocks.cov().values
    prc_gmv = _risk_contribution(w_gmv, S_daily)
    prc_tan = _risk_contribution(w_tan, S_daily)

    prc_df = pd.DataFrame({
        "Weight (GMV)":  w_gmv.round(4),
        "PRC (GMV)":     prc_gmv.round(4),
        "Weight (Tan)":  w_tan.round(4),
        "PRC (Tan)":     prc_tan.round(4),
    }, index=STOCKS)

    fig_prc = go.Figure()
    fig_prc.add_trace(go.Bar(name="Weight (GMV)",  x=STOCKS, y=w_gmv,   marker_color="seagreen",   opacity=0.6))
    fig_prc.add_trace(go.Bar(name="PRC (GMV)",     x=STOCKS, y=prc_gmv, marker_color="seagreen",   opacity=1.0))
    fig_prc.add_trace(go.Bar(name="Weight (Tan)",  x=STOCKS, y=w_tan,   marker_color="darkorange", opacity=0.6))
    fig_prc.add_trace(go.Bar(name="PRC (Tan)",     x=STOCKS, y=prc_tan, marker_color="darkorange", opacity=1.0))
    fig_prc.update_layout(
        barmode="group",
        title="Portfolio Weight vs Risk Contribution",
        xaxis_title="Stock", yaxis_title="Proportion",
        template="plotly_white",
    )
    st.plotly_chart(fig_prc, use_container_width=True)
    st.dataframe(prc_df, use_container_width=True)

    # ── Efficient Frontier ────────────────────────────────
    st.subheader("Efficient Frontier & Capital Allocation Line")
    st.caption(
        "The **efficient frontier** shows the best possible return for each level of risk. "
        "The **Capital Allocation Line (CAL)** extends from the risk-free rate through the "
        "tangency portfolio — portfolios on the CAL mix the tangency portfolio with the risk-free asset."
    )

    with st.spinner("Computing efficient frontier…"):
        ef_vols, ef_rets = _efficient_frontier(mu_full, S_full, n_points=40)

    slope_cal = (tan_ret - RF_ANNUAL) / tan_vol
    cal_x     = np.linspace(0, max(ef_vols) * 1.1, 100)
    cal_y     = RF_ANNUAL + slope_cal * cal_x

    stock_vols   = (r_stocks.std() * math.sqrt(TRADING_DAYS)).values
    stock_rets   = (r_stocks.mean() * TRADING_DAYS).values
    bench_vol_pt = bench_aligned.std()  * math.sqrt(TRADING_DAYS)
    bench_ret_pt = bench_aligned.mean() * TRADING_DAYS

    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=ef_vols, y=ef_rets, mode="lines",
        name="Efficient Frontier", line=dict(color="royalblue", width=2),
    ))
    fig_ef.add_trace(go.Scatter(
        x=cal_x, y=cal_y, mode="lines",
        name="CAL", line=dict(color="black", width=1.5, dash="dash"),
    ))
    fig_ef.add_trace(go.Scatter(
        x=stock_vols, y=stock_rets, mode="markers+text",
        text=STOCKS, textposition="top center",
        marker=dict(size=9, color="grey"),
        name="Individual Stocks",
    ))
    fig_ef.add_trace(go.Scatter(
        x=[bench_vol_pt], y=[bench_ret_pt], mode="markers+text",
        text=["S&P 500"], textposition="top center",
        marker=dict(size=10, color="black", symbol="x"),
        name="S&P 500",
    ))
    for label, vol, ret, sym, col in [
        ("Equal",    eq_vol,   eq_ret,   "square",      "steelblue"),
        ("GMV",      gmv_vol,  gmv_ret,  "diamond",     "seagreen"),
        ("Tangency", tan_vol,  tan_ret,  "star",        "darkorange"),
        ("Custom",   cust_vol, cust_ret, "triangle-up", "mediumpurple"),
    ]:
        fig_ef.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers+text",
            text=[label], textposition="top center",
            marker=dict(size=14, color=col, symbol=sym),
            name=label,
        ))
    fig_ef.update_layout(
        title="Efficient Frontier, CAL & Portfolio Points",
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        template="plotly_white", height=550,
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    # ── Portfolio Comparison ──────────────────────────────
    st.subheader("Portfolio Comparison")

    port_rets = pd.concat([
        r_eq_s.rename("Equal"),
        r_gmv_s.rename("GMV"),
        r_tan_s.rename("Tangency"),
        r_cust_s.rename("Custom"),
        bench_aligned.rename("S&P 500"),
    ], axis=1).dropna()

    wealth_cmp = 10_000 * (1 + port_rets).cumprod()
    fig_cmp = go.Figure()
    for col in wealth_cmp.columns:
        fig_cmp.add_trace(go.Scatter(
            x=wealth_cmp.index, y=wealth_cmp[col], mode="lines", name=col,
        ))
    fig_cmp.update_layout(
        title="Cumulative Wealth: Portfolios vs S&P 500 ($10,000)",
        xaxis_title="Date", yaxis_title="Value ($)",
        hovermode="x unified", template="plotly_white",
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.subheader("Performance Summary Table")
    perf_table = pd.DataFrame({
        "Equal":    perf_summary(r_eq_s,        RF_DAILY),
        "GMV":      perf_summary(r_gmv_s,       RF_DAILY),
        "Tangency": perf_summary(r_tan_s,       RF_DAILY),
        "Custom":   perf_summary(r_cust_s,      RF_DAILY),
        "S&P 500":  perf_summary(bench_aligned, RF_DAILY),
    }).T.round(4)
    st.dataframe(perf_table, use_container_width=True)


# ╔═══════════════════════════════════════════════════════╗
# ║  TAB 5  –  Estimation Window Sensitivity             ║
# ╚═══════════════════════════════════════════════════════╝
with tab5:
    st.header("Estimation Window Sensitivity")
    st.info(
        "**Why does this matter?**  "
        "Mean-variance optimisation is sensitive to its inputs — small changes in estimated "
        "returns or covariances can produce dramatically different portfolio weights. "
        "The tables below show how GMV and Tangency portfolios change as you shorten the "
        "lookback window used to estimate inputs."
    )

    n_obs   = len(r_stocks)
    options = {}
    if n_obs >= 252:
        options["1 Year"]  = 252
    if n_obs >= 252 * 3:
        options["3 Years"] = 252 * 3
    if n_obs >= 252 * 5:
        options["5 Years"] = 252 * 5
    options["Full Sample"] = n_obs

    if len(options) < 2:
        st.warning("Need at least 2 years of data for a meaningful sensitivity analysis.")
    else:
        with st.spinner("Running sensitivity analysis…"):
            rows_gmv, rows_tan = [], []

            for label, lookback in options.items():
                r_sub  = r_stocks.iloc[-lookback:]
                mu_sub = (r_sub.mean() * TRADING_DAYS).values
                S_sub  = (r_sub.cov()  * TRADING_DAYS).values

                wg = _gmv(mu_sub, S_sub)
                wt = _tangency(mu_sub, S_sub, RF_ANNUAL)

                row_g = {"Window": label}
                row_t = {"Window": label}
                for i, s in enumerate(STOCKS):
                    row_g[s] = round(float(wg[i]), 4)
                    row_t[s] = round(float(wt[i]), 4)
                row_g["Annual Return"] = round(float(_pret(wg, mu_sub)), 4)
                row_g["Annual Vol"]    = round(float(_pvol(wg, S_sub)),  4)
                row_t["Annual Return"] = round(float(_pret(wt, mu_sub)), 4)
                row_t["Annual Vol"]    = round(float(_pvol(wt, S_sub)),  4)
                vt = _pvol(wt, S_sub)
                row_t["Sharpe"] = round(float((_pret(wt, mu_sub) - RF_ANNUAL) / vt), 4) if vt > 0 else np.nan

                rows_gmv.append(row_g)
                rows_tan.append(row_t)

        gmv_sens = pd.DataFrame(rows_gmv).set_index("Window")
        tan_sens = pd.DataFrame(rows_tan).set_index("Window")

        st.subheader("GMV Portfolio — Weights & Performance by Window")
        st.dataframe(gmv_sens, use_container_width=True)

        st.subheader("Tangency Portfolio — Weights & Performance by Window")
        st.dataframe(tan_sens, use_container_width=True)

        weight_cols = [c for c in gmv_sens.columns if c in STOCKS]

        st.subheader("GMV Weight Sensitivity")
        fig_s1 = go.Figure()
        for w_label in gmv_sens.index:
            fig_s1.add_trace(go.Bar(
                name=w_label, x=weight_cols,
                y=gmv_sens.loc[w_label, weight_cols].values,
            ))
        fig_s1.update_layout(
            barmode="group",
            title="GMV Portfolio Weights Across Estimation Windows",
            xaxis_title="Stock", yaxis_title="Weight",
            template="plotly_white",
        )
        st.plotly_chart(fig_s1, use_container_width=True)

        st.subheader("Tangency Weight Sensitivity")
        fig_s2 = go.Figure()
        for w_label in tan_sens.index:
            fig_s2.add_trace(go.Bar(
                name=w_label, x=weight_cols,
                y=tan_sens.loc[w_label, weight_cols].values,
            ))
        fig_s2.update_layout(
            barmode="group",
            title="Tangency Portfolio Weights Across Estimation Windows",
            xaxis_title="Stock", yaxis_title="Weight",
            template="plotly_white",
        )
        st.plotly_chart(fig_s2, use_container_width=True)