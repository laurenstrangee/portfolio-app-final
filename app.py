# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with: uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import numpy as np
from scipy.optimize import minimize

# -- Page configuration ----------------------------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

tickers_input = st.sidebar.text_input(
    "Stock Tickers (3-10, comma separated)",
    value="AAPL,MSFT,AMZN,PG,JNJ,JPM"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Default date range: two years back
default_start = date.today() - timedelta(days=365*2)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today(), min_value=date(1970, 1, 1))

# Validate date range
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Moving average window
ma_window = st.sidebar.slider("Moving Average Window (days)", 5, 200, 50, 5)

# Risk-free rate for Sharpe ratio
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 20.0, 4.5, 0.1) / 100

# Rolling volatility window
vol_window = st.sidebar.slider("Rolling Volatility Window (days)", 10, 120, 30, 5)

# -- Data download ----------------------------------------
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(tickers, start: date, end: date) -> pd.DataFrame:
    df = yf.download(tickers + ["^GSPC"], start=start, end=end, progress=False)
    return df["Adj Close"]

# -- Main logic -------------------------------------------
if len(tickers) < 3 or len(tickers) > 10:
    st.error("Please enter between 3 and 10 tickers.")
    st.stop()

try:
    data = load_data(tickers, start_date, end_date)
except Exception as e:
    st.error(f"Download failed: {e}")
    st.stop()

if data.empty:
    st.error("No data found.")
    st.stop()

# Handle missing data
data = data.dropna()

returns = data.pct_change().dropna()
asset_returns = returns.drop(columns="^GSPC")
mean = asset_returns.mean() * 252
cov = asset_returns.cov() * 252

# -- Portfolio statistics function -----------------------
def portfolio_stats(w):
    port_ret = np.dot(w, mean)
    port_vol = np.sqrt(w.T @ cov @ w)
    sharpe = (port_ret - risk_free_rate) / port_vol
    
    # Sortino
    port_returns = asset_returns @ w
    downside = port_returns[port_returns < 0]
    downside_std = downside.std() * np.sqrt(252)
    sortino = (port_ret - risk_free_rate) / downside_std if downside_std != 0 else np.nan
    
    # Max Drawdown
    wealth = (1 + port_returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    max_dd = dd.min()
    
    return port_ret, port_vol, sharpe, sortino, max_dd

# -- Equal Weight Portfolio --------------------------------
st.header("Portfolio Analysis")
n = len(tickers)
w_eq = np.ones(n) / n

stats_eq = portfolio_stats(w_eq)
col1, col2, col3 = st.columns(3)
col1.metric("Return", f"{stats_eq[0]:.2%}")
col2.metric("Volatility", f"{stats_eq[1]:.2%}")
col3.metric("Sharpe", f"{stats_eq[2]:.2f}")
col4, col5 = st.columns(2)
col4.metric("Sortino", f"{stats_eq[3]:.2f}")
col5.metric("Max Drawdown", f"{stats_eq[4]:.2%}")

wealth = (1 + asset_returns @ w_eq).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=wealth.index, y=wealth, name="Equal Weight"))
st.plotly_chart(fig, width="stretch")

# -- Portfolio Optimization --------------------------------
st.header("Portfolio Optimization")

constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = tuple((0,1) for _ in range(n))

# GMV
def portfolio_variance(w):
    return w.T @ cov @ w

gmv = minimize(portfolio_variance, w_eq, bounds=bounds, constraints=constraints)
if not gmv.success: st.error("GMV optimization failed"); st.stop()
w_gmv = gmv.x

# Tangency
def neg_sharpe(w): return -portfolio_stats(w)[2]
tan = minimize(neg_sharpe, w_eq, bounds=bounds, constraints=constraints)
if not tan.success: st.error("Tangency optimization failed"); st.stop()
w_tan = tan.x

# Metrics Table
names = ["Equal Weight","GMV","Tangency"]
weights = [w_eq, w_gmv, w_tan]
results = [portfolio_stats(w) for w in weights]
df_results = pd.DataFrame(results, columns=["Return","Volatility","Sharpe","Sortino","Max Drawdown"], index=names)
st.subheader("Optimization Results")
st.dataframe(df_results)

# Portfolio Weights Chart
st.subheader("Portfolio Weights")
fig = go.Figure()
fig.add_trace(go.Bar(x=tickers, y=w_eq, name="Equal Weight"))
fig.add_trace(go.Bar(x=tickers, y=w_gmv, name="GMV"))
fig.add_trace(go.Bar(x=tickers, y=w_tan, name="Tangency"))
fig.update_layout(barmode='group', title="Portfolio Weights")
st.plotly_chart(fig, width="stretch")

# Risk Contribution
st.subheader("Risk Contribution")
for name, w in {"GMV": w_gmv, "Tangency": w_tan}.items():
    port_var = w.T @ cov @ w
    prc = (w * (cov @ w)) / port_var
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tickers, y=prc))
    fig.update_layout(title=f"{name} Risk Contribution")
    st.plotly_chart(fig)
    st.caption("Shows each asset's contribution to portfolio risk.")

# -- Custom Portfolio --------------------------------------
st.header("Custom Portfolio")
sliders = [st.slider(t, 0.0, 1.0, 1.0/n) for t in tickers]
w_custom = np.array(sliders)/np.sum(sliders)
st.write("Normalized Weights:")
st.write(dict(zip(tickers, w_custom.round(3))))

custom_stats = portfolio_stats(w_custom)
col1, col2, col3 = st.columns(3)
col1.metric("Return", f"{custom_stats[0]:.2%}")
col2.metric("Volatility", f"{custom_stats[1]:.2%}")
col3.metric("Sharpe", f"{custom_stats[2]:.2f}")
col4, col5 = st.columns(2)
col4.metric("Sortino", f"{custom_stats[3]:.2f}")
col5.metric("Max Drawdown", f"{custom_stats[4]:.2%}")

# -- Efficient Frontier + CAL --------------------------------
st.header("Efficient Frontier")
target_returns = np.linspace(mean.min(), mean.max(), 50)
vols = []

for r in target_returns:
    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: np.dot(w, mean) - r}
    )
    res = minimize(portfolio_variance, w_eq, bounds=bounds, constraints=cons)
    vols.append(np.sqrt(res.fun) if res.success else np.nan)

fig = go.Figure()
fig.add_trace(go.Scatter(x=vols, y=target_returns, mode='lines', name='Frontier'))

# Key portfolios
for name, w in [("GMV", w_gmv), ("Tangency", w_tan), ("Equal Weight", w_eq), ("Custom", w_custom)]:
    s = portfolio_stats(w)
    fig.add_trace(go.Scatter(x=[s[1]], y=[s[0]], mode='markers', name=name))

# CAL
tan_stats = portfolio_stats(w_tan)
rf_line = np.linspace(0, max(vols), 50)
cal = risk_free_rate + (tan_stats[0]-risk_free_rate)/tan_stats[1]*rf_line
fig.add_trace(go.Scatter(x=rf_line, y=cal, mode='lines', name='CAL'))

fig.update_layout(xaxis_title="Volatility", yaxis_title="Return", title="Efficient Frontier")
st.plotly_chart(fig, width="stretch")
st.caption("Efficient frontier shows optimal portfolios. CAL shows best risk-return tradeoff.")

# -- Portfolio Comparison ---------------------------------
st.header("Portfolio Comparison")
port_returns_df = pd.DataFrame({
    "Equal Weight": asset_returns @ w_eq,
    "GMV": asset_returns @ w_gmv,
    "Tangency": asset_returns @ w_tan,
    "Custom": asset_returns @ w_custom,
    "S&P 500": returns["^GSPC"]
})
wealth = (1 + port_returns_df).cumprod()
fig = go.Figure()
for col in wealth.columns:
    fig.add_trace(go.Scatter(x=wealth.index, y=wealth[col], name=col))
st.plotly_chart(fig, width="stretch")

# -- Correlation Matrix -----------------------------------
st.header("Correlation Matrix")
corr = asset_returns.corr()
fig = go.Figure(data=go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.columns,
    colorscale="RdBu", zmin=-1, zmax=1
))
st.plotly_chart(fig)

# -- Sensitivity Analysis ---------------------------------
st.header("Sensitivity Analysis")
windows = {"1 Year":252, "3 Years":756, "5 Years":1260}
results = []

for name, w_len in windows.items():
    if len(asset_returns) > w_len:
        sub = asset_returns.tail(w_len)
        m = sub.mean()*252
        c = sub.cov()*252
        res = minimize(lambda w: w.T @ c @ w, w_eq, bounds=bounds, constraints=constraints)
        if res.success:
            results.append([name, np.dot(res.x, m), np.sqrt(res.x.T @ c @ res.x)])

df_sens = pd.DataFrame(results, columns=["Window","Return","Volatility"])
st.dataframe(df_sens)
st.caption("Optimization results vary depending on estimation window.")

# -- Sidebar Info -----------------------------------------
st.sidebar.subheader("About")
st.sidebar.write("""
- Uses simple returns
- 252 trading day annualization
- Mean-variance framework
- Data from Yahoo Finance
""")