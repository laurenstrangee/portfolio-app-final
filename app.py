# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import math
import numpy as np
from scipy import stats

# -- Page configuration ----------------------------------
# st.set_page_config must be the FIRST Streamlit command in the script.
# If you add any other st.* calls above this line, you'll get an error.
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

tickers_input = st.sidebar.text_input(
    "Stock Tickers (3-10, comma separated)",
    value="AAPL,MSFT,AMZN,PG,JNJ,JPM"
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
# Default date range: one year back from today
default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today(),min_value=date(1970, 1, 1))

# Validate that the date range makes sense
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()
# Let the user pick a moving-average window
ma_window = st.sidebar.slider(
    "Moving Average Window (days)", min_value=5, max_value=200, value=50, step=5
)
# Risk-free rate for Sharpe ratio calculation
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=4.5, step=0.1
) / 100
# Rolling volatility window
vol_window = st.sidebar.slider(
    "Rolling Volatility Window (days)", min_value=10, max_value=120, value=30, step=5
)

# -- Data download ----------------------------------------
# We wrap the download in st.cache_data so repeated runs with
# the same inputs don't re-download every time. The ttl (time-to-live)
# ensures the cache expires after one hour so data stays fresh.
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(tickers, start: date, end: date) -> pd.DataFrame:
    df = yf.download(tickers + ["^GSPC"], start=start, end=end, progress=False)
    return df["Adj Close"]

# -- Main logic -------------------------------------------
if len(tickers) < 3 or len(tickers) > 10:
    st.error("Please enter between 3 and 10 tickers.")
    st.stop()

if (end_date - start_date).days < 365*2:
    st.error("Date range must be at least 2 years.")
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

def portfolio_stats(w):
    port_ret = np.dot(w, mean)
    port_vol = np.sqrt(w.T @ cov @ w)

    sharpe = (port_ret - risk_free_rate) / port_vol

    port_returns = asset_returns @ w

    downside = port_returns[port_returns < risk_free_rate/252]
    downside_std = downside.std() * np.sqrt(252)

    sortino = (port_ret - risk_free_rate) / downside_std if downside_std != 0 else np.nan

    wealth = (1 + port_returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    max_dd = dd.min()

    return port_ret, port_vol, sharpe, sortino, max_dd
st.header("Portfolio Analysis")

n = len(tickers)
w_eq = np.ones(n) / n  # YOUR ORIGINAL PROJECT

# Portfolio returns
port_returns = asset_returns @ w_eq

# Metrics
stats_eq = portfolio_stats(w_eq)

st.subheader("My Portfolio (Equal Weight from Project 1)")

col1, col2, col3 = st.columns(3)
col1.metric("Return", f"{stats_eq[0]:.2%}")
col2.metric("Volatility", f"{stats_eq[1]:.2%}")
col3.metric("Sharpe", f"{stats_eq[2]:.2f}")

col4, col5 = st.columns(2)
col4.metric("Sortino", f"{stats_eq[3]:.2f}")
col5.metric("Max Drawdown", f"{stats_eq[4]:.2%}")

# Wealth chart
wealth = (1 + port_returns).cumprod()

fig = go.Figure()
fig.add_trace(go.Scatter(x=wealth.index, y=wealth, name="My Portfolio"))
st.plotly_chart(fig, width="stretch")

st.write("""
This portfolio replicates my original project:
- Equal weight across AAPL, MSFT, AMZN, PG, JNJ, JPM
- Provides diversification across sectors
- Serves as baseline vs optimized portfolios
""")
st.header("Correlation Matrix")

corr = asset_returns.corr()

fig = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    text=np.round(corr.values, 2),
    texttemplate="%{text}",
    colorscale="RdBu"
))

st.plotly_chart(fig, width="stretch")

st.sidebar.subheader("About")

st.sidebar.write("""
- Uses simple returns
- 252 trading day annualization
- Mean-variance framework
- Data from Yahoo Finance
""")