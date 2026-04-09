# app.py
# -------------------------------------------------------
# Interactive Portfolio App - Fixed Tickers
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis, probplot

# ------------------------------
# Constants / Predefined Stocks
# ------------------------------
STOCKS = ["AAPL", "MSFT", "AMZN", "PG", "JNJ", "JPM"]
BENCHMARK = "^GSPC"
ALL_TICKERS = STOCKS + [BENCHMARK]

# ------------------------------
# Cached Data Download
# ------------------------------
@st.cache_data(ttl=3600)
def download_data(tickers, start, end):
    failed = []
    try:
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        # If single ticker, turn into DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()
        # Drop columns with >5% missing
        missing = data.isna().mean()
        for col in missing[missing > 0.05].index:
            failed.append(col)
            data = data.drop(columns=col)
        if data.empty:
            failed = tickers
        return data, failed
    except Exception as e:
        return pd.DataFrame(), tickers

# ------------------------------
# Return Computation
# ------------------------------
def compute_returns(prices):
    return prices.pct_change().dropna()

def annualized_stats(returns):
    df = pd.DataFrame()
    df['Mean'] = returns.mean()*252
    df['Vol'] = returns.std()*np.sqrt(252)
    df['Skew'] = returns.apply(skew)
    df['Kurtosis'] = returns.apply(kurtosis)
    df['Min'] = returns.min()
    df['Max'] = returns.max()
    return df

# ------------------------------
# Risk Metrics
# ------------------------------
def sharpe_ratio(ret, rf=0.02):
    ann_ret = ret.mean()*252
    ann_vol = ret.std()*np.sqrt(252)
    if ann_vol == 0: return np.nan
    return (ann_ret - rf)/ann_vol

def sortino_ratio(ret, rf=0.02):
    ann_ret = ret.mean()*252
    downside = ret[ret < rf/252].std()*np.sqrt(252)
    if downside == 0: return np.nan
    return (ann_ret - rf)/downside

# ------------------------------
# Portfolio Metrics
# ------------------------------
def portfolio_metrics(weights, mean_ret, cov_mat, rf=0.02):
    w = np.array(weights)
    port_ret = np.dot(w, mean_ret*252)
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat*252, w)))
    if port_vol == 0: sharpe = np.nan
    else: sharpe = (port_ret - rf)/port_vol
    downside = np.sqrt(np.dot(w.T, np.dot(np.where(mean_ret*252<rf, mean_ret*252-rf,0)**2, w)))
    if downside == 0: sortino = np.nan
    else: sortino = (port_ret - rf)/port_vol
    return port_ret, port_vol, sharpe, sortino, w

# ------------------------------
# Portfolio Optimization
# ------------------------------
def optimize_portfolio(mean_ret, cov_mat, rf=0.02, method='min_vol'):
    n = len(mean_ret)
    bounds = [(0,1) for _ in range(n)]
    constraints = {'type':'eq', 'fun': lambda w: np.sum(w)-1}
    
    def vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_mat*252, w)))
    
    def neg_sharpe(w):
        r,v,s,so,_ = portfolio_metrics(w, mean_ret, cov_mat, rf)
        return -s
    
    x0 = np.array([1/n]*n)
    try:
        if method=='min_vol':
            res = minimize(vol, x0, bounds=bounds, constraints=constraints)
        else:
            res = minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints)
        if res.success:
            return res.x
        else:
            return x0
    except:
        return x0

# ------------------------------
# Efficient Frontier
# ------------------------------
def efficient_frontier(mean_ret, cov_mat, points=50):
    n = len(mean_ret)
    bounds = [(0,1) for _ in range(n)]
    constraints = {'type':'eq', 'fun': lambda w: np.sum(w)-1}
    target_returns = np.linspace(mean_ret.min()*252, mean_ret.max()*252, points)
    volatilities = []
    weights = []
    for tr in target_returns:
        def fun(w): return np.sqrt(np.dot(w.T, np.dot(cov_mat*252, w)))
        cons = [constraints, {'type':'eq','fun': lambda w: np.dot(w, mean_ret*252)-tr}]
        x0 = np.array([1/n]*n)
        try:
            res = minimize(fun, x0, bounds=bounds, constraints=cons)
            if res.success:
                volatilities.append(res.x.dot(cov_mat*252).dot(res.x)**0.5)
                weights.append(res.x)
            else:
                volatilities.append(np.nan)
                weights.append(np.nan)
        except:
            volatilities.append(np.nan)
            weights.append(np.nan)
    return volatilities, target_returns, weights

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Portfolio Configuration")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-04-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-04-01"))
rf_rate = st.sidebar.number_input("Risk-free rate (%)", min_value=0.0, max_value=10.0, value=2.0)/100

# ------------------------------
# Tabs
# ------------------------------
tabs = st.tabs(["Data & Inputs","Stock Analysis","Portfolio Optimization","About"])

# ------------------------------
# Tab 1: Data & Inputs
# ------------------------------
with tabs[0]:
    st.header("Data Download")
    with st.spinner("Downloading data..."):
        prices, failed = download_data(ALL_TICKERS, start_date, end_date)
    if failed:
        st.error(f"Failed tickers or insufficient data: {', '.join(failed)}")
    if prices.empty:
        st.stop()
    st.dataframe(prices.tail())

# ------------------------------
# Tab 2: Stock Analysis
# ------------------------------
with tabs[1]:
    st.header("Stock Analysis")
    returns = compute_returns(prices)
    st.subheader("Annualized Summary Statistics")
    st.dataframe(annualized_stats(returns).style.format("{:.4f}"))
    
    st.subheader("Risk-Adjusted Metrics")
    df_ratios = pd.DataFrame({
        "Sharpe": returns.apply(lambda x: sharpe_ratio(x, rf_rate)),
        "Sortino": returns.apply(lambda x: sortino_ratio(x, rf_rate))
    })
    st.dataframe(df_ratios.style.format("{:.4f}"))

# ------------------------------
# Tab 3: Portfolio Optimization
# ------------------------------
with tabs[2]:
    st.header("Portfolio Optimization")
    mean_ret = returns[STOCKS].mean()
    cov_mat = returns[STOCKS].cov()
    n = len(mean_ret)
    
    eq_w = np.array([1/n]*n)
    gmv_w = optimize_portfolio(mean_ret, cov_mat, rf_rate, method='min_vol')
    tan_w = optimize_portfolio(mean_ret, cov_mat, rf_rate, method='sharpe')
    
    # Custom portfolio sliders
    st.subheader("Custom Portfolio")
    custom_weights = []
    for t in STOCKS:
        w = st.slider(f"{t} Weight", 0.0, 1.0, 1/n, 0.01)
        custom_weights.append(w)
    custom_weights = np.array(custom_weights)/sum(custom_weights)
    
    # Portfolio metrics
    portfolios = {"Equal Weight": eq_w, "GMV": gmv_w, "Tangency": tan_w, "Custom": custom_weights}
    metrics = {}
    for name, w in portfolios.items():
        r,v,s,so,_ = portfolio_metrics(w, mean_ret, cov_mat, rf_rate)
        metrics[name] = [r,v,s,so]
    st.dataframe(pd.DataFrame(metrics, index=["Return","Volatility","Sharpe","Sortino"]).T.round(4))
    
    # Efficient Frontier
    ef_vol, ef_ret, ef_w = efficient_frontier(mean_ret, cov_mat)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ef_vol, y=ef_ret, mode='lines', name="Efficient Frontier"))
    for name, w in portfolios.items():
        r,v,s,so,_ = portfolio_metrics(w, mean_ret, cov_mat, rf_rate)
        fig.add_trace(go.Scatter(x=[v], y=[r], mode='markers', name=name, marker=dict(size=10)))
    cal_x = np.linspace(0, max(ef_vol)*1.1, 100)
    cal_y = rf_rate + ((portfolio_metrics(tan_w, mean_ret, cov_mat, rf_rate)[0]-rf_rate)/portfolio_metrics(tan_w, mean_ret, cov_mat, rf_rate)[1])*cal_x
    fig.add_trace(go.Scatter(x=cal_x, y=cal_y, mode='lines', name="Capital Allocation Line", line=dict(dash='dash')))
    fig.update_layout(title="Efficient Frontier & CAL", xaxis_title="Volatility", yaxis_title="Expected Return")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab 4: About
# ------------------------------
with tabs[3]:
    st.header("About & Methodology")
    st.markdown("""
    **Portfolio App**
    
    - Stocks: AAPL, MSFT, AMZN, PG, JNJ, JPM
    - Benchmark: S&P 500 (^GSPC)
    - Returns are simple daily returns
    - Annualized Return = mean_daily * 252
    - Annualized Volatility = std_daily * sqrt(252)
    - Sharpe ratio = (Return - Risk-free) / Volatility
    - Sortino ratio uses downside deviation
    - Portfolio optimization uses no-short-selling constraints
    - Efficient frontier generated using constrained optimization
    - Data source: Yahoo Finance via yfinance
    """)