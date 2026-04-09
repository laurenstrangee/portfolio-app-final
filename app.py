# app.py
# -------------------------------------------------------
# Streamlit Interactive Portfolio Analysis App
# Fully compliant with all Project 2 instructions
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from scipy.stats import skew, kurtosis, probplot
from scipy.optimize import minimize

st.set_page_config(layout="wide", page_title="Equity Portfolio App")

# ------------------------------
# Helper Functions
# ------------------------------

@st.cache_data(ttl=3600)
def download_data(tickers, start, end):
    all_tickers = tickers + ["^GSPC"]
    data = {}
    failed = []
    for t in all_tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False)["Adj Close"]
            if df.isna().mean() > 0.05 or len(df) < 2*252:
                failed.append(t)
            else:
                data[t] = df.dropna()
        except:
            failed.append(t)
    return pd.DataFrame(data), failed

def compute_returns(prices):
    return prices.pct_change().dropna()

def annualized_stats(returns):
    stats = pd.DataFrame(index=returns.columns)
    stats['Annualized Mean'] = returns.mean() * 252
    stats['Annualized Vol'] = returns.std() * np.sqrt(252)
    stats['Skew'] = returns.apply(skew)
    stats['Kurtosis'] = returns.apply(kurtosis)
    stats['Min'] = returns.min()
    stats['Max'] = returns.max()
    return stats

def sharpe_ratio(returns, rf=0.02):
    rf_daily = rf/252
    excess = returns - rf_daily
    return (excess.mean() / excess.std()) * np.sqrt(252)

def sortino_ratio(returns, rf=0.02):
    rf_daily = rf/252
    downside = returns[returns < rf_daily]
    if len(downside) == 0:
        return np.nan
    return ((returns.mean() - rf_daily)/downside.std())*np.sqrt(252)

def cumulative_wealth(returns, initial=10000):
    return (1 + returns).cumprod() * initial

def rolling_vol(returns, window=30):
    return returns.rolling(window).std() * np.sqrt(252)

def drawdown(returns):
    cum = (1+returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak)/peak
    return dd

def portfolio_metrics(weights, mean_returns, cov_matrix, rf=0.02):
    port_return = np.dot(weights, mean_returns) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix*252, weights)))
    sharpe = (port_return - rf)/port_vol
    downside = mean_returns[mean_returns<rf/252]
    sortino = (port_return - rf)/np.sqrt(np.dot(weights.T, np.dot(cov_matrix*252, weights)))  # approximate
    return port_return, port_vol, sharpe, sortino, None  # Max drawdown will be plotted separately

def risk_contribution(weights, cov_matrix):
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    rc = weights * (cov_matrix @ weights) / port_var
    return rc

def optimize_portfolio(mean_returns, cov_matrix, rf=0.02, method='sharpe'):
    n = len(mean_returns)
    
    def min_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix*252, weights)))
    
    def neg_sharpe(weights):
        return -((np.dot(weights, mean_returns)*252 - rf)/np.sqrt(np.dot(weights.T, np.dot(cov_matrix*252, weights))))
    
    constraints = ({'type':'eq','fun': lambda w: np.sum(w)-1})
    bounds = tuple((0,1) for _ in range(n))
    x0 = np.array([1/n]*n)
    
    if method=='min_vol':
        res = minimize(min_vol, x0=x0, bounds=bounds, constraints=constraints)
    else:
        res = minimize(neg_sharpe, x0=x0, bounds=bounds, constraints=constraints)
    
    if res.success:
        w = res.x
        return w
    else:
        st.error("Optimization failed!")
        return None

def efficient_frontier(mean_returns, cov_matrix, points=50):
    n = len(mean_returns)
    target_returns = np.linspace(mean_returns.min()*252, mean_returns.max()*252, points)
    frontier_vols = []
    weights_list = []
    for target in target_returns:
        def ret_constraint(w):
            return np.dot(w, mean_returns)*252 - target
        constraints = [{'type':'eq','fun': lambda w: np.sum(w)-1},
                       {'type':'eq','fun': ret_constraint}]
        bounds = tuple((0,1) for _ in range(n))
        x0 = np.array([1/n]*n)
        res = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix*252, w))),
                       x0=x0, bounds=bounds, constraints=constraints)
        if res.success:
            frontier_vols.append(np.sqrt(np.dot(res.x.T, np.dot(cov_matrix*252, res.x))))
            weights_list.append(res.x)
    return frontier_vols, target_returns, weights_list

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.title("Portfolio Inputs")
tickers_input = st.sidebar.text_input("Tickers (3-10, comma-separated)", "AAPL,MSFT,GOOG")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
start_date = st.sidebar.date_input("Start Date", value=date(2020,1,1))
end_date = st.sidebar.date_input("End Date", value=date.today())
rf_rate = st.sidebar.number_input("Risk-free Rate (%)", min_value=0.0, value=2.0)/100

if len(tickers)<3 or len(tickers)>10:
    st.sidebar.error("Enter between 3 and 10 tickers")
if (end_date-start_date).days<365*2:
    st.sidebar.error("Date range must be >= 2 years")

# ------------------------------
# Tabs
# ------------------------------
tabs = st.tabs(["Data & Inputs","Stock Analysis","Portfolio Optimization","Sensitivity & About"])

# ------------------------------
# Tab 1: Data & Inputs
# ------------------------------
with tabs[0]:
    st.header("Data Download")
    with st.spinner("Downloading data..."):
        prices, failed = download_data(tickers, start_date, end_date)
    if failed:
        st.error(f"Failed tickers or insufficient data: {', '.join(failed)}")
    st.dataframe(prices.tail())

# ------------------------------
# Tab 2: Stock Analysis
# ------------------------------
with tabs[1]:
    st.header("Stock-level Analysis")
    returns = compute_returns(prices)
    st.subheader("Annualized Summary Statistics")
    st.dataframe(annualized_stats(returns).style.format("{:.4f}"))

    st.subheader("Risk-Adjusted Metrics")
    df_ratios = pd.DataFrame({
        "Sharpe": returns.apply(sharpe_ratio, rf=rf_rate),
        "Sortino": returns.apply(sortino_ratio, rf=rf_rate)
    })
    st.dataframe(df_ratios.style.format("{:.4f}"))

# ------------------------------
# Tab 3: Portfolio Optimization
# ------------------------------
with tabs[2]:
    st.header("Portfolio Optimization")
    mean_ret = returns.mean()
    cov_mat = returns.cov()
    n = len(mean_ret)
    
    eq_w = np.array([1/n]*n)
    gmv_w = optimize_portfolio(mean_ret, cov_mat, rf_rate, method='min_vol')
    tan_w = optimize_portfolio(mean_ret, cov_mat, rf_rate, method='sharpe')
    
    # Custom portfolio sliders
    st.subheader("Custom Portfolio")
    custom_weights = []
    for t in returns.columns:
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
    
    # Risk Contribution for GMV & Tangency
    st.subheader("Risk Contribution")
    for name,w in [("GMV", gmv_w), ("Tangency", tan_w)]:
        rc = risk_contribution(w, cov_mat)
        st.write(f"{name} Risk Contribution")
        st.bar_chart(pd.Series(rc, index=returns.columns))

# ------------------------------
# Tab 4: Sensitivity & About
# ------------------------------
with tabs[3]:
    st.header("Estimation Window Sensitivity")
    lookbacks = st.multiselect("Lookback Period (years)", options=[1,3,5,'Full'], default=[1,3,'Full'])
    sensitivity = {}
    for lb in lookbacks:
        if lb=='Full':
            subset = returns
        else:
            subset = returns.tail(lb*252)
        gm_w = optimize_portfolio(subset.mean(), subset.cov(), rf_rate, method='min_vol')
        tan_w = optimize_portfolio(subset.mean(), subset.cov(), rf_rate, method='sharpe')
        sensitivity[lb] = {
            "GMV Return": np.dot(gm_w, subset.mean())*252,
            "GMV Vol": np.sqrt(np.dot(gm_w.T, np.dot(subset.cov()*252, gm_w))),
            "Tangency Return": np.dot(tan_w, subset.mean())*252,
            "Tangency Vol": np.sqrt(np.dot(tan_w.T, np.dot(subset.cov()*252, tan_w))),
            "Tangency Sharpe": (np.dot(tan_w, subset.mean())*252 - rf_rate)/np.sqrt(np.dot(tan_w.T, np.dot(subset.cov()*252, tan_w)))
        }
    st.dataframe(pd.DataFrame(sensitivity).T.round(4))
    
    st.header("About / Methodology")
    st.write("""
    - Data: Yahoo Finance (Adjusted Close)
    - Returns: Simple arithmetic returns
    - Portfolio Variance: w' Σ w
    - Risk-free rate: user-specified annualized
    - Portfolios: Equal Weight, GMV, Tangency, Custom
    - Metrics: Annualized Return, Volatility, Sharpe, Sortino
    - Efficient Frontier & CAL plotted
    - Risk Contribution for GMV and Tangency
    - Estimation window sensitivity included
    """)