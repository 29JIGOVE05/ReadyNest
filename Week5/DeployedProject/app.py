import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import json
import os

st.set_page_config(page_title="Stock Analytics Dashboard", layout="wide", page_icon="📈")

DATA_PATH = os.path.join(os.path.dirname(__file__), "stock_cleaned.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "models", "feature_columns.json")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "models", "model_metrics.json")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return model, features, metrics


df = load_data()
model, features, metrics = load_model()

st.sidebar.title("Filters")
tickers = sorted(df["Ticker"].unique())
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)
date_range = st.sidebar.date_input(
    "Date Range",
    [df["Date"].min(), df["Date"].max()],
)

ticker_df = df[df["Ticker"] == selected_ticker].copy()
if len(date_range) == 2:
    mask = (
        (ticker_df["Date"] >= pd.to_datetime(date_range[0])) &
        (ticker_df["Date"] <= pd.to_datetime(date_range[1]))
    )
    ticker_df = ticker_df[mask]

st.title("📈 End-to-End Stock Analytics & Prediction Dashboard")
st.caption("Data pipeline: Collection → ETL → EDA → Predictive Model → Dashboard")

latest = ticker_df.iloc[-1]
prev = ticker_df.iloc[-2] if len(ticker_df) > 1 else latest
pct_change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close", f"${latest['Close']:.2f}", f"{pct_change:.2f}%")
col2.metric("7-Day MA", f"${latest['MA_7']:.2f}")
col3.metric("30-Day MA", f"${latest['MA_30']:.2f}")
col4.metric("RSI (14)", f"{latest['RSI_14']:.1f}")

st.divider()

st.subheader(f"{selected_ticker} — Price Trend with Moving Averages")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ticker_df["Date"],
    y=ticker_df["Close"],
    name="Close",
    line=dict(color="seagreen")
))
fig.add_trace(go.Scatter(
    x=ticker_df["Date"],
    y=ticker_df["MA_7"],
    name="MA 7",
    line=dict(color="orange", dash="dot")
))
fig.add_trace(go.Scatter(
    x=ticker_df["Date"],
    y=ticker_df["MA_30"],
    name="MA 30",
    line=dict(color="royalblue", dash="dot")
))
fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Price ($)")
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)

with c1:
    st.subheader("Trading Volume")
    fig_vol = go.Figure(
        go.Bar(
            x=ticker_df["Date"],
            y=ticker_df["Volume"],
            marker_color="teal"
        )
    )
    fig_vol.update_layout(height=350)
    st.plotly_chart(fig_vol, use_container_width=True)

with c2:
    st.subheader("Volatility (7-Day Rolling Std Dev)")
    fig_volat = go.Figure(
        go.Scatter(
            x=ticker_df["Date"],
            y=ticker_df["Volatility_7"],
            line=dict(color="crimson")
        )
    )
    fig_volat.update_layout(height=350)
    st.plotly_chart(fig_volat, use_container_width=True)

st.divider()

st.subheader("🔮 Next-Day Closing Price Prediction")

pred_col1, pred_col2 = st.columns([1, 2])

with pred_col1:
    latest_features = ticker_df.iloc[[-1]][features]
    prediction = model.predict(latest_features)[0]

    st.metric(
        "Predicted Next Close",
        f"${prediction:.2f}",
        f"{((prediction - latest['Close']) / latest['Close'] * 100):.2f}% vs latest"
    )

    st.markdown(f"**Model used:** {metrics['best_model']['model']}")
    st.markdown(f"**Test RMSE:** {metrics['best_model']['rmse']:.3f}")
    st.markdown(f"**Test R²:** {metrics['best_model']['r2']:.4f}")

with pred_col2:
    st.markdown("**Model Comparison (on held-out test data)**")

    comp_df = pd.DataFrame(metrics["all_models"])

    st.dataframe(
        comp_df.set_index("model").style.highlight_min(
            subset=["rmse", "mae"],
            color="lightgreen"
        ),
        use_container_width=True
    )

st.divider()

st.subheader("💡 Business Insights & Recommendations")

avg_volatility = (
    df.groupby("Ticker")["Volatility_7"]
    .mean()
    .sort_values(ascending=False)
)

most_volatile = avg_volatility.index[0]
least_volatile = avg_volatility.index[-1]

st.markdown(f"""
- **Most volatile stock in this dataset:** `{most_volatile}` — higher risk, potentially higher reward; suited to investors with higher risk tolerance.
- **Most stable stock:** `{least_volatile}` — lower volatility, may suit conservative/long-term holding strategies.
- **Model insight:** Simpler linear models can outperform complex ensembles for next-day price prediction, since short-horizon stock prices behave close to a random walk.
- **RSI signal:** An RSI above 70 typically signals an overbought condition; below 30 signals oversold.
""")

st.caption("Built as part of the End-to-End Data Pipeline & Predictive Analytics project.")