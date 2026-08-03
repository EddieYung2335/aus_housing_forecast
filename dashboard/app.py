from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

st.set_page_config(page_title="Housing Forecast", layout="wide")


@st.cache_data
def load_data():
    predictions = pd.read_parquet(PROCESSED_DIR / "predictions.parquet")
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    return predictions, features


predictions, features = load_data()

st.title("Australia Housing Forecast")

# Region Selection
regions = sorted(predictions["region"].unique())
default = 0
if "Brisbane" in regions:
    default = regions.index("Brisbane")
selected = st.selectbox("Region", regions, index=default)

row = predictions.loc[predictions["region"] == selected].iloc[0]
hist = features.loc[features["region"] == selected].sort_values("date")

# Data of current price, predicted price, percentage increase, and range
pct = (row["pred_price"] / row["last_price"] - 1) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Current", f"\\${row['last_price']:,.0f}k")
c2.metric("Forecast", f"\\${row['pred_price']:,.0f}k", delta=f"{pct:+.2f}%")
c3.metric("Range", f"\\${row['lower']:,.0f}k – \\${row['upper']:,.0f}k")

# Main Graph
#
# Include the data from most recent 10 years only
cutoff = hist.date.max() - pd.DateOffset(years=10)
hist_plot = hist[hist.date >= cutoff]

fig = go.Figure()
fig.add_scatter(
    x=hist_plot.date, y=hist_plot.median_price, mode="lines", name="History"
)
fig.add_scatter(
    x=[row.last_date, row.target_date],
    y=[row.last_price, row.pred_price],
    mode="lines",
    line=dict(dash="dot"),
    showlegend=False,
)

fig.add_scatter(
    x=[row.target_date],
    y=[row.pred_price],
    mode="markers",
    name="Forecast",
    error_y=dict(
        type="data",
        symmetric=False,
        array=[row.upper - row.pred_price],
        arrayminus=[row.pred_price - row.lower],
        visible=True,
    ),
)

fig.update_layout(
    yaxis_title="Median price ($'000)",
    xaxis_title=None,
    hovermode="x unified",
)

st.plotly_chart(fig, width="stretch")

# So we dont touch the data in cache
table = predictions.copy()
table["change_pct"] = (table.pred_price / table.last_price - 1) * 100
table = table[["region", "last_price", "pred_price", "change_pct"]]
table = table.sort_values("change_pct", ascending=False)

st.dataframe(
    table,
    hide_index=True,
    column_config={
        "region": "Region",
        "last_price": st.column_config.NumberColumn("Current ($'000)", format="%.0f"),
        "pred_price": st.column_config.NumberColumn("Forecast ($'000)", format="%.0f"),
        "change_pct": st.column_config.NumberColumn("Change", format="%+.2f%%"),
    },
)

st.caption(
    f"Model: {row['model_name']} \n\nData through {row['last_date']:%Y-%m}      "
    f"Forecast for {row['target_date']:%Y-%m}  \n\n"
    f"Range is ±1 test MAE in log space (typical error, not a confidence interval)."
)
