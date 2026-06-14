from __future__ import annotations

import pandas as pd
import streamlit as st

from football_forecast.data.io import read_matches_csv
from football_forecast.features.build import build_feature_table

st.set_page_config(page_title="Football Forecasting Engine", layout="wide")
st.title("Football Forecasting Engine")
st.caption("Leakage-safe probabilistic football forecasting scaffold")

uploaded = st.file_uploader("Upload canonical matches CSV", type=["csv"])

if uploaded is None:
    st.info("Using sample data. Upload a canonical matches CSV for real experiments.")
    path = "data/sample/international_results_sample.csv"
    matches = read_matches_csv(path)
else:
    raw = pd.read_csv(uploaded)
    tmp_path = "data/interim/uploaded_matches.csv"
    raw.to_csv(tmp_path, index=False)
    matches = read_matches_csv(tmp_path)

features = build_feature_table(matches, windows=(3, 5))
st.subheader("Matches")
st.dataframe(matches.tail(20), use_container_width=True)

st.subheader("Feature sample")
st.dataframe(features.tail(20), use_container_width=True)

st.warning("This app scaffold does not train a production model yet. Use scripts/run_sample_pipeline.py for the runnable sample pipeline.")
