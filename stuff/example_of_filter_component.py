import streamlit as st # type: ignore
from filter_component import filter_dataframe
import pandas as pd

df = pd.read_pickle("pickle/synthetic_benchmarks_all-devices_all.pkl")

filtered_df = filter_dataframe(df, show_df_by_default=False)