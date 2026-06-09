import streamlit as st
import pandas as pd
import plotly.express as px

st.title(
    "🎯 Cluster Analysis"
)

clusters = pd.read_csv(
    "artifacts/cluster_labels.csv"
)

cluster_counts = (
    clusters["Cluster"]
    .value_counts()
    .reset_index()
)

cluster_counts.columns = [
    "Cluster",
    "Count"
]

fig = px.bar(
    cluster_counts,
    x="Cluster",
    y="Count",
    title="Cluster Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    cluster_counts,
    use_container_width=True
)