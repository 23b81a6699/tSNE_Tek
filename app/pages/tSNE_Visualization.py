import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils import load_object

st.title(
    "📉 t-SNE Visualization"
)

embedding = load_object(
    "artifacts/tsne_embedding.pkl"
)

clusters = pd.read_csv(
    "artifacts/cluster_labels.csv"
)

plot_df = pd.DataFrame(
    embedding,
    columns=["TSNE_1","TSNE_2"]
)

plot_df["Cluster"] = (
    clusters["Cluster"]
)

fig = px.scatter(
    plot_df,
    x="TSNE_1",
    y="TSNE_2",
    color="Cluster",
    title="Customer Segments in 2D Space",
    height=700
)

st.plotly_chart(
    fig,
    use_container_width=True
)