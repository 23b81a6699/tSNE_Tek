import os
import sys

# Add project root to Python path
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils import load_object

st.title(
    "📉 t-SNE Visualization"
)

import pickle

with open(
    "artifacts/tsne_embedding.pkl",
    "rb"
) as file:
    embedding = pickle.load(file)

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