import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title(
    "📊 Customer Segmentation using KMeans + t-SNE"
)

st.markdown("---")

col1,col2,col3 = st.columns(3)

with col1:
    st.metric(
        "Algorithm",
        "KMeans"
    )

with col2:
    st.metric(
        "Visualization",
        "t-SNE"
    )

with col3:
    st.metric(
        "Learning Type",
        "Unsupervised"
    )

st.markdown("---")

st.markdown("""
## Project Overview

This project demonstrates:

- Data Cleaning
- Feature Engineering
- Feature Scaling
- Customer Segmentation using KMeans
- Dimensionality Reduction using t-SNE
- Interactive Visualization using Plotly

### Workflow

Dataset
↓
Preprocessing
↓
Scaling
↓
KMeans Clustering
↓
t-SNE Visualization
↓
Business Insights
""")

st.success(
    "Use the left sidebar to explore the dashboard."
)