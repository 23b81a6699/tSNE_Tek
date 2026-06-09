import streamlit as st
import pandas as pd

st.title(
    "🔍 Customer Explorer"
)

customers = pd.read_csv(
    "artifacts/customer_data.csv"
)

clusters = pd.read_csv(
    "artifacts/cluster_labels.csv"
)

customers["Cluster"] = (
    clusters["Cluster"]
)

selected_row = st.slider(
    "Select Customer",
    0,
    len(customers)-1,
    0
)

st.dataframe(
    customers.iloc[[selected_row]],
    use_container_width=True
)

st.success(
    f"Cluster: {customers.iloc[selected_row]['Cluster']}"
)