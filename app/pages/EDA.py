import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("📊 Exploratory Data Analysis")

df = pd.read_csv(
    "artifacts/customer_data.csv"
)

# ==================================================
# Dataset Preview
# ==================================================

st.header("Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

# ==================================================
# Shape Metrics
# ==================================================

col1,col2,col3 = st.columns(3)

with col1:
    st.metric(
        "Rows",
        df.shape[0]
    )

with col2:
    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:
    st.metric(
        "Missing Values",
        df.isnull().sum().sum()
    )

st.markdown("---")

# ==================================================
# Missing Values
# ==================================================

st.header("Missing Values")

missing_df = pd.DataFrame(
    {
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    }
)

fig = px.bar(
    missing_df,
    x="Column",
    y="Missing Values",
    title="Missing Values by Feature"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# Numerical Features
# ==================================================

st.header("Numerical Feature Distribution")

numerical_cols = df.select_dtypes(
    include=["int64","float64"]
).columns

selected_feature = st.selectbox(
    "Select Numerical Feature",
    numerical_cols
)

fig = px.histogram(
    df,
    x=selected_feature,
    nbins=30,
    title=f"Distribution of {selected_feature}"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# Correlation Heatmap
# ==================================================

st.header("Correlation Heatmap")

corr = (
    df[numerical_cols]
    .corr()
)

fig, ax = plt.subplots(
    figsize=(12,8)
)

sns.heatmap(
    corr,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# ==================================================
# Income Distribution
# ==================================================

if "Income" in df.columns:

    st.header("Income Distribution")

    fig = px.box(
        df,
        y="Income",
        title="Income Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# Age Distribution
# ==================================================

if "Year_Birth" in df.columns:

    st.header("Age Distribution")

    df["Age"] = (
        2026 - df["Year_Birth"]
    )

    fig = px.histogram(
        df,
        x="Age",
        nbins=30,
        title="Customer Age Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# Education Analysis
# ==================================================

if "Education" in df.columns:

    st.header("Education Analysis")

    edu_counts = (
        df["Education"]
        .value_counts()
        .reset_index()
    )

    edu_counts.columns = [
        "Education",
        "Count"
    ]

    fig = px.pie(
        edu_counts,
        names="Education",
        values="Count",
        title="Education Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# Marital Status
# ==================================================

if "Marital_Status" in df.columns:

    st.header("Marital Status Analysis")

    marital_counts = (
        df["Marital_Status"]
        .value_counts()
        .reset_index()
    )

    marital_counts.columns = [
        "Marital Status",
        "Count"
    ]

    fig = px.bar(
        marital_counts,
        x="Marital Status",
        y="Count",
        title="Marital Status Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# Spending Analysis
# ==================================================

spending_columns = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

available_cols = [
    col
    for col in spending_columns
    if col in df.columns
]

if len(available_cols) > 0:

    st.header("Product Spending Analysis")

    spending_df = (
        df[available_cols]
        .sum()
        .reset_index()
    )

    spending_df.columns = [
        "Product",
        "Total Spending"
    ]

    fig = px.bar(
        spending_df,
        x="Product",
        y="Total Spending",
        title="Total Spending by Product Category"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# Campaign Analysis
# ==================================================

campaign_cols = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5"
]

campaign_cols = [
    col
    for col in campaign_cols
    if col in df.columns
]

if campaign_cols:

    st.header("Campaign Acceptance")

    campaign_df = (
        df[campaign_cols]
        .sum()
        .reset_index()
    )

    campaign_df.columns = [
        "Campaign",
        "Accepted"
    ]

    fig = px.bar(
        campaign_df,
        x="Campaign",
        y="Accepted",
        title="Campaign Acceptance Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# Pair Plot
# ==================================================

st.header("Feature Relationships")

pair_features = [
    col
    for col in [
        "Income",
        "MntWines",
        "MntMeatProducts",
        "NumWebPurchases"
    ]
    if col in df.columns
]

if len(pair_features) >= 2:

    fig = px.scatter_matrix(
        df,
        dimensions=pair_features
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# Outlier Detection
# ==================================================

st.header("Outlier Detection")

outlier_feature = st.selectbox(
    "Select Feature for Outlier Detection",
    numerical_cols,
    key="outlier"
)

fig = px.box(
    df,
    y=outlier_feature,
    title=f"Outlier Analysis: {outlier_feature}"
)

st.plotly_chart(
    fig,
    use_container_width=True
)