import plotly.express as px
import streamlit as st

from src.config import PROJECT_TITLE, SUBJECT
from src.ui import card, kpi_row, page_header, show


def render(df):
    page_header(PROJECT_TITLE, SUBJECT)

    st.markdown(
        "This decision-support system integrates **weather, soil and crop data** into a SQLite data warehouse, applies "
        "**data-mining techniques** (classification, regression, PCA, clustering and anomaly detection) and delivers "
        "**crop yield forecasts and farming recommendations** through interactive dashboards.")

    kpi_row([
        ("Total Records", f"{len(df):,}", "Fact table rows"),
        ("Districts", df["district"].nunique(), f"{df['zone'].nunique()} agro-climatic zones"),
        ("Crops", df["crop_name"].nunique(), f"{df['crop_type'].nunique()} crop categories"),
        ("Average Yield", f"{df['yield_t_ha'].mean():.2f} t/ha", f"{df['year'].min()} - {df['year'].max()}"),
        ("Total Production", f"{df['production_t'].sum() / 1e6:.2f} M t", "All crops, all seasons"),
    ])

    st.subheader("System Workflow")
    st.markdown(
        '<div class="flow">'
        '<div class="step"><b>1. Data Sources</b><span>Weather, soil, crop yield records</span></div><div class="arrow">➜</div>'
        '<div class="step"><b>2. ETL</b><span>Extract, clean, transform, load</span></div><div class="arrow">➜</div>'
        '<div class="step"><b>3. Data Warehouse</b><span>SQLite star schema</span></div><div class="arrow">➜</div>'
        '<div class="step"><b>4. OLAP</b><span>Roll-up, drill-down, slice, dice</span></div><div class="arrow">➜</div>'
        '<div class="step"><b>5. Data Mining</b><span>PCA, clustering, classification</span></div><div class="arrow">➜</div>'
        '<div class="step"><b>6. Decision Support</b><span>Predictions &amp; recommendations</span></div>'
        '</div>', unsafe_allow_html=True)

    st.subheader("Key Modules")
    c = st.columns(4)
    mods = [
        ("📊 Analytics Dashboard", "KPIs, yield trends, district comparison, heat-maps and correlation analysis."),
        ("🗄️ Data Warehouse & ETL", "Star schema, ETL audit log, table explorer, SQL console, upload your own data."),
        ("🧊 OLAP Explorer", "Multidimensional analysis with roll-up, drill-down, slice, dice and pivot."),
        ("⛏️ Data Mining Lab", "PCA, K-Means clustering, classifiers, regressors and anomaly detection."),
    ]
    for col, (t, d) in zip(c, mods):
        col.markdown(card(t, d), unsafe_allow_html=True)
    st.write("")
    c = st.columns(4)
    mods = [
        ("🌱 Yield Prediction", "Predict yield (t/ha) and yield class for any field condition, plus trend forecasts."),
        ("🧑‍🌾 Farmer Recommendations", "Best-crop ranking and agronomic advice for water, nutrients and soil pH."),
        ("🧮 Concepts Covered", "Data Warehouse, ETL, OLAP, PCA, Classification, Clustering, Anomaly Detection."),
        ("🛠️ Technology", "Python, Streamlit, SQLite, Pandas, Scikit-learn, Plotly."),
    ]
    for col, (t, d) in zip(c, mods):
        col.markdown(card(t, d), unsafe_allow_html=True)

    st.subheader("Snapshot: Average Yield by Crop")
    g = df.groupby("crop_name")["yield_t_ha"].mean().sort_values().reset_index()
    fig = px.bar(g, x="yield_t_ha", y="crop_name", orientation="h", text_auto=".2f",
                 labels={"yield_t_ha": "Average Yield (t/ha)", "crop_name": "Crop"},
                 color="yield_t_ha", color_continuous_scale="Greens", title="Average Yield by Crop (t/ha)")
    fig.update_layout(coloraxis_showscale=False)
    show(fig, 380)
