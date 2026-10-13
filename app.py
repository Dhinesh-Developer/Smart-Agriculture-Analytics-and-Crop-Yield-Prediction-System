"""Smart Agriculture Analytics and Crop Yield Prediction System - Streamlit entry point.

Run with:  streamlit run app.py
"""
import streamlit as st

from src import mining, warehouse
from src.config import PROJECT_TITLE
from src.etl import ensure_warehouse
from src.ui import apply_theme, footer

st.set_page_config(page_title="Smart Agriculture Analytics", page_icon="🌾", layout="wide",
                   initial_sidebar_state="expanded")
apply_theme()


@st.cache_resource(show_spinner="Building data warehouse (first run only)...")
def init_warehouse():
    return ensure_warehouse()


@st.cache_data(show_spinner=False)
def get_data():
    return warehouse.load_analysis_df()


@st.cache_resource(show_spinner="Training data-mining models...")
def get_models():
    df = get_data()
    return {"reg": mining.train_regression(df), "cls": mining.train_classification(df)}


def reset_caches():
    st.cache_data.clear()
    st.cache_resource.clear()


init_warehouse()
df = get_data()

from views import about, dashboard, home, mining_view, olap, predict, recommend, warehouse_view  # noqa: E402

PAGES = {
    "🏠  Home": lambda: home.render(df),
    "📊  Analytics Dashboard": lambda: dashboard.render(df),
    "🗄️  Data Warehouse & ETL": lambda: warehouse_view.render(df, reset_caches),
    "🧊  OLAP Explorer": lambda: olap.render(df),
    "⛏️  Data Mining Lab": lambda: mining_view.render(df, get_models()),
    "🌱  Yield Prediction": lambda: predict.render(df, get_models()),
    "🧑‍🌾  Farmer Recommendations": lambda: recommend.render(df, get_models()),
    "ℹ️  About Project": lambda: about.render(),
}

with st.sidebar:
    st.markdown("### 🌾 Smart Agri Analytics")
    st.caption(PROJECT_TITLE)
    choice = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption(f"Warehouse records: **{len(df):,}**")
    st.caption(f"Period: **{df['year'].min()} - {df['year'].max()}**")
    st.caption("Database: SQLite (star schema)")

PAGES[choice]()
footer()
