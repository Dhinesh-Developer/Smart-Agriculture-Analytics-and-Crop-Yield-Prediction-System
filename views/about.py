import pandas as pd
import streamlit as st

from src.config import PROJECT_TITLE, SUBJECT, TEAM
from src.ui import html_table, page_header


def render():
    page_header("About the Project", PROJECT_TITLE)
    st.subheader("Project Details")
    html_table(pd.DataFrame([
        ("Project Title", PROJECT_TITLE), ("Subject", "Data Warehousing and Data Mining"), ("Section / Team", "A Section - Team No. 5"),
        ("Difficulty", "Medium"), ("Core Concepts", "Predictive Analytics, Data Mining, Classification"),
        ("Description", "Integrate weather, crop and soil data to predict crop yields and recommend farming strategies."),
        ("Key Features", "Crop forecasting, anomaly detection, farmer recommendations, analytics dashboards"),
        ("Concepts Used", "Data Warehouse, ETL, PCA, Classification, Clustering, OLAP"),
        ("Expected Output", "Crop prediction reports and agricultural decision-support dashboard"),
    ], columns=["Field", "Details"]))

    st.subheader("Project Team")
    html_table(pd.DataFrame([(i + 1, r, n) for i, (r, n) in enumerate(TEAM)], columns=["S.No", "Reg. No.", "Name of the Student"]), 0)

    st.subheader("Concept-to-Module Mapping")
    html_table(pd.DataFrame([
        ("Data Warehouse", "SQLite star schema (1 fact + 4 dimensions)", "Data Warehouse & ETL"),
        ("ETL", "Extraction, cleaning, imputation, transformation, loading", "Data Warehouse & ETL"),
        ("OLAP", "Roll-up, drill-down, slice, dice, pivot via SQL", "OLAP Explorer"),
        ("PCA", "Dimensionality reduction, scree plot, loadings", "Data Mining Lab"),
        ("Clustering", "K-Means with elbow/silhouette analysis", "Data Mining Lab"),
        ("Classification", "Decision Tree, Random Forest, Logistic Regression, KNN", "Data Mining Lab"),
        ("Predictive Analytics", "Regression models & trend forecasting", "Yield Prediction"),
        ("Anomaly Detection", "Isolation Forest + z-score", "Data Mining Lab"),
        ("Decision Support", "Crop ranking and agronomic advice", "Farmer Recommendations"),
    ], columns=["Concept", "Implementation", "Module"]))

    st.subheader("Technology Stack")
    html_table(pd.DataFrame([
        ("Python 3.10+", "Core programming language"), ("Streamlit", "Frontend + backend web application framework"),
        ("SQLite", "Embedded relational database for the data warehouse"), ("Pandas / NumPy", "Data manipulation and numerical computing"),
        ("Scikit-learn", "Machine-learning and data-mining algorithms"), ("Plotly", "Interactive charts and dashboards"),
    ], columns=["Tool", "Purpose"]))
    st.caption(SUBJECT)
