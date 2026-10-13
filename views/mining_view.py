import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import mining
from src.config import CLASS_ORDER, LABELS
from src.ui import html_table, kpi_row, page_header, show, show_df


@st.cache_data(show_spinner="Running PCA...")
def _pca(df):
    return mining.run_pca(df)


@st.cache_data(show_spinner="Running K-Means...")
def _cluster(df, k):
    return mining.run_clustering(df, k)


@st.cache_data(show_spinner="Detecting anomalies...")
def _anomaly(df, c):
    return mining.detect_anomalies(df, c)


def render(df, models):
    page_header("Data Mining Lab", "PCA, clustering, classification, regression and anomaly detection on warehouse data")
    t1, t2, t3, t4, t5 = st.tabs(["📉 PCA", "🧩 Clustering", "🎯 Classification", "📈 Regression", "🚨 Anomaly Detection"])

    # ---------------------------------------------------------------- PCA
    with t1:
        p = _pca(df)
        st.markdown(f"Principal Component Analysis compresses the 11 numeric variables. **{p['n95']} components** retain 95% of the variance.")
        c1, c2 = st.columns(2)
        with c1:
            v = p["variance"]
            fig = go.Figure()
            fig.add_bar(x=v["Component"], y=v["Explained Variance (%)"], name="Explained variance", marker_color="#2E7D32")
            fig.add_scatter(x=v["Component"], y=v["Cumulative (%)"], name="Cumulative", mode="lines+markers", line=dict(color="#C62828"))
            fig.update_layout(title="Scree Plot", yaxis_title="Variance (%)")
            show(fig, 400)
        with c2:
            color = st.radio("Colour points by", ["Crop", "Yield Class"], horizontal=True)
            fig = px.scatter(p["projection"], x="PC1", y="PC2", color=color, opacity=0.75, title="PCA Projection (PC1 vs PC2)",
                             category_orders={"Yield Class": CLASS_ORDER})
            show(fig, 400)
        st.subheader("Component Loadings")
        ld = p["loadings"].round(3).reset_index().rename(columns={"index": "Variable"})
        ld["Variable"] = ld["Variable"].map(lambda c: LABELS.get(c, c))
        html_table(ld, 3)

    # ---------------------------------------------------------------- clustering
    with t2:
        k = st.slider("Number of clusters (k)", 2, 8, 4)
        c = _cluster(df, k)
        c1, c2 = st.columns(2)
        with c1:
            e = c["elbow"]
            fig = go.Figure()
            fig.add_scatter(x=e["k"], y=e["Inertia"], mode="lines+markers", name="Inertia", line=dict(color="#2E7D32"))
            fig.add_scatter(x=e["k"], y=e["Silhouette"], mode="lines+markers", name="Silhouette", yaxis="y2", line=dict(color="#EF6C00"))
            fig.update_layout(title="Elbow & Silhouette Analysis", xaxis_title="k", yaxis_title="Inertia",
                              yaxis2=dict(title="Silhouette", overlaying="y", side="right"))
            show(fig, 400)
        with c2:
            fig = px.scatter(c["coords"], x="PC1", y="PC2", color="Cluster", opacity=0.8, title=f"K-Means Clusters (k={k}) in PCA Space",
                             category_orders={"Cluster": sorted(c["coords"]["Cluster"].unique())})
            show(fig, 400)
        st.subheader("Cluster Profiles")
        html_table(c["profile"], 2)
        st.caption("Clusters are ranked by average yield: Cluster 1 = highest-yielding agro-climatic group.")

    # ---------------------------------------------------------------- classification
    with t3:
        cl = models["cls"]
        st.markdown("**Task:** classify each crop record into a crop-relative yield class (**Low / Medium / High**, tertiles within each crop). "
                    "Hold-out test set = 20%.")
        best = cl["results"].iloc[0]
        kpi_row([("Best Model", cl["best_name"], "Highest accuracy"), ("Accuracy", f"{best['Accuracy']:.1%}", "Test set"),
                 ("F1-Score", f"{best['F1-Score']:.3f}", "Weighted"), ("Precision", f"{best['Precision']:.3f}", "Weighted")])
        c1, c2 = st.columns(2)
        with c1:
            r = cl["results"].melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig = px.bar(r, x="Model", y="Score", color="Metric", barmode="group", title="Classifier Comparison")
            fig.update_yaxes(range=[0, 1])
            show(fig, 400)
        with c2:
            cm = cl["confusion"]
            fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, colorscale="Greens", text=cm.values,
                                       texttemplate="%{text}", textfont=dict(size=18)))
            fig.update_layout(title=f"Confusion Matrix - {cl['best_name']}", yaxis_autorange="reversed")
            show(fig, 400)
        html_table(cl["results"], 3)

    # ---------------------------------------------------------------- regression
    with t4:
        rg = models["reg"]
        best = rg["results"].iloc[0]
        st.markdown("**Task:** predict continuous yield (t/ha) from climate, soil, management and crop variables.")
        kpi_row([("Best Model", rg["best_name"].replace(" Regressor", ""), "Highest R²"), ("R² Score", f"{best['R²']:.3f}", "Test set"),
                 ("MAE", f"{best['MAE']:.3f} t/ha", "Mean abs. error"), ("RMSE", f"{best['RMSE']:.3f} t/ha", "Root mean sq. error")])
        c1, c2 = st.columns(2)
        with c1:
            t = rg["test"]
            mx = float(max(t["Actual"].max(), t["Predicted"].max()))
            fig = px.scatter(t, x="Actual", y="Predicted", opacity=0.7, title="Actual vs Predicted Yield (test set)",
                             labels={"Actual": "Actual (t/ha)", "Predicted": "Predicted (t/ha)"})
            fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(color="#C62828", dash="dash"))
            show(fig, 420)
        with c2:
            imp = rg["importance"]
            if imp is not None:
                top = imp.head(12).iloc[::-1].copy()
                top["Feature"] = top["Feature"].map(lambda f: LABELS.get(f, f.replace("_", " ").title()))
                fig = px.bar(top, x="Importance", y="Feature", orientation="h", title="Top Feature Importances",
                             color="Importance", color_continuous_scale="Greens")
                fig.update_layout(coloraxis_showscale=False)
                show(fig, 420)
        html_table(rg["results"], 3)

    # ---------------------------------------------------------------- anomaly
    with t5:
        cont = st.slider("Expected anomaly rate", 0.01, 0.10, 0.04, 0.01)
        a = _anomaly(df, cont)
        an = a[a["is_anomaly"]]
        kpi_row([("Anomalies Found", len(an), f"{len(an) / len(a):.1%} of records"),
                 ("Avg Yield (anomalies)", f"{an['yield_t_ha'].mean():.2f} t/ha", f"vs {a['yield_t_ha'].mean():.2f} overall"),
                 ("Most Affected Crop", an["crop_name"].mode()[0], "Highest count"),
                 ("Most Affected District", an["district"].mode()[0], "Highest count")])
        st.caption("Method: Isolation Forest (multivariate) combined with a per-crop yield z-score threshold.")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(a, x="rainfall_mm", y="yield_zscore", color="is_anomaly", opacity=0.75,
                             color_discrete_map={True: "#C62828", False: "#9E9E9E"},
                             labels={"rainfall_mm": "Rainfall (mm)", "yield_zscore": "Yield z-score (per crop)", "is_anomaly": "Anomaly"},
                             title="Yield z-score vs Rainfall")
            show(fig, 400)
        with c2:
            by = an.groupby("year").size().reset_index(name="Anomalies")
            fig = px.bar(by, x="year", y="Anomalies", title="Anomalies per Year", text_auto=True, color_discrete_sequence=["#C62828"])
            fig.update_xaxes(dtick=1)
            show(fig, 400)
        st.subheader("Flagged Records")
        cols = ["year", "season", "district", "crop_name", "rainfall_mm", "yield_t_ha", "yield_zscore", "anomaly_score", "reason"]
        show_df(an[cols].sort_values("anomaly_score", ascending=False).round(2), 340)
