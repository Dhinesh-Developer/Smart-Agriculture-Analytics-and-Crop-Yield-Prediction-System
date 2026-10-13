import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import LABELS, NUMERIC_FEATURES
from src.ui import GREEN_SCALE, kpi_row, page_header, show, show_df


def render(df):
    page_header("Analytics Dashboard", "Interactive overview of crop production, yield and climate indicators")

    with st.expander("🔎 Filters", expanded=True):
        c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
        yr = c1.slider("Year range", int(df["year"].min()), int(df["year"].max()),
                       (int(df["year"].min()), int(df["year"].max())))
        seasons = c2.multiselect("Season", sorted(df["season"].unique()), default=sorted(df["season"].unique()))
        zones = c3.multiselect("Zone", sorted(df["zone"].unique()), default=sorted(df["zone"].unique()))
        crops = c4.multiselect("Crop", sorted(df["crop_name"].unique()), default=sorted(df["crop_name"].unique()))
    f = df[df["year"].between(*yr) & df["season"].isin(seasons) & df["zone"].isin(zones) & df["crop_name"].isin(crops)]
    if f.empty:
        st.warning("No records match the selected filters.")
        return

    best = f.groupby("crop_name")["yield_t_ha"].mean().idxmax()
    kpi_row([
        ("Records", f"{len(f):,}", "Filtered fact rows"),
        ("Avg Yield", f"{f['yield_t_ha'].mean():.2f} t/ha", f"Best crop: {best}"),
        ("Production", f"{f['production_t'].sum() / 1e6:.2f} M t", "Total output"),
        ("Avg Rainfall", f"{f['rainfall_mm'].mean():.0f} mm", "Per crop season"),
        ("Cultivated Area", f"{f['area_ha'].sum() / 1e6:.2f} M ha", "Total area"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        t = f.groupby(["year", "season"])["yield_t_ha"].mean().reset_index()
        fig = px.line(t, x="year", y="yield_t_ha", color="season", markers=True,
                      labels={"yield_t_ha": "Avg Yield (t/ha)", "year": "Year", "season": "Season"},
                      title="Yield Trend by Season")
        fig.update_xaxes(dtick=1)
        show(fig)
    with c2:
        b = f.groupby("crop_name")["yield_t_ha"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(b, x="crop_name", y="yield_t_ha", text_auto=".2f", color="crop_name",
                     labels={"yield_t_ha": "Avg Yield (t/ha)", "crop_name": "Crop"}, title="Average Yield by Crop")
        fig.update_layout(showlegend=False)
        show(fig)

    c1, c2 = st.columns(2)
    with c1:
        d = f.groupby("district")["production_t"].sum().sort_values().reset_index()
        fig = px.bar(d, x="production_t", y="district", orientation="h", color="production_t",
                     color_continuous_scale="Greens", labels={"production_t": "Production (t)", "district": "District"},
                     title="Total Production by District")
        fig.update_layout(coloraxis_showscale=False)
        show(fig, 440)
    with c2:
        p = f.groupby("crop_type")["production_t"].sum().reset_index()
        fig = px.pie(p, names="crop_type", values="production_t", hole=0.5, title="Production Share by Crop Category")
        fig.update_traces(textinfo="percent+label")
        show(fig, 440)

    c1, c2 = st.columns(2)
    with c1:
        h = f.pivot_table(index="crop_name", columns="year", values="yield_t_ha", aggfunc="mean")
        fig = go.Figure(go.Heatmap(z=h.values, x=h.columns.astype(str), y=h.index, colorscale="YlGn",
                                   colorbar=dict(title="t/ha"), hovertemplate="Crop %{y}<br>Year %{x}<br>%{z:.2f} t/ha<extra></extra>"))
        fig.update_layout(title="Heat-map: Average Yield (Crop × Year)")
        show(fig, 420)
    with c2:
        corr = f[NUMERIC_FEATURES + ["yield_t_ha"]].corr().round(2)
        names = [LABELS[c].split(" (")[0] for c in corr.columns]
        fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1, colorscale="RdYlGn",
                                   text=corr.values, texttemplate="%{text}", textfont=dict(size=10)))
        fig.update_layout(title="Correlation Matrix of Climate, Soil & Yield")
        show(fig, 420)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(f, x="rainfall_mm", y="yield_t_ha", color="crop_name", opacity=0.75,
                         labels={"rainfall_mm": "Rainfall (mm)", "yield_t_ha": "Yield (t/ha)", "crop_name": "Crop"},
                         title="Rainfall vs Yield")
        show(fig, 420)
    with c2:
        fig = px.box(f, x="crop_name", y="yield_t_ha", color="crop_name", points="outliers",
                     labels={"yield_t_ha": "Yield (t/ha)", "crop_name": "Crop"}, title="Yield Distribution by Crop")
        fig.update_layout(showlegend=False)
        show(fig, 420)

    st.subheader("Zone-wise Summary Table")
    z = f.groupby("zone").agg(Records=("fact_id", "count"), Avg_Yield=("yield_t_ha", "mean"),
                              Production=("production_t", "sum"), Rainfall=("rainfall_mm", "mean"),
                              Temperature=("temperature_c", "mean")).round(2).reset_index()
    z.columns = ["Zone", "Records", "Avg Yield (t/ha)", "Production (t)", "Rainfall (mm)", "Temperature (°C)"]
    show_df(z, 260)

    with st.expander("View filtered records"):
        show_df(f, 360)
        st.download_button("⬇ Download filtered data (CSV)", f.to_csv(index=False).encode(), "filtered_agri_data.csv", "text/csv")
