import plotly.express as px
import streamlit as st

from src import warehouse
from src.ui import html_table, page_header, show, show_df


def render(df):
    page_header("OLAP Explorer", "Multidimensional analysis: roll-up, drill-down, slice, dice and pivot on the star schema")
    op = st.radio("OLAP operation", ["Roll-up", "Drill-down", "Slice", "Dice", "Pivot"], horizontal=True)
    measure = st.selectbox("Measure", list(warehouse.MEASURES.keys()))
    ylab = measure

    if op == "Roll-up":
        st.caption("Aggregate to higher levels of the hierarchy; 'ALL' rows are sub-totals (like GROUP BY ROLLUP).")
        hier = st.selectbox("Hierarchy", ["Zone → District", "Year → Season", "Crop Type → Crop"])
        dims = {"Zone → District": ["zone", "district"], "Year → Season": ["year", "season"],
                "Crop Type → Crop": ["crop_type", "crop_name"]}[hier]
        res, sql = warehouse.run_olap(dims, measure, rollup=True)
        top = res[res[dims[1]] == "ALL"]
        top = top[top[dims[0]] != "ALL"]
        fig = px.bar(top, x=dims[0], y="value", text_auto=True, color=dims[0], labels={"value": ylab},
                     title=f"Rolled-up {measure} by {dims[0].replace('_', ' ').title()}")
        fig.update_layout(showlegend=False)
        show(fig)
        show_df(res, 340)

    elif op == "Drill-down":
        st.caption("Move from a summary level to finer detail within the hierarchy.")
        level = st.select_slider("Detail level", ["Zone", "Zone → District", "Zone → District → Crop"])
        dims = {"Zone": ["zone"], "Zone → District": ["zone", "district"],
                "Zone → District → Crop": ["zone", "district", "crop_name"]}[level]
        res, sql = warehouse.run_olap(dims, measure)
        fig = px.sunburst(res, path=dims, values="value", title=f"{measure}: {level}") \
            if measure in ("Total Production (t)", "Total Area (ha)", "Record Count") else \
            px.bar(res, x=dims[-1], y="value", color=dims[0], labels={"value": ylab}, title=f"{measure}: {level}")
        show(fig, 460)
        show_df(res, 320)

    elif op == "Slice":
        st.caption("Fix ONE dimension to a single value and analyse the resulting sub-cube.")
        c1, c2, c3 = st.columns(3)
        sdim = c1.selectbox("Slice dimension", ["Season", "Year", "Zone", "Crop", "Soil Type"])
        col = warehouse.DIMENSIONS[sdim]
        val = c2.selectbox("Value", sorted(df[col].unique().tolist()))
        gdim = c3.selectbox("Group result by", [d for d in warehouse.DIMENSIONS if d != sdim], index=2)
        res, sql = warehouse.run_olap([warehouse.DIMENSIONS[gdim]], measure, {col: [val]})
        fig = px.bar(res, x=warehouse.DIMENSIONS[gdim], y="value", text_auto=True, color=warehouse.DIMENSIONS[gdim],
                     labels={"value": ylab, warehouse.DIMENSIONS[gdim]: gdim},
                     title=f"{measure} by {gdim} (slice: {sdim} = {val})")
        fig.update_layout(showlegend=False)
        show(fig)
        show_df(res, 300)

    elif op == "Dice":
        st.caption("Select value ranges on SEVERAL dimensions simultaneously to carve out a sub-cube.")
        c1, c2, c3 = st.columns(3)
        zones = c1.multiselect("Zone", sorted(df["zone"].unique()), default=sorted(df["zone"].unique())[:3])
        crops = c2.multiselect("Crop", sorted(df["crop_name"].unique()), default=sorted(df["crop_name"].unique())[:3])
        seasons = c3.multiselect("Season", sorted(df["season"].unique()), default=sorted(df["season"].unique()))
        gdim = st.selectbox("Group result by", ["Year", "Crop", "Zone", "District", "Season"])
        gcol = warehouse.DIMENSIONS[gdim]
        res, sql = warehouse.run_olap([gcol, "crop_name"] if gcol != "crop_name" else ["crop_name"], measure,
                                      {"zone": zones, "crop_name": crops, "season": seasons})
        if res.empty:
            st.warning("Empty sub-cube - widen the selection.")
        else:
            xcol = gcol
            fig = px.bar(res, x=xcol, y="value", color="crop_name", barmode="group",
                         labels={"value": ylab, "crop_name": "Crop", xcol: gdim}, title=f"Diced cube: {measure} by {gdim} and Crop")
            show(fig)
            show_df(res, 300)

    else:
        st.caption("Rotate the cube: choose which dimensions appear on rows and columns.")
        c1, c2 = st.columns(2)
        rdim = c1.selectbox("Rows", list(warehouse.DIMENSIONS.keys()), index=3)
        cdim = c2.selectbox("Columns", [d for d in warehouse.DIMENSIONS if d != rdim], index=3)
        res, sql = warehouse.run_olap([warehouse.DIMENSIONS[rdim], warehouse.DIMENSIONS[cdim]], measure)
        pv = res.pivot_table(index=warehouse.DIMENSIONS[rdim], columns=warehouse.DIMENSIONS[cdim], values="value", aggfunc="sum")
        fig = px.imshow(pv, color_continuous_scale="YlGn", aspect="auto", text_auto=".2f" if measure.startswith("Average") else ".0f",
                        labels=dict(x=cdim, y=rdim, color=ylab), title=f"Pivot: {measure} ({rdim} × {cdim})")
        show(fig, 480)
        out = pv.reset_index()
        out.columns = [str(c) for c in out.columns]
        html_table(out.head(15), 2) if out.shape[1] <= 8 else show_df(out, 340)

    with st.expander("🧾 SQL executed on the warehouse"):
        st.code(sql, language="sql")
