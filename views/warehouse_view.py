import pandas as pd
import streamlit as st

from src import etl, warehouse
from src.config import REQUIRED_COLUMNS
from src.data_generator import generate_raw_data
from src.ui import html_table, page_header, show_df

STAR = '''
digraph G {
  rankdir=LR; bgcolor="white"; nodesep=0.5; ranksep=1.1;
  node [shape=plaintext, fontname="Times New Roman", fontsize=13];
  edge [color="#2E7D32", penwidth=1.6, arrowhead=none];
  fact [label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5" COLOR="#1B5E20">
    <TR><TD BGCOLOR="#2E7D32"><FONT COLOR="white"><B>fact_crop_yield</B></FONT></TD></TR>
    <TR><TD ALIGN="LEFT" BGCOLOR="#F1F8E9"><B>PK</B> fact_id</TD></TR>
    <TR><TD ALIGN="LEFT">FK date_id, region_id,<BR ALIGN="LEFT"/>FK crop_id, soil_id</TD></TR>
    <TR><TD ALIGN="LEFT">area_ha, rainfall_mm, temperature_c,<BR ALIGN="LEFT"/>humidity_pct, nitrogen, phosphorus,<BR ALIGN="LEFT"/>potassium, soil_ph, fertilizer_kg_ha,<BR ALIGN="LEFT"/>irrigation_pct, pesticide_kg_ha,<BR ALIGN="LEFT"/>yield_t_ha, production_t, yield_class</TD></TR>
  </TABLE>>];
  d [label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5" COLOR="#1565C0">
    <TR><TD BGCOLOR="#1565C0"><FONT COLOR="white"><B>dim_date</B></FONT></TD></TR>
    <TR><TD ALIGN="LEFT">PK date_id<BR ALIGN="LEFT"/>year, season</TD></TR></TABLE>>];
  r [label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5" COLOR="#1565C0">
    <TR><TD BGCOLOR="#1565C0"><FONT COLOR="white"><B>dim_region</B></FONT></TD></TR>
    <TR><TD ALIGN="LEFT">PK region_id<BR ALIGN="LEFT"/>state, district, zone</TD></TR></TABLE>>];
  c [label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5" COLOR="#1565C0">
    <TR><TD BGCOLOR="#1565C0"><FONT COLOR="white"><B>dim_crop</B></FONT></TD></TR>
    <TR><TD ALIGN="LEFT">PK crop_id<BR ALIGN="LEFT"/>crop_name, crop_type</TD></TR></TABLE>>];
  s [label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5" COLOR="#1565C0">
    <TR><TD BGCOLOR="#1565C0"><FONT COLOR="white"><B>dim_soil</B></FONT></TD></TR>
    <TR><TD ALIGN="LEFT">PK soil_id<BR ALIGN="LEFT"/>soil_type</TD></TR></TABLE>>];
  d -> fact; r -> fact; c -> fact; s -> fact;
}
'''


def render(df, reset_caches):
    page_header("Data Warehouse & ETL", "Star-schema design, ETL audit trail, table explorer and SQL console")
    t1, t2, t3, t4 = st.tabs(["⭐ Star Schema", "🔄 ETL Pipeline", "📋 Table Explorer", "💻 SQL Console"])

    with t1:
        st.markdown("The warehouse follows a **star schema**: one central fact table of crop-yield measurements "
                    "surrounded by four dimension tables (Date, Region, Crop, Soil). Hierarchies supporting OLAP: "
                    "**Season → Year**, **District → Zone → State**, **Crop → Crop Type**.")
        st.graphviz_chart(STAR, use_container_width=True)
        st.subheader("Warehouse Table Sizes")
        html_table(warehouse.table_counts(), 0)

    with t2:
        st.markdown("**Extract → Transform → Load** pipeline executed when the warehouse was built:")
        html_table(warehouse.etl_log(), 0)
        st.info("Transformations: schema validation, text standardisation, duplicate removal, median imputation, "
                "range validation, derived measures (production, yield class).")
        st.subheader("Rebuild the Warehouse")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Option A - Regenerate sample data**")
            seed = st.number_input("Random seed", 1, 9999, 42)
            if st.button("Regenerate & re-run ETL"):
                with st.spinner("Running ETL..."):
                    etl.run_pipeline(generate_raw_data(int(seed)))
                    reset_caches()
                st.success("Warehouse rebuilt. Reloading...")
                st.rerun()
        with c2:
            st.markdown("**Option B - Upload your own CSV**")
            st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))
            up = st.file_uploader("Raw agriculture CSV", type="csv")
            if up is not None and st.button("Load uploaded file through ETL"):
                try:
                    with st.spinner("Running ETL..."):
                        etl.run_pipeline(pd.read_csv(up))
                        reset_caches()
                    st.success("Uploaded data loaded into the warehouse. Reloading...")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"ETL failed: {e}")

    with t3:
        table = st.selectbox("Choose a table / view", warehouse.TABLES + ["vw_yield_analysis"])
        data = warehouse.query(f"SELECT * FROM {table} LIMIT 500")
        st.caption(f"Showing up to 500 rows of `{table}`")
        show_df(data, 420)

    with t4:
        st.caption("Read-only console. Only a single SELECT statement is permitted.")
        default = ("SELECT zone, crop_name, ROUND(AVG(yield_t_ha),2) AS avg_yield, COUNT(*) AS n\n"
                   "FROM vw_yield_analysis\nGROUP BY zone, crop_name\nORDER BY avg_yield DESC\nLIMIT 15")
        sql = st.text_area("SQL query", default, height=140)
        if st.button("Run query"):
            try:
                res = warehouse.safe_select(sql)
                st.success(f"{len(res)} rows returned")
                show_df(res, 360)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
