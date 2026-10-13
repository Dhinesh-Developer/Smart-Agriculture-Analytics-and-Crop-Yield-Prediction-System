"""Data-warehouse access layer: SQL helpers and OLAP operations (roll-up, drill-down, slice, dice, pivot)."""
import sqlite3

import pandas as pd

from .config import DB_PATH

DIMENSIONS = {"Year": "year", "Season": "season", "Zone": "zone", "District": "district",
              "Crop": "crop_name", "Crop Type": "crop_type", "Soil Type": "soil_type"}
MEASURES = {
    "Average Yield (t/ha)": "ROUND(AVG(yield_t_ha), 3)",
    "Total Production (t)": "ROUND(SUM(production_t), 1)",
    "Total Area (ha)": "ROUND(SUM(area_ha), 1)",
    "Average Rainfall (mm)": "ROUND(AVG(rainfall_mm), 1)",
    "Record Count": "COUNT(*)",
}
TABLES = ["fact_crop_yield", "dim_date", "dim_region", "dim_crop", "dim_soil", "etl_log"]


def connect(readonly=False):
    if readonly:
        return sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True)
    return sqlite3.connect(DB_PATH)


def query(sql: str, params=()) -> pd.DataFrame:
    con = connect()
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def safe_select(sql: str) -> pd.DataFrame:
    """Run a user-typed query on a read-only connection (SELECT only)."""
    cleaned = sql.strip().rstrip(";")
    if not cleaned.lower().startswith(("select", "with")) or ";" in cleaned:
        raise ValueError("Only a single SELECT statement is allowed.")
    con = connect(readonly=True)
    try:
        return pd.read_sql_query(cleaned, con)
    finally:
        con.close()


def load_analysis_df() -> pd.DataFrame:
    return query("SELECT * FROM vw_yield_analysis")


def table_counts() -> pd.DataFrame:
    rows = [(t, int(query(f"SELECT COUNT(*) AS n FROM {t}")["n"][0])) for t in TABLES]
    return pd.DataFrame(rows, columns=["Table", "Rows"])


def etl_log() -> pd.DataFrame:
    return query("SELECT step_no AS Step, stage AS Stage, description AS Description, "
                 "rows_in AS 'Rows In', rows_out AS 'Rows Out', run_time AS 'Run Time' FROM etl_log ORDER BY step_no")


def build_olap_sql(dims, measure_label, filters=None, rollup=False):
    """Build an OLAP query. dims = list of column names, filters = {column: [values]}."""
    measure = MEASURES[measure_label]
    filters = {k: v for k, v in (filters or {}).items() if v}
    where, params = "", []
    if filters:
        parts = []
        for col, vals in filters.items():
            parts.append(f"{col} IN ({','.join('?' * len(vals))})")
            params.extend(vals)
        where = "WHERE " + " AND ".join(parts)

    def one_level(level):
        cols = dims[:level]
        sel = [f"CAST({d} AS TEXT) AS {d}" if d in cols else f"'ALL' AS {d}" for d in dims]
        group = f"GROUP BY {', '.join(cols)}" if cols else ""
        return f"SELECT {', '.join(sel)}, {measure} AS value FROM vw_yield_analysis {where} {group}".strip()

    if rollup and dims:
        sql = "\nUNION ALL\n".join(one_level(i) for i in range(len(dims), -1, -1))
        params = params * (len(dims) + 1)
        sql += f"\nORDER BY {', '.join(str(i + 1) for i in range(len(dims)))}"
    elif dims:
        sql = one_level(len(dims)) + f"\nORDER BY {', '.join(dims)}"
    else:
        sql = f"SELECT {measure} AS value FROM vw_yield_analysis {where}"
    return sql, params


def inline_sql(sql, params):
    """Render the SQL with parameters substituted - for display only."""
    for p in params:
        sql = sql.replace("?", repr(p), 1)
    return sql


def run_olap(dims, measure_label, filters=None, rollup=False):
    sql, params = build_olap_sql(dims, measure_label, filters, rollup)
    return query(sql, params), inline_sql(sql, params)
