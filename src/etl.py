"""ETL pipeline: Extract raw CSV -> Transform (clean, enrich) -> Load into SQLite star schema."""
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from .config import CLASS_ORDER, DB_PATH, RAW_CSV, REQUIRED_COLUMNS
from .data_generator import generate_raw_data

SCHEMA_SQL = """
DROP VIEW  IF EXISTS vw_yield_analysis;
DROP TABLE IF EXISTS fact_crop_yield;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_region;
DROP TABLE IF EXISTS dim_crop;
DROP TABLE IF EXISTS dim_soil;
DROP TABLE IF EXISTS etl_log;

CREATE TABLE dim_date   (date_id INTEGER PRIMARY KEY, year INTEGER NOT NULL, season TEXT NOT NULL);
CREATE TABLE dim_region (region_id INTEGER PRIMARY KEY, state TEXT, district TEXT, zone TEXT);
CREATE TABLE dim_crop   (crop_id INTEGER PRIMARY KEY, crop_name TEXT, crop_type TEXT);
CREATE TABLE dim_soil   (soil_id INTEGER PRIMARY KEY, soil_type TEXT);

CREATE TABLE fact_crop_yield (
    fact_id INTEGER PRIMARY KEY,
    date_id INTEGER NOT NULL REFERENCES dim_date(date_id),
    region_id INTEGER NOT NULL REFERENCES dim_region(region_id),
    crop_id INTEGER NOT NULL REFERENCES dim_crop(crop_id),
    soil_id INTEGER NOT NULL REFERENCES dim_soil(soil_id),
    area_ha REAL, rainfall_mm REAL, temperature_c REAL, humidity_pct REAL,
    nitrogen REAL, phosphorus REAL, potassium REAL, soil_ph REAL,
    fertilizer_kg_ha REAL, irrigation_pct REAL, pesticide_kg_ha REAL,
    yield_t_ha REAL, production_t REAL, yield_class TEXT
);
CREATE INDEX idx_fact_date   ON fact_crop_yield(date_id);
CREATE INDEX idx_fact_region ON fact_crop_yield(region_id);
CREATE INDEX idx_fact_crop   ON fact_crop_yield(crop_id);

CREATE TABLE etl_log (step_no INTEGER, stage TEXT, description TEXT, rows_in INTEGER, rows_out INTEGER, run_time TEXT);

CREATE VIEW vw_yield_analysis AS
SELECT f.fact_id, d.year, d.season, r.state, r.district, r.zone, c.crop_name, c.crop_type, s.soil_type,
       f.area_ha, f.rainfall_mm, f.temperature_c, f.humidity_pct, f.nitrogen, f.phosphorus, f.potassium,
       f.soil_ph, f.fertilizer_kg_ha, f.irrigation_pct, f.pesticide_kg_ha, f.yield_t_ha, f.production_t, f.yield_class
FROM fact_crop_yield f
JOIN dim_date d   ON f.date_id = d.date_id
JOIN dim_region r ON f.region_id = r.region_id
JOIN dim_crop c   ON f.crop_id = c.crop_id
JOIN dim_soil s   ON f.soil_id = s.soil_id;
"""

NUM_COLS = ["area_ha", "rainfall_mm", "temperature_c", "humidity_pct", "nitrogen", "phosphorus",
            "potassium", "soil_ph", "fertilizer_kg_ha", "irrigation_pct", "pesticide_kg_ha", "yield_t_ha"]
VALID_RANGE = {"soil_ph": (3.5, 10), "humidity_pct": (0, 100), "irrigation_pct": (0, 100),
               "temperature_c": (5, 50), "rainfall_mm": (0, 3000)}


def extract(path=RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        generate_raw_data()
    return pd.read_csv(path)


def transform(raw: pd.DataFrame):
    """Return (clean_dataframe, log) where log is a list of (description, rows_in, rows_out)."""
    log = []
    df = raw.copy()
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded data is missing required columns: {missing_cols}")
    df = df[REQUIRED_COLUMNS]
    log.append(("Select required columns / validate schema", len(raw), len(df)))

    n = len(df)
    for col in ["season", "state", "district", "zone", "crop_name", "crop_type", "soil_type"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    log.append(("Standardise text (trim whitespace, Title Case)", n, len(df)))

    n = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "record_id"]).reset_index(drop=True)
    log.append(("Remove duplicate records", n, len(df)))

    n = len(df)
    n_missing = int(df[NUM_COLS].isna().sum().sum())
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df.groupby(["crop_name", "season"])[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())
    log.append((f"Impute {n_missing} missing values (median by crop & season)", n, len(df)))

    n = len(df)
    for col, (lo, hi) in VALID_RANGE.items():
        df[col] = df[col].clip(lo, hi)
    df = df[(df["yield_t_ha"] > 0) & (df["area_ha"] > 0)].reset_index(drop=True)
    log.append(("Enforce valid ranges, drop non-positive yield/area", n, len(df)))

    n = len(df)
    df["year"] = df["year"].astype(int)
    df["production_t"] = (df["area_ha"] * df["yield_t_ha"]).round(2)
    df["yield_t_ha"] = df["yield_t_ha"].round(3)
    df["yield_class"] = df.groupby("crop_name")["yield_t_ha"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=CLASS_ORDER).astype(str))
    log.append(("Derive production_t and crop-relative yield_class (tertiles)", n, len(df)))
    return df, log


def load(df: pd.DataFrame, log, db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA_SQL)

    dim_date = df[["year", "season"]].drop_duplicates().sort_values(["year", "season"]).reset_index(drop=True)
    dim_date.insert(0, "date_id", dim_date.index + 1)
    dim_region = df[["state", "district", "zone"]].drop_duplicates().sort_values("district").reset_index(drop=True)
    dim_region.insert(0, "region_id", dim_region.index + 1)
    dim_crop = df[["crop_name", "crop_type"]].drop_duplicates().sort_values("crop_name").reset_index(drop=True)
    dim_crop.insert(0, "crop_id", dim_crop.index + 1)
    dim_soil = df[["soil_type"]].drop_duplicates().sort_values("soil_type").reset_index(drop=True)
    dim_soil.insert(0, "soil_id", dim_soil.index + 1)

    fact = (df.merge(dim_date, on=["year", "season"]).merge(dim_region, on=["state", "district", "zone"])
              .merge(dim_crop, on=["crop_name", "crop_type"]).merge(dim_soil, on="soil_type"))
    fact = fact.sort_values("record_id").reset_index(drop=True)
    fact.insert(0, "fact_id", fact.index + 1)
    fact_cols = ["fact_id", "date_id", "region_id", "crop_id", "soil_id", "area_ha", "rainfall_mm", "temperature_c",
                 "humidity_pct", "nitrogen", "phosphorus", "potassium", "soil_ph", "fertilizer_kg_ha",
                 "irrigation_pct", "pesticide_kg_ha", "yield_t_ha", "production_t", "yield_class"]

    for name, frame in [("dim_date", dim_date), ("dim_region", dim_region), ("dim_crop", dim_crop),
                        ("dim_soil", dim_soil), ("fact_crop_yield", fact[fact_cols])]:
        frame.to_sql(name, con, if_exists="append", index=False)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    steps = [("EXTRACT", "Read raw CSV into DataFrame", log[0][1], log[0][1])]
    steps += [("TRANSFORM", d, a, b) for d, a, b in log]
    steps.append(("LOAD", "Insert into star schema (1 fact + 4 dimensions)", len(df), len(fact)))
    con.executemany("INSERT INTO etl_log VALUES (?,?,?,?,?,?)",
                    [(i + 1, s, d, int(a), int(b), now) for i, (s, d, a, b) in enumerate(steps)])
    con.commit()
    con.close()


def run_pipeline(raw: pd.DataFrame = None, db_path=DB_PATH):
    if raw is None:
        raw = extract()
    clean, log = transform(raw)
    load(clean, log, db_path)
    return clean


def ensure_warehouse():
    """Build the warehouse on first run (e.g. fresh Streamlit Cloud container)."""
    if not DB_PATH.exists():
        if not RAW_CSV.exists():
            generate_raw_data()
        run_pipeline()
    return True
