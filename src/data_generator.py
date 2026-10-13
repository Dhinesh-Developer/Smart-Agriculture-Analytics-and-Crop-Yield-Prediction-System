"""Synthetic agricultural data generator (Tamil Nadu districts, 2014-2023).

The data mimics what would be collected from weather stations, soil health cards and
agriculture department yield records. It is generated with agronomic relationships
(rainfall, temperature, nutrients, pH, irrigation) so the data-mining models have a
genuine signal to learn. Deliberate data-quality problems (missing values, duplicates,
inconsistent text, outliers) are injected so the ETL stage has real work to do.
"""
import numpy as np
import pandas as pd

from .config import RAW_CSV

# district: (zone, typical soil, rainfall factor, temperature offset)
DISTRICTS = {
    "Thanjavur": ("Delta", "Alluvial", 1.15, 0.0),
    "Tiruvarur": ("Delta", "Alluvial", 1.15, 0.2),
    "Cuddalore": ("Coastal", "Alluvial", 1.10, 0.2),
    "Madurai": ("Southern", "Red Loam", 0.85, 1.0),
    "Tirunelveli": ("Southern", "Black Cotton", 0.80, 0.8),
    "Coimbatore": ("Western", "Red Loam", 0.90, -1.0),
    "Erode": ("Western", "Black Cotton", 0.85, 0.5),
    "Salem": ("Northern", "Red Loam", 0.90, 0.3),
    "Vellore": ("Northern", "Sandy Loam", 0.85, 0.8),
    "Villupuram": ("Northern", "Red Loam", 0.95, 0.4),
    "Tiruchirappalli": ("Central", "Sandy Loam", 0.90, 0.6),
}

# Agronomic profile of every crop (also reused by the recommendation engine).
CROPS = {
    "Rice":       dict(type="Cereal",  base=4.2, rain=800, tol=320, temp=27, ttol=6, n_opt=150, irr=80,
                       seasons=["Kharif", "Rabi"],           soil={"Alluvial": 1.10, "Black Cotton": 0.95}),
    "Maize":      dict(type="Cereal",  base=5.0, rain=500, tol=250, temp=26, ttol=6, n_opt=160, irr=55,
                       seasons=["Kharif", "Rabi", "Summer"], soil={"Red Loam": 1.05, "Alluvial": 1.05}),
    "Cotton":     dict(type="Fibre",   base=2.0, rain=550, tol=250, temp=28, ttol=5, n_opt=110, irr=50,
                       seasons=["Kharif", "Rabi"],           soil={"Black Cotton": 1.15, "Sandy Loam": 0.9}),
    "Groundnut":  dict(type="Oilseed", base=2.4, rain=450, tol=220, temp=27, ttol=5, n_opt=60, irr=50,
                       seasons=["Kharif", "Rabi", "Summer"], soil={"Sandy Loam": 1.12, "Red Loam": 1.05}),
    "Ragi":       dict(type="Millet",  base=2.8, rain=400, tol=220, temp=26, ttol=6, n_opt=80, irr=40,
                       seasons=["Kharif", "Rabi"],           soil={"Red Loam": 1.10, "Sandy Loam": 1.0}),
    "Sorghum":    dict(type="Millet",  base=2.6, rain=420, tol=230, temp=28, ttol=6, n_opt=90, irr=40,
                       seasons=["Kharif", "Rabi"],           soil={"Black Cotton": 1.10, "Red Loam": 1.0}),
    "Black Gram": dict(type="Pulse",   base=0.9, rain=350, tol=200, temp=28, ttol=5, n_opt=40, irr=40,
                       seasons=["Kharif", "Rabi", "Summer"], soil={"Alluvial": 1.08, "Sandy Loam": 1.0}),
}

# season: (mean rainfall mm, sd, mean temp, mean humidity)
SEASONS = {"Kharif": (480, 120, 30.0, 70), "Rabi": (420, 110, 26.0, 76), "Summer": (110, 50, 33.0, 60)}
SOIL_TYPES = ["Alluvial", "Red Loam", "Black Cotton", "Sandy Loam"]


def _gauss(x, opt, tol):
    return float(np.exp(-((x - opt) / tol) ** 2))


def generate_raw_data(seed: int = 42, save: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    rid = 1
    for year in range(2014, 2024):
        for district, (zone, soil_typ, rain_f, t_off) in DISTRICTS.items():
            year_climate = rng.normal(1.0, 0.12)
            for crop, p in CROPS.items():
                for season in p["seasons"]:
                    if rng.random() > 0.62:
                        continue
                    s_rain, s_sd, s_temp, s_hum = SEASONS[season]
                    rain = max(15.0, s_rain * rain_f * year_climate + rng.normal(0, s_sd))
                    temp = s_temp + t_off + rng.normal(0, 1.2)
                    hum = float(np.clip(s_hum + rng.normal(0, 6) + (rain - s_rain) * 0.02, 35, 98))
                    soil = soil_typ if rng.random() > 0.12 else str(rng.choice(SOIL_TYPES))
                    n = float(np.clip(rng.normal(130, 35), 50, 240))
                    ph_ = float(np.clip(rng.normal(6.8, 0.7), 4.8, 8.6))
                    phos = float(np.clip(rng.normal(45, 14), 12, 95))
                    pot = float(np.clip(rng.normal(110, 30), 40, 220))
                    fert = float(np.clip(rng.normal(p["n_opt"] * 0.9, 25), 10, 260))
                    irr_base = p["irr"] + (15 if season == "Summer" else 0)
                    irr = float(np.clip(rng.normal(irr_base, 15), 5, 100))
                    pest = float(np.clip(rng.normal(2.5, 0.8), 0.4, 5.5))
                    area = float(np.clip(rng.lognormal(6.5, 0.6), 60, 6000))

                    eff_water = rain + irr / 100 * 300
                    suit = (_gauss(eff_water, p["rain"], p["tol"]) ** 0.6) * (_gauss(temp, p["temp"], p["ttol"]) ** 0.6)
                    n_eff = n + 0.4 * fert
                    n_f = 0.7 + 0.4 * min(n_eff / (p["n_opt"] * 1.4), 1.0)
                    ph_f = 0.72 + 0.28 * _gauss(ph_, 6.8, 1.6)
                    p_f = 0.94 + 0.06 * min(phos / 45, 1.2)
                    pest_f = 0.95 + 0.05 * min(pest / 2.5, 1.2)
                    trend = 1 + 0.012 * (year - 2014)
                    soil_f = p["soil"].get(soil, 0.97)
                    y = p["base"] * (0.42 + 0.58 * suit) * n_f * ph_f * p_f * pest_f * trend * soil_f
                    y *= 1 + rng.normal(0, 0.06)
                    rows.append(dict(
                        record_id=rid, year=year, season=season, state="Tamil Nadu", district=district,
                        zone=zone, crop_name=crop, crop_type=p["type"], soil_type=soil, area_ha=round(area, 1),
                        rainfall_mm=round(rain, 1), temperature_c=round(temp, 1), humidity_pct=round(hum, 1),
                        nitrogen=round(n, 1), phosphorus=round(phos, 1), potassium=round(pot, 1),
                        soil_ph=round(ph_, 2), fertilizer_kg_ha=round(fert, 1), irrigation_pct=round(irr, 1),
                        pesticide_kg_ha=round(pest, 2), yield_t_ha=y,
                    ))
                    rid += 1
    df = pd.DataFrame(rows)

    # --- inject extreme events (drought / pest outbreak / flood) -------------------------
    idx = rng.choice(df.index, size=int(len(df) * 0.03), replace=False)
    for k, i in enumerate(idx):
        df.loc[i, "yield_t_ha"] *= rng.uniform(0.30, 0.50)
        if k % 3 == 0:
            df.loc[i, "rainfall_mm"] = round(df.loc[i, "rainfall_mm"] * 2.6, 1)
    df["yield_t_ha"] = df["yield_t_ha"].round(3)

    # --- inject data-quality problems for the ETL stage ----------------------------------
    for col in ["rainfall_mm", "soil_ph", "nitrogen", "humidity_pct"]:
        df.loc[rng.choice(df.index, size=int(len(df) * 0.015), replace=False), col] = np.nan
    messy = rng.choice(df.index, size=25, replace=False)
    df.loc[messy[:12], "district"] = df.loc[messy[:12], "district"].str.lower() + " "
    df.loc[messy[12:], "crop_name"] = df.loc[messy[12:], "crop_name"].str.upper()
    df = pd.concat([df, df.sample(8, random_state=seed)], ignore_index=True)  # duplicates
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if save:
        RAW_CSV.parent.mkdir(exist_ok=True)
        df.to_csv(RAW_CSV, index=False)
    return df
