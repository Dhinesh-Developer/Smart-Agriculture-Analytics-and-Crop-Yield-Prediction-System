"""Farmer recommendation engine: crop ranking (model-driven) and agronomic advice (rule-driven)."""
import pandas as pd

from .data_generator import CROPS
from .mining import FEATURES


def predict_yield(model, inputs: dict) -> float:
    return float(model.predict(pd.DataFrame([inputs])[FEATURES])[0])


def predict_class(model, inputs: dict):
    X = pd.DataFrame([inputs])[FEATURES]
    probs = model.predict_proba(X)[0]
    return str(model.predict(X)[0]), dict(zip(model.classes_, probs))


def rank_crops(reg_model, conditions: dict, season: str, hist_mean: dict = None) -> pd.DataFrame:
    """Rank every crop grown in the season by model-predicted yield under the given field conditions.

    Each crop is evaluated with its own standard fertiliser dose so that crops are compared fairly.
    """
    hist_mean = hist_mean or {}
    rows = []
    for crop, p in CROPS.items():
        if season not in p["seasons"]:
            continue
        x = dict(conditions, crop_name=crop, season=season, fertilizer_kg_ha=p["n_opt"] * 0.9)
        typical = hist_mean.get(crop, p["base"] * 0.75)
        pred = predict_yield(reg_model, x)
        rows.append({"Crop": crop, "Crop Type": p["type"], "Predicted Yield (t/ha)": pred,
                     "Historical Avg (t/ha)": typical, "Vs Historical (%)": (pred / typical - 1) * 100})
    return pd.DataFrame(rows).sort_values("Predicted Yield (t/ha)", ascending=False).reset_index(drop=True)


def advise(crop: str, c: dict) -> pd.DataFrame:
    """Compare field conditions with the crop's agronomic profile and give actionable advice."""
    p = CROPS[crop]
    eff_water = c["rainfall_mm"] + c["irrigation_pct"] / 100 * 300
    n_eff = c["nitrogen"] + 0.4 * c["fertilizer_kg_ha"]
    rows = []

    def add(param, current, optimal, ok, good_msg, bad_msg):
        rows.append({"Parameter": param, "Current": current, "Recommended": optimal,
                     "Status": "Optimal" if ok else "Action needed", "Advice": good_msg if ok else bad_msg})

    lo, hi = p["rain"] - p["tol"] * 0.6, p["rain"] + p["tol"] * 0.6
    add("Effective water (rain + irrigation, mm)", f"{eff_water:.0f}", f"{lo:.0f} - {hi:.0f}", lo <= eff_water <= hi,
        "Water supply matches crop demand.",
        "Increase irrigation / water harvesting." if eff_water < lo else "Improve drainage; excess water risks root disease.")
    tlo, thi = p["temp"] - p["ttol"] * 0.5, p["temp"] + p["ttol"] * 0.5
    add("Temperature (°C)", f"{c['temperature_c']:.1f}", f"{tlo:.1f} - {thi:.1f}", tlo <= c["temperature_c"] <= thi,
        "Temperature is suitable.",
        "Consider mulching / shade nets or shift sowing date." if c["temperature_c"] > thi else "Delay sowing or use protective cover.")
    add("Effective nitrogen (kg/ha)", f"{n_eff:.0f}", f">= {p['n_opt'] * 1.0:.0f}", n_eff >= p["n_opt"],
        "Nitrogen supply is adequate.", "Apply split doses of nitrogenous fertiliser (e.g. urea) or organic manure.")
    add("Soil pH", f"{c['soil_ph']:.2f}", "6.2 - 7.4", 6.2 <= c["soil_ph"] <= 7.4,
        "Soil reaction is suitable.",
        "Apply lime to raise pH." if c["soil_ph"] < 6.2 else "Apply gypsum / organic matter to lower pH.")
    add("Phosphorus (kg/ha)", f"{c['phosphorus']:.0f}", ">= 35", c["phosphorus"] >= 35,
        "Phosphorus level is sufficient.", "Apply SSP / DAP as basal dose.")
    add("Irrigation coverage (%)", f"{c['irrigation_pct']:.0f}", f">= {p['irr'] - 10}", c["irrigation_pct"] >= p["irr"] - 10,
        "Irrigation coverage is sufficient.", "Adopt drip / sprinkler irrigation to improve coverage.")
    return pd.DataFrame(rows)
