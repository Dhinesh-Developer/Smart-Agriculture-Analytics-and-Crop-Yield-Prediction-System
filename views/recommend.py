import plotly.express as px
import streamlit as st

from src import recommender
from src.data_generator import CROPS, DISTRICTS, SEASONS
from src.ui import html_table, kpi_row, page_header, show


def render(df, models):
    page_header("Farmer Recommendations", "Best-crop ranking and agronomic advice for the farmer's field conditions")
    t1, t2 = st.tabs(["🧑‍🌾 Personalised Advisor", "🗺️ District Crop Guide"])

    with t1:
        c1, c2 = st.columns(2)
        district = c1.selectbox("District", list(DISTRICTS.keys()), key="r_dist")
        season = c2.selectbox("Season", list(SEASONS.keys()), key="r_season")
        zone, soil_default = DISTRICTS[district][0], DISTRICTS[district][1]
        ref = df[(df["district"] == district) & (df["season"] == season)]
        ref = ref if len(ref) else df[df["district"] == district]
        soil_opts = sorted(df["soil_type"].unique())
        st.caption("Field conditions are pre-filled with the district's historical averages for the chosen season - adjust as needed.")
        c1, c2, c3, c4 = st.columns(4)
        soil = c1.selectbox("Soil type", soil_opts, index=soil_opts.index(soil_default) if soil_default in soil_opts else 0, key="r_soil")
        rain = c2.number_input("Rainfall (mm)", 0.0, 2500.0, float(round(ref["rainfall_mm"].mean(), 0)), 10.0, key="r_rain")
        temp = c3.number_input("Temperature (°C)", 10.0, 45.0, float(round(ref["temperature_c"].mean(), 1)), 0.5, key="r_temp")
        hum = c4.number_input("Humidity (%)", 20.0, 100.0, float(round(ref["humidity_pct"].mean(), 0)), 1.0, key="r_hum")
        c1, c2, c3, c4 = st.columns(4)
        n = c1.number_input("Nitrogen (kg/ha)", 20.0, 300.0, float(round(ref["nitrogen"].mean(), 0)), 5.0, key="r_n")
        p = c2.number_input("Phosphorus (kg/ha)", 5.0, 120.0, float(round(ref["phosphorus"].mean(), 0)), 1.0, key="r_p")
        ph = c3.number_input("Soil pH", 4.0, 9.5, float(round(ref["soil_ph"].mean(), 1)), 0.1, key="r_ph")
        irr = c4.slider("Irrigation coverage (%)", 0, 100, int(round(ref["irrigation_pct"].mean())), key="r_irr")

        cond = dict(rainfall_mm=rain, temperature_c=temp, humidity_pct=hum, nitrogen=n, phosphorus=p,
                    potassium=float(ref["potassium"].mean()), soil_ph=ph, fertilizer_kg_ha=0.0, irrigation_pct=irr,
                    pesticide_kg_ha=float(ref["pesticide_kg_ha"].mean()), soil_type=soil, zone=zone)
        hist = df.groupby("crop_name")["yield_t_ha"].mean().to_dict()
        rank = recommender.rank_crops(models["reg"]["model"], cond, season, hist)

        top = rank.iloc[0]
        kpi_row([("Recommended Crop", top["Crop"], f"{top['Crop Type']}"), ("Expected Yield", f"{top['Predicted Yield (t/ha)']:.2f} t/ha", "Model prediction"),
                 ("Vs Historical", f"{top['Vs Historical (%)']:+.1f}%", "Crop average")])
        fig = px.bar(rank.iloc[::-1], x="Predicted Yield (t/ha)", y="Crop", orientation="h", text_auto=".2f", color="Vs Historical (%)",
                     color_continuous_scale="RdYlGn", color_continuous_midpoint=0, title=f"Crop Ranking - {district}, {season}")
        show(fig, 400)
        html_table(rank.round(2), 2)

        st.subheader("Agronomic Advice")
        pick = st.selectbox("Crop to evaluate", rank["Crop"].tolist())
        adv_cond = dict(cond, fertilizer_kg_ha=CROPS[pick]["n_opt"] * 0.9)
        adv = recommender.advise(pick, adv_cond)
        html_table(adv, 2, status_col="Status")
        n_bad = int((adv["Status"] != "Optimal").sum())
        (st.success if n_bad == 0 else st.warning)(
            "All monitored parameters are within the optimal range." if n_bad == 0 else f"{n_bad} parameter(s) need attention for {pick}.")

    with t2:
        st.markdown("Data-driven best crop for every district and season, based on historical average yield relative to each crop's own average.")
        rel = df.copy()
        rel["relative"] = rel["yield_t_ha"] / rel.groupby("crop_name")["yield_t_ha"].transform("mean")
        g = rel.groupby(["district", "crop_name"])["relative"].mean().reset_index()
        pv = g.pivot(index="district", columns="crop_name", values="relative")
        fig = px.imshow(pv, color_continuous_scale="RdYlGn", color_continuous_midpoint=1, aspect="auto", text_auto=".2f",
                        labels=dict(x="Crop", y="District", color="Relative yield"), title="Relative Yield Index (1.00 = crop average)")
        show(fig, 520)
        best = g.loc[g.groupby("district")["relative"].idxmax()].rename(columns={"district": "District", "crop_name": "Best-Performing Crop", "relative": "Relative Yield Index"})
        st.subheader("Best-Performing Crop per District")
        html_table(best.sort_values("District"), 2)
