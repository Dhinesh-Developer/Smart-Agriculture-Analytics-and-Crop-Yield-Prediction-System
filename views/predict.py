import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import mining, recommender
from src.config import CLASS_ORDER
from src.data_generator import CROPS, DISTRICTS, SEASONS
from src.ui import html_table, kpi_row, page_header, show


def render(df, models):
    page_header("Crop Yield Prediction", "Predict yield for a field and forecast long-term crop trends")
    t1, t2 = st.tabs(["🌾 Field-level Prediction", "📅 Yield Forecast (Trend)"])

    with t1:
        c1, c2, c3 = st.columns(3)
        district = c1.selectbox("District", list(DISTRICTS.keys()))
        crop = c2.selectbox("Crop", list(CROPS.keys()))
        season = c3.selectbox("Season", [s for s in SEASONS if s in CROPS[crop]["seasons"]] or list(SEASONS))
        zone, soil_default = DISTRICTS[district][0], DISTRICTS[district][1]
        with st.form("predict_form"):
            soil_opts = sorted(df["soil_type"].unique())
            c1, c2, c3, c4 = st.columns(4)
            soil = c1.selectbox("Soil type", soil_opts, index=soil_opts.index(soil_default) if soil_default in soil_opts else 0)
            rain = c2.number_input("Rainfall (mm)", 0.0, 2500.0, 450.0, 10.0)
            temp = c3.number_input("Temperature (°C)", 10.0, 45.0, 28.0, 0.5)
            hum = c4.number_input("Humidity (%)", 20.0, 100.0, 70.0, 1.0)
            c1, c2, c3, c4 = st.columns(4)
            n = c1.number_input("Nitrogen (kg/ha)", 20.0, 300.0, 130.0, 5.0)
            p = c2.number_input("Phosphorus (kg/ha)", 5.0, 120.0, 45.0, 1.0)
            k = c3.number_input("Potassium (kg/ha)", 20.0, 260.0, 110.0, 5.0)
            ph = c4.number_input("Soil pH", 4.0, 9.5, 6.8, 0.1)
            c1, c2, c3 = st.columns(3)
            fert = c1.number_input("Fertilizer (kg/ha)", 0.0, 300.0, float(CROPS[crop]["n_opt"] * 0.9), 5.0)
            irr = c2.slider("Irrigation coverage (%)", 0, 100, int(CROPS[crop]["irr"]))
            pest = c3.number_input("Pesticide (kg/ha)", 0.0, 8.0, 2.5, 0.1)
            go_btn = st.form_submit_button("🔮 Predict Yield")

        if go_btn:
            x = dict(rainfall_mm=rain, temperature_c=temp, humidity_pct=hum, nitrogen=n, phosphorus=p, potassium=k, soil_ph=ph,
                     fertilizer_kg_ha=fert, irrigation_pct=irr, pesticide_kg_ha=pest, crop_name=crop, season=season,
                     soil_type=soil, zone=zone)
            y = recommender.predict_yield(models["reg"]["model"], x)
            label, probs = recommender.predict_class(models["cls"]["model"], x)
            hist = df[df["crop_name"] == crop]["yield_t_ha"]
            kpi_row([("Predicted Yield", f"{y:.2f} t/ha", f"{models['reg']['best_name']}"),
                     ("Predicted Class", label, "Relative to crop history"),
                     ("Historical Average", f"{hist.mean():.2f} t/ha", f"{crop}, all records"),
                     ("Vs Historical", f"{(y / hist.mean() - 1) * 100:+.1f}%", "Difference")])
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=y, number={"suffix": " t/ha"},
                    delta={"reference": float(hist.mean()), "valueformat": ".2f"},
                    title={"text": f"Predicted Yield - {crop}"},
                    gauge={"axis": {"range": [0, float(hist.max() * 1.1)]}, "bar": {"color": "#2E7D32"},
                           "steps": [{"range": [0, float(hist.quantile(.33))], "color": "#FFCDD2"},
                                     {"range": [float(hist.quantile(.33)), float(hist.quantile(.66))], "color": "#FFF9C4"},
                                     {"range": [float(hist.quantile(.66)), float(hist.max() * 1.1)], "color": "#C8E6C9"}]}))
                show(fig, 360)
            with c2:
                pr = pd.DataFrame({"Class": CLASS_ORDER, "Probability": [probs.get(c, 0) for c in CLASS_ORDER]})
                fig = go.Figure(go.Bar(x=pr["Class"], y=pr["Probability"], marker_color=["#C62828", "#F9A825", "#2E7D32"],
                                       text=[f"{v:.0%}" for v in pr["Probability"]], textposition="outside"))
                fig.update_layout(title="Yield Class Probability", yaxis=dict(range=[0, 1.1], tickformat=".0%"))
                show(fig, 360)

            report = pd.DataFrame({"Parameter": ["District", "Zone", "Crop", "Season", "Soil", "Rainfall (mm)", "Temperature (°C)",
                                                 "Humidity (%)", "Nitrogen", "Phosphorus", "Potassium", "Soil pH", "Fertilizer",
                                                 "Irrigation (%)", "Pesticide", "Predicted yield (t/ha)", "Predicted class"],
                                   "Value": [district, zone, crop, season, soil, rain, temp, hum, n, p, k, ph, fert, irr, pest,
                                             round(y, 3), label]}).astype(str)
            st.subheader("Prediction Report")
            html_table(report)
            st.download_button("⬇ Download prediction report (CSV)", report.to_csv(index=False).encode(), "crop_prediction_report.csv", "text/csv")
        else:
            st.info("Enter field conditions above and press **Predict Yield**.")

    with t2:
        c1, c2, c3 = st.columns(3)
        crop = c1.selectbox("Crop", sorted(df["crop_name"].unique()), key="fc_crop")
        dist = c2.selectbox("District", ["All Districts"] + sorted(df["district"].unique()), key="fc_dist")
        hz = c3.slider("Forecast horizon (years)", 1, 5, 3)
        f = mining.forecast_trend(df, crop, hz, dist)
        if f is None:
            st.warning("Not enough yearly data for this selection.")
        else:
            h, fu = f["history"], f["future"]
            fig = go.Figure()
            fig.add_scatter(x=h["year"], y=h["yield_t_ha"], mode="lines+markers", name="Observed", line=dict(color="#1565C0"))
            fig.add_scatter(x=h["year"], y=f["fitted"], mode="lines", name="Linear trend", line=dict(color="#9E9E9E", dash="dot"))
            fig.add_scatter(x=fu["year"], y=fu["forecast"], mode="lines+markers", name="Forecast", line=dict(color="#2E7D32", dash="dash"))
            fig.add_scatter(x=list(fu["year"]) + list(fu["year"])[::-1], y=list(fu["upper"]) + list(fu["lower"])[::-1], fill="toself",
                            fillcolor="rgba(46,125,50,0.15)", line=dict(width=0), name="95% interval", hoverinfo="skip")
            fig.update_layout(title=f"{crop} Yield Forecast - {dist}", xaxis_title="Year", yaxis_title="Average Yield (t/ha)")
            fig.update_xaxes(dtick=1)
            show(fig, 440)
            kpi_row([("Trend", f"{f['slope']:+.3f} t/ha/yr", "Linear slope"), ("Next-Year Forecast", f"{fu['forecast'].iloc[0]:.2f} t/ha", f"{int(fu['year'].iloc[0])}"),
                     ("Uncertainty (σ)", f"{f['sigma']:.2f} t/ha", "Residual std. dev.")])
            out = fu.rename(columns={"year": "Year", "forecast": "Forecast (t/ha)", "lower": "Lower 95%", "upper": "Upper 95%"})
            out["Year"] = out["Year"].astype(str)
            html_table(out, 3)
