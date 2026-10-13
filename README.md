# 🌾 Smart Agriculture Analytics and Crop Yield Prediction System

## 1. Abstract

Agriculture is highly sensitive to weather, soil condition and farm-management decisions, yet the data that describe
these factors are usually scattered across different departments and formats. This project presents the
**Smart Agriculture Analytics and Crop Yield Prediction System**, a decision-support platform that integrates
**weather, soil and crop data** into a centralised **data warehouse** (SQLite star schema) through an automated
**ETL pipeline**. The warehouse is explored through **OLAP operations** (roll-up, drill-down, slice, dice and pivot)
and mined with **data-mining techniques**: **Principal Component Analysis (PCA)** for dimensionality reduction,
**K-Means clustering** to discover agro-climatic groups, **classification** (Decision Tree, Random Forest, Logistic
Regression, KNN) to label yield as Low / Medium / High, **regression** (Linear, Random Forest, Gradient Boosting) to
predict yield in tonnes per hectare, and **anomaly detection** (Isolation Forest + z-score) to identify drought,
flood and pest-affected records. The results are delivered through an interactive, white-themed **Streamlit**
dashboard offering crop forecasting, farmer recommendations, and downloadable crop-prediction reports. The whole
application (frontend and backend) is written in Python and can be deployed directly on Streamlit Community Cloud.

**Keywords:** Data Warehouse, ETL, OLAP, PCA, Classification, Clustering, Anomaly Detection, Crop Yield Prediction, Streamlit, SQLite.

## 2. Introduction

Crop yield depends on many interacting variables – rainfall, temperature, humidity, soil nutrients (N, P, K), soil pH,
irrigation, fertiliser and pesticide use. Traditional record-keeping makes it difficult to analyse these together.
A data-warehousing approach provides a clean, historical, subject-oriented store of integrated data, while data mining
extracts the hidden patterns that support better planning for farmers, agronomists and policy makers.

## 3. Problem Statement

*"How can heterogeneous weather, soil and crop records be integrated into a single analytical repository and mined to
forecast crop yields, detect abnormal seasons and recommend suitable crops and farming practices?"*

## 4. Objectives

1. Design and build a **data warehouse** (star schema) for agricultural data using SQLite.
2. Implement an **ETL pipeline** that cleans, standardises and loads raw data with an audit log.
3. Provide **OLAP analysis** over the Date, Region, Crop and Soil dimensions.
4. Apply **PCA, clustering, classification and regression** to understand and predict crop yield.
5. Detect **anomalous** seasons caused by drought, floods or pest outbreaks.
6. Offer **farmer recommendations** (best-crop ranking and agronomic advice).
7. Deliver everything as a professional **analytics dashboard** and **decision-support** application.

## 5. Key Features

| Feature | Description |
|---|---|
| Crop forecasting | Field-level yield prediction (t/ha) + class probability, and multi-year trend forecast with 95 % interval |
| Anomaly detection | Isolation Forest + per-crop z-score flag abnormal yields with probable cause |
| Farmer recommendations | Ranks crops for the farmer's conditions and gives parameter-wise agronomic advice |
| Analytics dashboards | KPIs, trend lines, bar graphs, donut chart, heat-maps, correlation matrix, box plots |
| OLAP explorer | Roll-up, drill-down, slice, dice, pivot – with the SQL shown for each operation |
| Warehouse tools | Star-schema diagram, ETL log, table explorer, read-only SQL console, CSV upload |
| Reports | Downloadable prediction report and filtered-data CSV |

## 6. System Architecture

```
 Data Sources            ETL                 Data Warehouse            Analytics & Mining            Presentation
┌──────────────┐   ┌──────────────┐      ┌──────────────────┐      ┌─────────────────────┐     ┌────────────────────┐
│ Weather data │   │ Extract      │      │ SQLite           │      │ OLAP (SQL)          │     │ Streamlit          │
│ Soil data    │──►│ Transform    │─────►│ Star schema      │─────►│ PCA, K-Means        │────►│ Dashboards, charts │
│ Crop yields  │   │ Load         │      │ 1 fact + 4 dims  │      │ Classification      │     │ Prediction forms   │
└──────────────┘   └──────────────┘      └──────────────────┘      │ Regression, Anomaly │     │ Recommendations    │
                                                                   └─────────────────────┘     └────────────────────┘
```

### 6.1 Star Schema

| Table | Type | Columns |
|---|---|---|
| `fact_crop_yield` | Fact | fact_id, date_id, region_id, crop_id, soil_id, area_ha, rainfall_mm, temperature_c, humidity_pct, nitrogen, phosphorus, potassium, soil_ph, fertilizer_kg_ha, irrigation_pct, pesticide_kg_ha, yield_t_ha, production_t, yield_class |
| `dim_date` | Dimension | date_id, year, season |
| `dim_region` | Dimension | region_id, state, district, zone |
| `dim_crop` | Dimension | crop_id, crop_name, crop_type |
| `dim_soil` | Dimension | soil_id, soil_type |

Hierarchies: **Season → Year**, **District → Zone → State**, **Crop → Crop Type**.
A view `vw_yield_analysis` joins the fact and dimensions for analysis.

### 6.2 ETL Pipeline

| Stage | Operation |
|---|---|
| Extract | Read the raw CSV (`data/raw_agri_data.csv`) |
| Transform | Validate schema → standardise text → remove duplicates → impute missing values (median by crop & season) → enforce valid ranges → derive `production_t` and `yield_class` |
| Load | Create star schema and bulk-insert dimensions and fact; write an ETL audit log |

## 7. Dataset

The project ships with a **synthetic but agronomically realistic dataset** (1,137 raw records; 1,129 after ETL) covering
**11 Tamil Nadu districts × 7 crops × 3 seasons × 10 years (2014 – 2023)**. It is generated by
`src/data_generator.py` using crop-specific optimum ranges for water, temperature, nutrients and pH, with random noise,
extreme events, missing values, duplicates and inconsistent text so that ETL and anomaly detection have real work to do.

> ⚠️ Because the data are synthetic, the reported accuracy figures describe the demonstration dataset, not real farm
> performance. To use real data, open **Data Warehouse & ETL → ETL Pipeline → Upload your own CSV** (columns listed in
> section 12) or replace `data/raw_agri_data.csv` and delete `data/agri_warehouse.db`.

| Group | Variables |
|---|---|
| Weather | rainfall_mm, temperature_c, humidity_pct |
| Soil | nitrogen, phosphorus, potassium, soil_ph, soil_type |
| Management | fertilizer_kg_ha, irrigation_pct, pesticide_kg_ha, area_ha |
| Target | yield_t_ha (regression), yield_class – Low / Medium / High (classification) |

## 8. Data-Mining Techniques and Results

| Technique | Algorithm(s) | Purpose |
|---|---|---|
| PCA | PCA on 11 standardised variables | Dimensionality reduction, scree plot, loadings |
| Clustering | K-Means (+ elbow & silhouette) | Group records into agro-climatic yield clusters |
| Classification | Decision Tree, Random Forest, Logistic Regression, KNN | Predict Low / Medium / High yield class |
| Regression | Linear, Random Forest, Gradient Boosting | Predict yield in t/ha |
| Anomaly detection | Isolation Forest + z-score | Flag drought / flood / pest affected records |
| Forecasting | Linear trend with 95 % interval | Multi-year crop yield outlook |

**Regression results (20 % hold-out test set, seed 42):**

| Model | R² | MAE (t/ha) | RMSE (t/ha) |
|---|:---:|:---:|:---:|
| Random Forest Regressor | **0.863** | **0.292** | **0.456** |
| Gradient Boosting Regressor | 0.853 | 0.317 | 0.472 |
| Linear Regression | 0.775 | 0.436 | 0.585 |

**Classification results (20 % hold-out test set, 3 balanced classes):**

| Model | Accuracy | F1-Score (weighted) |
|---|:---:|:---:|
| Random Forest | **0.633** | **0.623** |
| Decision Tree | 0.588 | 0.597 |
| K-Nearest Neighbours | 0.562 | 0.549 |
| Logistic Regression | 0.504 | 0.492 |

The classes are crop-relative tertiles (a record is "High" if it is in the top third of yields *for its own crop*),
which is a deliberately hard task; the random baseline is 33 %.

## 9. Application Modules

| Page | What it shows |
|---|---|
| 🏠 Home | Project summary, KPIs, workflow, module cards |
| 📊 Analytics Dashboard | Filters, KPI cards, line / bar / donut / heat-map / scatter / box charts, zone table |
| 🗄️ Data Warehouse & ETL | Star-schema diagram, ETL log, table explorer, SQL console, rebuild / upload data |
| 🧊 OLAP Explorer | Roll-up, drill-down, slice, dice and pivot with generated SQL |
| ⛏️ Data Mining Lab | PCA, clustering, classification, regression, anomaly detection |
| 🌱 Yield Prediction | Field-level prediction + gauge + class probabilities + trend forecast + CSV report |
| 🧑‍🌾 Farmer Recommendations | Crop ranking, agronomic advice table, district crop-guide heat-map |
| ℹ️ About Project | Project details, team, concept mapping, tools |

## 10. Tools and Technologies Used

| Tool / Library | Purpose |
|---|---|
| **Python 3.10+** | Main programming language |
| **Streamlit** | Frontend and backend (single-codebase web app, Streamlit Cloud deployment) |
| **SQLite** (`sqlite3`) | Data warehouse (star schema), OLAP queries |
| **Pandas, NumPy** | ETL and data manipulation |
| **Scikit-learn** | PCA, K-Means, classifiers, regressors, Isolation Forest, metrics |
| **Plotly** | Interactive charts: bar, line, pie/donut, heat-map, scatter, box, gauge, sunburst |
| **Graphviz (via Streamlit)** | Star-schema diagram |
| **venv** | Isolated Python environment |
| **VS Code / Git / GitHub** | Development and version control |
| **Streamlit Community Cloud** | Deployment |

Design: white theme, **Times New Roman** typography, green agricultural colour palette.

## 11. Project Structure

```
smart_agri_project/
├── app.py                    # Streamlit entry point (navigation, caching)
├── requirements.txt          # Python dependencies
├── README.md
├── .streamlit/config.toml    # White theme configuration
├── data/
│   └── raw_agri_data.csv     # Raw dataset (agri_warehouse.db is created on first run)
├── src/
│   ├── config.py             # Paths, constants, team details
│   ├── data_generator.py     # Synthetic raw-data generator
│   ├── etl.py                # Extract – Transform – Load, star-schema DDL
│   ├── warehouse.py          # SQL access layer and OLAP query builder
│   ├── mining.py             # PCA, clustering, classification, regression, anomaly, forecast
│   ├── recommender.py        # Crop ranking and agronomic advice
│   └── ui.py                 # Theme, fonts, KPI cards, chart styling
└── views/                    # One module per page
    ├── home.py   dashboard.py   warehouse_view.py   olap.py
    └── mining_view.py   predict.py   recommend.py   about.py
```

## 12. Setup and Run (VS Code)

**Windows (PowerShell / VS Code terminal)**
```powershell
cd smart_agri_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**
```bash
cd smart_agri_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open **http://localhost:8501**. On the first run the app builds the SQLite warehouse automatically (a few seconds).
In VS Code select the interpreter **Python: Select Interpreter → venv** so imports resolve.

> If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

**Using your own data** – upload a CSV with these columns:
`record_id, year, season, state, district, zone, crop_name, crop_type, soil_type, area_ha, rainfall_mm, temperature_c,
humidity_pct, nitrogen, phosphorus, potassium, soil_ph, fertilizer_kg_ha, irrigation_pct, pesticide_kg_ha, yield_t_ha`.
The district/crop pickers in the *Prediction* and *Recommendation* pages use the built-in Tamil Nadu profile; extend
`DISTRICTS` and `CROPS` in `src/data_generator.py` for other regions or crops.

## 13. Deploy on Streamlit Community Cloud

1. Push the folder to a GitHub repository (the `venv/` folder is excluded by `.gitignore`).
2. Go to **share.streamlit.io → New app**, choose the repository and set **Main file path = `app.py`**.
3. Click **Deploy**. Streamlit Cloud installs `requirements.txt` and the app builds its SQLite warehouse on first launch.

## 14. Conclusion

The system demonstrates the full data-warehousing and data-mining life cycle – from raw data integration through ETL, a
star-schema warehouse and OLAP, to PCA, clustering, classification, regression and anomaly detection – and turns the
results into practical forecasts and recommendations through a single Streamlit application.

## 15. Future Enhancements

- Integrate live weather APIs and satellite NDVI imagery.
- Add deep-learning / time-series models (LSTM, Prophet) for seasonal forecasting.
- Multilingual (Tamil / English) interface and SMS alerts for farmers.
- Migrate to PostgreSQL / cloud data warehouse for larger datasets.
- Market-price integration for profit-based crop recommendation.

## 16. References

1. J. Han, M. Kamber, J. Pei – *Data Mining: Concepts and Techniques*, Morgan Kaufmann.
2. R. Kimball, M. Ross – *The Data Warehouse Toolkit*, Wiley.
3. Scikit-learn documentation – https://scikit-learn.org
4. Streamlit documentation – https://docs.streamlit.io
5. Plotly documentation – https://plotly.com/python
6. SQLite documentation – https://sqlite.org/docs.html
