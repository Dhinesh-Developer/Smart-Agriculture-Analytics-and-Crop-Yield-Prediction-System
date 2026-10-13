"""Data-mining engine: regression, classification, PCA, clustering, anomaly detection, trend forecasting."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import CATEGORICAL_FEATURES, CLASS_ORDER, NUMERIC_FEATURES, TARGET_CLS, TARGET_REG

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SEED = 42


def _preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def _feature_names(pipe):
    pre = pipe.named_steps["prep"]
    return list(NUMERIC_FEATURES) + list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))


# ----------------------------------------------------------------------------- regression
def train_regression(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET_REG]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=250, random_state=SEED, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=SEED),
    }
    rows, fitted = [], {}
    for name, est in candidates.items():
        pipe = Pipeline([("prep", _preprocessor()), ("model", est)]).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rows.append({"Model": name, "R²": r2_score(yte, pred), "MAE": mean_absolute_error(yte, pred),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, pred)))})
        fitted[name] = (pipe, pred)
    results = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    best_pipe = fitted[best_name][0]
    # final model re-fit on all data for deployment-style predictions
    final = Pipeline([("prep", _preprocessor()), ("model", candidates[best_name].__class__(**candidates[best_name].get_params()))]).fit(X, y)
    est = best_pipe.named_steps["model"]
    imp = None
    if hasattr(est, "feature_importances_"):
        imp = pd.DataFrame({"Feature": _feature_names(best_pipe), "Importance": est.feature_importances_})
        imp = imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    test_df = pd.DataFrame({"Actual": yte.values, "Predicted": fitted[best_name][1]})
    return {"results": results, "best_name": best_name, "model": final, "importance": imp, "test": test_df}


# ----------------------------------------------------------------------------- classification
def train_classification(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET_CLS]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    candidates = {
        "Decision Tree": DecisionTreeClassifier(max_depth=7, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=SEED, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=9),
    }
    rows, fitted = [], {}
    for name, est in candidates.items():
        pipe = Pipeline([("prep", _preprocessor()), ("model", est)]).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred),
                     "Precision": precision_score(yte, pred, average="weighted", zero_division=0),
                     "Recall": recall_score(yte, pred, average="weighted", zero_division=0),
                     "F1-Score": f1_score(yte, pred, average="weighted", zero_division=0)})
        fitted[name] = pred
    results = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    cm = pd.DataFrame(confusion_matrix(yte, fitted[best_name], labels=CLASS_ORDER),
                      index=[f"Actual {c}" for c in CLASS_ORDER], columns=[f"Pred {c}" for c in CLASS_ORDER])
    final = Pipeline([("prep", _preprocessor()),
                      ("model", candidates[best_name].__class__(**candidates[best_name].get_params()))]).fit(X, y)
    return {"results": results, "best_name": best_name, "model": final, "confusion": cm}


# ----------------------------------------------------------------------------- PCA
def run_pca(df: pd.DataFrame):
    cols = NUMERIC_FEATURES + [TARGET_REG]
    scaled = StandardScaler().fit_transform(df[cols])
    pca = PCA().fit(scaled)
    ratio = pca.explained_variance_ratio_
    var_df = pd.DataFrame({"Component": [f"PC{i + 1}" for i in range(len(ratio))],
                           "Explained Variance (%)": ratio * 100, "Cumulative (%)": np.cumsum(ratio) * 100})
    coords = pca.transform(scaled)[:, :3]
    proj = pd.DataFrame(coords, columns=["PC1", "PC2", "PC3"])
    proj["Crop"] = df["crop_name"].values
    proj["Yield Class"] = df[TARGET_CLS].values
    loadings = pd.DataFrame(pca.components_[:3].T, index=cols, columns=["PC1", "PC2", "PC3"])
    return {"variance": var_df, "projection": proj, "loadings": loadings, "n95": int((np.cumsum(ratio) < 0.95).sum() + 1)}


# ----------------------------------------------------------------------------- clustering
def run_clustering(df: pd.DataFrame, k: int = 4):
    cols = NUMERIC_FEATURES + [TARGET_REG]
    scaled = StandardScaler().fit_transform(df[cols])
    reduced = PCA(n_components=3, random_state=SEED).fit_transform(scaled)
    elbow = []
    for kk in range(2, 9):
        km = KMeans(n_clusters=kk, n_init=10, random_state=SEED).fit(scaled)
        elbow.append({"k": kk, "Inertia": km.inertia_, "Silhouette": silhouette_score(scaled, km.labels_)})
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(scaled)
    out = df.copy()
    out["cluster_id"] = km.labels_
    rank = out.groupby("cluster_id")["yield_t_ha"].mean().rank(ascending=False).astype(int)
    out["Cluster"] = out["cluster_id"].map(lambda c: f"Cluster {rank[c]}")
    profile = out.groupby("Cluster").agg(Records=("fact_id", "count"), Avg_Yield=("yield_t_ha", "mean"),
                                         Rainfall=("rainfall_mm", "mean"), Temperature=("temperature_c", "mean"),
                                         Nitrogen=("nitrogen", "mean"), Soil_pH=("soil_ph", "mean"),
                                         Irrigation=("irrigation_pct", "mean")).round(2).reset_index()
    profile.columns = ["Cluster", "Records", "Avg Yield (t/ha)", "Rainfall (mm)", "Temp (°C)", "Nitrogen", "Soil pH", "Irrigation (%)"]
    top_crop = out.groupby("Cluster")["crop_name"].agg(lambda s: s.value_counts().index[0])
    profile["Dominant Crop"] = profile["Cluster"].map(top_crop)
    coords = pd.DataFrame(reduced, columns=["PC1", "PC2", "PC3"])
    coords["Cluster"] = out["Cluster"].values
    return {"elbow": pd.DataFrame(elbow), "profile": profile.sort_values("Cluster"), "coords": coords}


# ----------------------------------------------------------------------------- anomaly detection
def detect_anomalies(df: pd.DataFrame, contamination: float = 0.04):
    out = df.copy()
    grp = out.groupby("crop_name")["yield_t_ha"]
    out["yield_zscore"] = (out["yield_t_ha"] - grp.transform("mean")) / grp.transform("std")
    cols = NUMERIC_FEATURES + [TARGET_REG]
    iso = IsolationForest(contamination=contamination, random_state=SEED, n_estimators=200)
    Xs = StandardScaler().fit_transform(out[cols])
    iso.fit(Xs)
    out["anomaly_score"] = -iso.score_samples(Xs)
    out["is_anomaly"] = (iso.predict(Xs) == -1) | (out["yield_zscore"].abs() > 2.8)

    def reason(r):
        why = []
        if r["yield_zscore"] < -2.0:
            why.append("Yield far below crop norm")
        if r["yield_zscore"] > 2.0:
            why.append("Yield far above crop norm")
        if r["rainfall_mm"] > 900:
            why.append("Extreme rainfall (flood risk)")
        if r["rainfall_mm"] < 60 and r["irrigation_pct"] < 40:
            why.append("Drought stress")
        return "; ".join(why) if why else "Unusual multivariate pattern"

    out["reason"] = ""
    out.loc[out["is_anomaly"], "reason"] = out[out["is_anomaly"]].apply(reason, axis=1)
    return out


# ----------------------------------------------------------------------------- forecasting
def forecast_trend(df: pd.DataFrame, crop: str, horizon: int = 3, district: str = None):
    sub = df[df["crop_name"] == crop]
    if district and district != "All Districts":
        sub = sub[sub["district"] == district]
    yearly = sub.groupby("year")["yield_t_ha"].mean().reset_index()
    if len(yearly) < 4:
        return None
    slope, intercept = np.polyfit(yearly["year"], yearly["yield_t_ha"], 1)
    fitted = slope * yearly["year"] + intercept
    sigma = float(np.std(yearly["yield_t_ha"] - fitted, ddof=2))
    last = int(yearly["year"].max())
    fut = pd.DataFrame({"year": range(last + 1, last + horizon + 1)})
    fut["forecast"] = slope * fut["year"] + intercept
    fut["lower"] = fut["forecast"] - 1.96 * sigma
    fut["upper"] = fut["forecast"] + 1.96 * sigma
    return {"history": yearly, "fitted": fitted, "future": fut, "slope": slope, "sigma": sigma}
