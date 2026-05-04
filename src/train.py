"""
Task 1 — Experiment Tracking & Model Comparison
Trains LinearRegression and RandomForest, logs to MLflow, saves step1_s1.json
"""
import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENT_NAME = "cropsense-irrigation-hours"
TRACKING_URI = f"file://{os.path.join(BASE_DIR, 'mlruns')}"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Load & split data
df = pd.read_csv(DATA_PATH)
FEATURES = ["soil_moisture_pct", "crop_type_index", "field_size_hectares", "temperature_c"]
X = df[FEATURES]
y = df["irrigation_hours"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {"experiment_name": EXPERIMENT_NAME, "models": []}
best_mae = float("inf")
best_model_name = None
best_run_id = None
run_ids = {}

# --- LinearRegression ---
with mlflow.start_run(run_name="LinearRegression") as run:
    mlflow.set_tag("experiment_type", "baseline_comparison")
    params = {"fit_intercept": True}
    mlflow.log_params(params)

    lr = LinearRegression(**params)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(lr, "model")

    lr_run_id = run.info.run_id
    run_ids["LinearRegression"] = lr_run_id
    results["models"].append({"name": "LinearRegression", "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)})

    if mae < best_mae:
        best_mae, best_model_name, best_run_id = mae, "LinearRegression", lr_run_id

# --- RandomForest ---
with mlflow.start_run(run_name="RandomForest") as run:
    mlflow.set_tag("experiment_type", "baseline_comparison")
    params = {"n_estimators": 100, "random_state": 42}
    mlflow.log_params(params)

    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(rf, "model")

    rf_run_id = run.info.run_id
    run_ids["RandomForest"] = rf_run_id
    results["models"].append({"name": "RandomForest", "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)})

    if mae < best_mae:
        best_mae, best_model_name, best_run_id = mae, "RandomForest", rf_run_id

results["best_model"] = best_model_name
results["best_metric_name"] = "mae"
results["best_metric_value"] = round(best_mae, 4)

# Persist info for downstream scripts
meta = {
    "best_model": best_model_name,
    "best_run_id": best_run_id,
    "run_ids": run_ids,
    "tracking_uri": TRACKING_URI,
}
with open(os.path.join(MODELS_DIR, "best_run_info.json"), "w") as f:
    json.dump(meta, f, indent=2)

with open(os.path.join(RESULTS_DIR, "step1_s1.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Task 1 complete.")
print(json.dumps(results, indent=2))
