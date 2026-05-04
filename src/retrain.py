"""
Task 4 — Retraining Pipeline
- Combines training_data.csv + new_data.csv
- Retrains the winning model type from Task 1
- Evaluates on the same test set (20% of original training data, random_state=42)
- Promotes if retrained MAE < champion MAE (any improvement)
- Saves step4_s8.json
"""
import os
import json
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REGISTERED_MODEL_NAME = "cropsense-irrigation-hours-predictor"
ALIAS = "live"
FEATURES = ["soil_moisture_pct", "crop_type_index", "field_size_hectares", "temperature_c"]

with open(os.path.join(MODELS_DIR, "best_run_info.json")) as f:
    meta = json.load(f)

mlflow.set_tracking_uri(meta["tracking_uri"])
mlflow.set_experiment("cropsense-irrigation-hours")
client = MlflowClient()

best_model_name = meta["best_model"]
champion_mae    = meta["champion_mae"]

# Load datasets
train_df = pd.read_csv(os.path.join(BASE_DIR, "data", "training_data.csv"))
new_df   = pd.read_csv(os.path.join(BASE_DIR, "data", "new_data.csv"))
original_rows = len(train_df)
new_rows      = len(new_df)
combined_rows = original_rows + new_rows

# Recreate exact same test split from Task 1 (same test set)
X_orig = train_df[FEATURES]
y_orig = train_df["irrigation_hours"]
X_train_orig, X_test, y_train_orig, y_test = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42
)

# Combine: original 80% training portion + ALL new data
X_new = new_df[FEATURES]
y_new = new_df["irrigation_hours"]
X_combined_train = pd.concat([X_train_orig, X_new], ignore_index=True)
y_combined_train = pd.concat([y_train_orig, y_new], ignore_index=True)

# Re-evaluate champion on the same test set (for fair comparison)
champion_version = meta["champion_version"]
champion_model_uri = f"models:/{REGISTERED_MODEL_NAME}/{champion_version}"
champion_model = mlflow.sklearn.load_model(champion_model_uri)
champion_pred = champion_model.predict(X_test)
champion_mae_on_test = float(mean_absolute_error(y_test, champion_pred))

# Retrain with combined data
with mlflow.start_run(run_name=f"{best_model_name}_retrained") as run:
    mlflow.set_tag("experiment_type", "retraining")

    if best_model_name == "RandomForest":
        params = {"n_estimators": 100, "random_state": 42}
        model = RandomForestRegressor(**params)
    else:
        params = {"fit_intercept": True}
        model = LinearRegression(**params)

    mlflow.log_params(params)
    mlflow.log_param("original_data_rows", original_rows)
    mlflow.log_param("new_data_rows", new_rows)
    mlflow.log_param("combined_data_rows", combined_rows)

    model.fit(X_combined_train, y_combined_train)
    y_pred = model.predict(X_test)

    retrained_mae  = float(mean_absolute_error(y_test, y_pred))
    retrained_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    retrained_r2   = float(r2_score(y_test, y_pred))

    mlflow.log_metric("mae",  retrained_mae)
    mlflow.log_metric("rmse", retrained_rmse)
    mlflow.log_metric("r2",   retrained_r2)
    mlflow.sklearn.log_model(model, "model")
    retrained_run_id = run.info.run_id

improvement = champion_mae_on_test - retrained_mae

if improvement > 0:
    model_uri = f"runs:/{retrained_run_id}/model"
    mv_new = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    time.sleep(2)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS, str(mv_new.version))
    action = "promoted"
else:
    action = "kept_champion"

result = {
    "original_data_rows": original_rows,
    "new_data_rows": new_rows,
    "combined_data_rows": combined_rows,
    "champion_mae": round(champion_mae_on_test, 4),
    "retrained_mae": round(retrained_mae, 4),
    "improvement": round(improvement, 4),
    "min_improvement_threshold": 0,
    "action": action,
    "comparison_metric": "mae",
}

with open(os.path.join(RESULTS_DIR, "step4_s8.json"), "w") as f:
    json.dump(result, f, indent=2)

print("Task 4 complete.")
print(json.dumps(result, indent=2))
