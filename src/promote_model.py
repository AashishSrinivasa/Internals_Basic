"""
Task 3 — Model Promotion
- Assigns alias "live" to version 1 (champion)
- Trains a challenger with random_state=99, registers as version 2
- Promotes if challenger MAE < champion MAE
- Saves step3_s7.json
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
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REGISTERED_MODEL_NAME = "cropsense-irrigation-hours-predictor"
ALIAS = "live"

with open(os.path.join(MODELS_DIR, "best_run_info.json")) as f:
    meta = json.load(f)

mlflow.set_tracking_uri(meta["tracking_uri"])
mlflow.set_experiment("cropsense-irrigation-hours")
client = MlflowClient()

best_model_name = meta["best_model"]
v1_version = meta["v1_version"]
v1_mae = meta["v1_mae"]

# Assign "live" alias to champion (version 1)
client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS, str(v1_version))

# Load data — same split as Task 1
df = pd.read_csv(DATA_PATH)
FEATURES = ["soil_moisture_pct", "crop_type_index", "field_size_hectares", "temperature_c"]
X = df[FEATURES]
y = df["irrigation_hours"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train challenger with random_state=99
with mlflow.start_run(run_name=f"{best_model_name}_challenger_rs99") as run:
    mlflow.set_tag("experiment_type", "baseline_comparison")

    if best_model_name == "RandomForest":
        params = {"n_estimators": 100, "random_state": 99}
        challenger = RandomForestRegressor(**params)
    else:
        params = {"fit_intercept": True}
        challenger = LinearRegression(**params)

    mlflow.log_params(params)
    challenger.fit(X_train, y_train)
    y_pred = challenger.predict(X_test)

    v2_mae  = float(mean_absolute_error(y_test, y_pred))
    v2_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    v2_r2   = float(r2_score(y_test, y_pred))

    mlflow.log_metric("mae", v2_mae)
    mlflow.log_metric("rmse", v2_rmse)
    mlflow.log_metric("r2", v2_r2)
    mlflow.sklearn.log_model(challenger, "model")
    challenger_run_id = run.info.run_id

# Register challenger as version 2
model_uri = f"runs:/{challenger_run_id}/model"
mv2 = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
time.sleep(2)
v2_version = int(mv2.version)

# Compare and decide
if v2_mae < v1_mae:
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS, str(v2_version))
    champion_version = v2_version
    action = "promoted"
else:
    champion_version = v1_version
    action = "kept"

result = {
    "registered_model_name": REGISTERED_MODEL_NAME,
    "alias_name": ALIAS,
    "champion_version": champion_version,
    "challenger_version": v2_version,
    "action": action,
}

# Persist champion info for Task 4
champion_mae = v2_mae if action == "promoted" else v1_mae
meta["champion_version"] = champion_version
meta["champion_mae"] = champion_mae
with open(os.path.join(MODELS_DIR, "best_run_info.json"), "w") as f:
    json.dump(meta, f, indent=2)

with open(os.path.join(RESULTS_DIR, "step3_s7.json"), "w") as f:
    json.dump(result, f, indent=2)

print("Task 3 complete.")
print(json.dumps(result, indent=2))
