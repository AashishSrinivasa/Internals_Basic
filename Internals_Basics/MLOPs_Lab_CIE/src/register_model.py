"""
Task 2 — Model Versioning
Registers the best model from Task 1 into the MLflow Model Registry, saves step2_s6.json
"""
import os
import json
import time
import mlflow
from mlflow import MlflowClient

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REGISTERED_MODEL_NAME = "cropsense-irrigation-hours-predictor"

# Load best run metadata from Task 1
with open(os.path.join(MODELS_DIR, "best_run_info.json")) as f:
    meta = json.load(f)

mlflow.set_tracking_uri(meta["tracking_uri"])
client = MlflowClient()

best_run_id = meta["best_run_id"]
run = client.get_run(best_run_id)
mae = run.data.metrics["mae"]

# Register best model
model_uri = f"runs:/{best_run_id}/model"
mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

# Wait briefly for registration to settle
time.sleep(2)

result = {
    "registered_model_name": REGISTERED_MODEL_NAME,
    "version": int(mv.version),
    "run_id": best_run_id,
    "source_metric": "mae",
    "source_metric_value": round(mae, 4),
}

# Persist version info for Task 3
meta["v1_version"] = int(mv.version)
meta["v1_mae"] = mae
with open(os.path.join(MODELS_DIR, "best_run_info.json"), "w") as f:
    json.dump(meta, f, indent=2)

with open(os.path.join(RESULTS_DIR, "step2_s6.json"), "w") as f:
    json.dump(result, f, indent=2)

print("Task 2 complete.")
print(json.dumps(result, indent=2))
