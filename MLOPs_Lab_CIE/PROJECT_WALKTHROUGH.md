# CropSense Irrigation MLOps Project Walkthrough

## 1. What this project is about

This project builds a small MLOps pipeline for predicting `irrigation_hours` for a farm field.

The idea is:

1. Train more than one model.
2. Compare them using metrics.
3. Track everything in MLflow.
4. Register the best model in the MLflow Model Registry.
5. Create a challenger model and decide whether to promote it.
6. Retrain using new data and again decide whether to promote the retrained model.

So this project is not only about machine learning.
It is mainly about the full model lifecycle:

- experimentation
- tracking
- versioning
- promotion
- retraining

## 2. Problem statement

The model predicts how many hours of irrigation are needed based on these input features:

- `soil_moisture_pct`
- `crop_type_index`
- `field_size_hectares`
- `temperature_c`

Target column:

- `irrigation_hours`

## 3. Main tools and libraries used

From `requirements.txt`, the project uses:

- `mlflow` for experiment tracking, model registry, versioning, and alias management
- `scikit-learn` for machine learning models and evaluation
- `pandas` for reading and handling CSV data
- `numpy` for numerical calculations like RMSE

## 4. Important folders in the project

### `data/`

Contains the datasets:

- `training_data.csv`: original dataset used for training and testing
- `new_data.csv`: additional data used later in retraining

### `src/`

Contains the Python scripts that form the pipeline:

- `train.py`
- `register_model.py`
- `promote_model.py`
- `retrain.py`

### `results/`

Contains saved JSON outputs from each major task:

- `step1_s1.json`
- `step2_s6.json`
- `step3_s7.json`
- `step4_s8.json`

These are useful because they give the final answer of each step in a clean format.

### `models/`

Contains:

- `best_run_info.json`

This file is very important because it passes information from one script to the next, like:

- best model name
- best run ID
- MLflow tracking URI
- registered version
- champion version and MAE

### `mlruns/`

This is the local MLflow tracking directory.
It stores:

- experiment metadata
- run metadata
- metrics
- parameters
- logged models
- registered model versions
- alias information

## 5. Dataset summary

### Original training dataset

File: `data/training_data.csv`

- total rows: `25`
- columns: `5`

Columns:

- `soil_moisture_pct`
- `crop_type_index`
- `field_size_hectares`
- `temperature_c`
- `irrigation_hours`

### New dataset for retraining

File: `data/new_data.csv`

- total rows: `20`
- same columns as training data

### Combined dataset during retraining

- original rows: `25`
- new rows: `20`
- combined rows: `45`

## 6. Step-by-step pipeline explanation

## Step 1: Train and compare models

Script used:

- `src/train.py`

What this script does:

1. Sets the MLflow tracking URI to the local `mlruns` folder.
2. Creates or uses the MLflow experiment named `cropsense-irrigation-hours`.
3. Loads `data/training_data.csv`.
4. Selects the four feature columns.
5. Uses `train_test_split(test_size=0.2, random_state=42)`.
6. Trains two models:
   - `LinearRegression`
   - `RandomForestRegressor`
7. Evaluates both models using:
   - `MAE`
   - `RMSE`
   - `R2`
8. Logs parameters, metrics, tags, and model artifacts to MLflow.
9. Chooses the best model based on the lowest `MAE`.
10. Saves summary output to `results/step1_s1.json`.
11. Saves pipeline metadata to `models/best_run_info.json`.

Why MAE is important here:

- MAE tells the average prediction error in irrigation hours.
- Lower MAE means better predictions.
- This project uses MAE as the main decision metric.

### Step 1 actual results

From `results/step1_s1.json`:

- `LinearRegression`
  - MAE: `1.215`
  - RMSE: `1.3421`
  - R2: `0.4312`

- `RandomForest`
  - MAE: `1.1066`
  - RMSE: `1.2696`
  - R2: `0.491`

Best model selected:

- `RandomForest`

Reason:

- it has lower MAE than Linear Regression

Extra metadata saved:

- best run ID: `739db6bed8564cd8a39d9a95d2fbd105`
- tracking URI: `file:///Users/aashishsrinivasa/Documents/MLOPS/MLOPs_Lab_CIE/mlruns`

## Step 2: Register the best model in MLflow Model Registry

Script used:

- `src/register_model.py`

What this script does:

1. Reads `models/best_run_info.json`.
2. Gets the best run ID from Step 1.
3. Builds the model URI using `runs:/<run_id>/model`.
4. Registers that model in MLflow Model Registry.
5. Saves the registration result in `results/step2_s6.json`.
6. Updates `best_run_info.json` with the registered version and best MAE.

Registered model name:

- `cropsense-irrigation-hours-predictor`

### Step 2 actual results

From `results/step2_s6.json`:

- registered model name: `cropsense-irrigation-hours-predictor`
- registered version: `5`
- run ID: `739db6bed8564cd8a39d9a95d2fbd105`
- source metric: `mae`
- source metric value: `1.1066`

Important note:

- The version is `5`, not `1`.
- This means the registration script was likely run multiple times earlier.
- MLflow keeps increasing version numbers instead of restarting from 1.

## Step 3: Promote model using champion vs challenger logic

Script used:

- `src/promote_model.py`

What this script does:

1. Reads the current best model information from `best_run_info.json`.
2. Uses the registered model from Step 2 as the initial champion.
3. Assigns the MLflow alias `live` to that champion version.
4. Loads the same training dataset again.
5. Uses the same train/test split as Step 1.
6. Trains a challenger model.
7. For Random Forest, the challenger changes `random_state` from `42` to `99`.
8. Logs challenger metrics to MLflow.
9. Registers the challenger as a new model version.
10. Compares challenger MAE with champion MAE.
11. If challenger is better, alias `live` is moved to challenger.
12. If not, the old champion stays live.
13. Saves the decision in `results/step3_s7.json`.

Why alias is useful:

- The alias `live` acts like a pointer to the production model.
- Instead of hardcoding a version number, users can always refer to the `live` model.

### Step 3 actual results

From `results/step3_s7.json`:

- registered model name: `cropsense-irrigation-hours-predictor`
- alias name: `live`
- champion version: `5`
- challenger version: `6`
- action: `kept`

Meaning:

- version `5` stayed as the champion
- version `6` was created as challenger
- challenger did not beat the champion on MAE

Registry alias status:

From `mlruns/models/cropsense-irrigation-hours-predictor/meta.yaml`:

- alias `live` points to version `5`

## Step 4: Retrain using new data

Script used:

- `src/retrain.py`

What this script does:

1. Reads current best model metadata from `best_run_info.json`.
2. Loads:
   - `data/training_data.csv`
   - `data/new_data.csv`
3. Recreates the exact same test set from Step 1 using:
   - `test_size=0.2`
   - `random_state=42`
4. Keeps the original 20% test set unchanged for fair comparison.
5. Uses:
   - original 80% training portion
   - plus all rows from `new_data.csv`
6. Retrains the winning model type from Step 1, which is `RandomForest`.
7. Logs retraining parameters and metrics to MLflow.
8. Compares retrained model MAE against the champion MAE on the same test set.
9. Promotes only if there is improvement greater than `0`.
10. Saves the final decision in `results/step4_s8.json`.

Why this design is good:

- The test set is kept fixed.
- This makes the comparison fair.
- If the test set changed, the comparison would not be reliable.

### Step 4 actual results

From `results/step4_s8.json`:

- original data rows: `25`
- new data rows: `20`
- combined data rows: `45`
- champion MAE: `1.1066`
- retrained MAE: `1.3726`
- improvement: `-0.266`
- action: `kept_champion`

Meaning:

- the retrained model became worse
- MAE increased instead of decreasing
- so the retrained model was not promoted

## 7. Final project status

At the end of the full pipeline:

- best model type: `RandomForest`
- main evaluation metric: `MAE`
- registered model name: `cropsense-irrigation-hours-predictor`
- current live alias version: `5`
- challenger version `6` was not promoted
- retrained model also was not promoted

So the current production-like model in this project remains:

- `cropsense-irrigation-hours-predictor` version `5`

## 8. Files and their purpose in one line

- `src/train.py`: trains and compares baseline models
- `src/register_model.py`: registers the best run as a model version
- `src/promote_model.py`: creates a challenger and decides whether to promote it
- `src/retrain.py`: retrains the winning model using new data and checks if it should replace the champion
- `models/best_run_info.json`: stores metadata shared across pipeline stages
- `results/step1_s1.json`: model comparison result
- `results/step2_s6.json`: registration result
- `results/step3_s7.json`: promotion result
- `results/step4_s8.json`: retraining result
- `mlruns/`: MLflow backend storage for runs, models, metrics, and registry data

## 9. End-to-end flow in very simple words

If you want to explain it very simply in presentation:

1. I started with historical irrigation data.
2. I trained two models: Linear Regression and Random Forest.
3. I compared them using MAE, RMSE, and R2.
4. Random Forest performed better, so I selected it.
5. I tracked all experiments in MLflow.
6. I registered the best model in the MLflow Model Registry.
7. I created a challenger model and compared it against the current champion.
8. Since the challenger was not better, I kept the existing live model.
9. Then I added new data and retrained the model.
10. The retrained model also performed worse, so I again kept the existing champion.

## 10. What MLOps concepts you demonstrated

This project shows these MLOps concepts clearly:

- experiment tracking
- reproducible train/test split
- model comparison
- metric-based model selection
- model registry
- model versioning
- alias-based promotion
- challenger vs champion workflow
- retraining with new data
- promotion decision based on performance

## 11. Why the retrained model may have become worse

Possible reasons:

- the new dataset looks very different from the original dataset
- the new data has much larger field sizes and temperatures
- the original dataset is small, only `25` rows
- adding different data may reduce performance on the original test set
- the model may need tuning, not just more data

This is actually a useful presentation point:

- more data does not automatically guarantee a better model
- MLOps helps us test changes before blindly promoting them

## 12. Good points to say during presentation

You can say:

- "This project predicts irrigation hours from environmental and field features."
- "I used MLflow to track experiments and manage the model lifecycle."
- "I compared Linear Regression and Random Forest and selected the best model using MAE."
- "I registered the best model and used model versioning in MLflow."
- "I implemented champion-challenger logic using the `live` alias."
- "I also built a retraining step with new incoming data."
- "Because the challenger and retrained models performed worse, the existing champion stayed live."

## 13. One short presentation script

This project is an end-to-end MLOps workflow for irrigation prediction. First, I trained two machine learning models, Linear Regression and Random Forest, on farm-related data such as soil moisture, crop type, field size, and temperature. I tracked the experiments in MLflow and compared the models using MAE, RMSE, and R2. Random Forest performed best with an MAE of 1.1066, so I selected it as the best model.

Next, I registered this best model in the MLflow Model Registry under the name `cropsense-irrigation-hours-predictor`. Then I created a challenger model and compared it against the current champion using champion-challenger logic. Since the challenger did not improve the MAE, I kept the existing live model. After that, I retrained the winning model using new incoming data, but the retrained model performed worse, so it was also not promoted. This shows how MLOps helps manage the full model lifecycle and prevents weaker models from going into production.

## 14. Commands that likely ran in order

From the project structure, the workflow is designed to run in this order:

```bash
python src/train.py
python src/register_model.py
python src/promote_model.py
python src/retrain.py
```

## 15. Final conclusion

This project successfully demonstrates a complete local MLOps pipeline using MLflow and scikit-learn.

It shows:

- how to train and compare models
- how to register the best model
- how to manage versions
- how to use a live alias
- how to test challengers
- how to retrain with new data
- how to avoid promoting weaker models

The final live model is still version `5` of `cropsense-irrigation-hours-predictor`, because both the challenger and the retrained model performed worse than the current champion.
