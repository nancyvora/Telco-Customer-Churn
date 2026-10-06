# Telco Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to leave, and serves those predictions through a REST API and a web form.

It covers the full path from raw data to a running service: exploration, data validation, feature engineering, model training with experiment tracking, and a containerised app.

## Why churn prediction

Keeping a customer is usually cheaper than winning a new one. If a company can tell in advance who is likely to leave, it can step in with a retention offer before that happens.

The cost of mistakes is uneven. Missing a customer who then leaves is expensive; sending an offer to a customer who would have stayed is cheap. The model is therefore tuned to catch as many churners as possible (high recall), accepting that some loyal customers get flagged too.

## Results

The training pipeline produces an XGBoost model with these results on a held-out test set of 1,409 customers (374 churners), using a decision threshold of 0.35:

| Metric | Value | Meaning |
|---|---|---|
| Recall | 0.824 | 82% of the customers who churned were caught |
| Precision | 0.483 | About half of the flagged customers really churned |
| F1 | 0.609 | Balance of the two |
| ROC AUC | 0.836 | How well the model ranks customers by risk |

Three models were compared in the notebook at the same threshold (0.30) and the same train/test split:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Random Forest | 0.532 | 0.725 | 0.614 |
| LightGBM | 0.498 | 0.818 | 0.619 |
| XGBoost | 0.487 | 0.821 | 0.611 |

Both boosting models clearly beat Random Forest on recall. XGBoost and LightGBM are effectively tied, and XGBoost was carried forward. The reasoning and the trade-offs are written up in [notebooks/EDA.ipynb](notebooks/EDA.ipynb).

## How it works

```
Training:  CSV -> validate -> preprocess -> build features -> train XGBoost -> log to MLflow
Serving:   request -> same feature transformations -> model -> "Likely to churn" / "Not likely to churn"
```

| Stage | Tool | Where |
|---|---|---|
| Exploration and model comparison | pandas, scikit-learn, LightGBM, XGBoost, Optuna | `notebooks/EDA.ipynb` |
| Data validation | Great Expectations (25 checks) | `src/utils/validate_data.py` |
| Preprocessing and features | pandas | `src/data/`, `src/features/` |
| Training and tracking | XGBoost, MLflow | `scripts/run_pipeline.py` |
| Prediction logic | MLflow model loading | `src/serving/inference.py` |
| API and web form | FastAPI, Gradio | `src/app/main.py` |
| Packaging | Docker | `dockerfile` |
| CI | GitHub Actions, Docker Hub | `.github/workflows/ci.yml` |

## Getting started

Requires Python 3.11 or 3.12. Commands are shown for Windows; on macOS or Linux, activate the environment with `source .venv/bin/activate`.

### 1. Install

```bash
git clone <this-repo-url>
cd Telco-Customer-Churn
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the data

The dataset is not stored in this repository. Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and save it as:

```
data/raw/Telco-Customer-Churn.csv
```

### 3. Train the model

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
```

This validates the data, trains the model, prints the metrics, and logs the run to the `mlruns/` folder. Optional flags: `--threshold` (default 0.35) and `--test_size` (default 0.2).

To browse the logged runs:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### 4. Run the app

```bash
python -m uvicorn src.app.main:app --port 8000
```

| URL | What it is |
|---|---|
| http://localhost:8000/ui | Web form for trying predictions |
| http://localhost:8000/docs | Interactive API documentation |
| http://localhost:8000/ | Health check |

On startup the app prints a `Failed to load model from /app/model` message and then a `Fallback: Loaded model from ...` message. This is expected outside Docker: it uses the most recent model you trained, or the model bundled in `src/serving/model/` if you have not trained one.

To call the API from a second terminal while the app is running:

```bash
python scripts/test_fastapi.py
```

### 5. Run with Docker (optional)

```bash
docker build -t telco-churn-app .
docker run -p 8000:8000 telco-churn-app
```

The image serves the model bundled in `src/serving/model/`, so it does not need the dataset or a training run.

## Continuous integration

Every push to `main` triggers a GitHub Actions workflow that builds the Docker image and pushes it to Docker Hub as `ncyvora/telco-fastapi:latest`. It needs two repository secrets: `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.

## Project structure

```
├── notebooks/EDA.ipynb          Exploration, model comparison and tuning
├── scripts/
│   ├── run_pipeline.py          Full training pipeline
│   ├── prepare_processed_data.py
│   └── test_*.py                Manual test scripts
├── src/
│   ├── app/                     FastAPI app and Gradio form
│   ├── data/                    Loading and preprocessing
│   ├── features/                Feature engineering
│   ├── models/                  Training, tuning and evaluation helpers
│   ├── serving/                 Prediction logic and the bundled model
│   └── utils/                   Data validation
├── dockerfile
└── requirements.txt
```

## Problems faced and how they were solved

These are the issues hit while getting the project running on Windows, and what fixed each one.

### MLflow rejected the tracking path on Windows

- **Symptom**: `MlflowException: file://c:\Users\... is not a valid remote uri`.
- **Cause**: The path was built as `f"file://{project_root}/mlruns"`. On Windows this puts the drive letter where a `file://` address expects a machine name, so MLflow read `c:` as a remote computer. It works on macOS and Linux only because paths there start with a slash.
- **Fix**: Build the address with `Path(project_root, "mlruns").as_uri()`, which produces a correct `file:///C:/...` address on any operating system.

### Predictions ignored most of the customer's details

- **Symptom**: The app returned predictions without errors, but the same customer got different answers from two models trained the same way.
- **Cause**: The serving code one-hot encoded each request with `drop_first=True`, as training does. Training sees thousands of rows, so dropping the first category is harmless. A single request has only one category per column, so that one was dropped and every one-hot feature became 0. The model never saw the contract type, internet service or payment method.
- **Fix**: Encode requests without `drop_first` and let the existing column alignment remove the baseline categories. A check on 300 real customers showed the serving features matched the training features for none of them before the fix and for all of them after.
- **Lesson**: A service that responds is not the same as a service that is right. Comparing serving features against training features on real rows is what exposed this.

### The training pipeline crashed at data validation

- **Symptom**: `AttributeError: module 'great_expectations' has no attribute 'dataset'`.
- **Cause**: The validation code used an older Great Expectations interface that no longer exists in the installed version (1.5.8).
- **Fix**: Rewrote the 25 checks using the current interface. `TotalCharges` is stored as text in the CSV, so it is now converted to a number before its range checks run.

### The app would not start because of mismatched Gradio packages

- **Symptom**: `ModuleNotFoundError: No module named 'gradio_client.serializing'`.
- **Cause**: `gradio` was not pinned in `requirements.txt`. Other pinned packages forced an old `gradio`, while its companion package `gradio_client` installed at the newest version, and the two were incompatible.
- **Fix**: Pinned a matching pair, `gradio==3.36.1` and `gradio_client==0.2.10`.

### The model could not be found outside Docker

- **Symptom**: `Failed to load model: No such file or directory: '\app\model'. Fallback failed: No model found in local mlruns`.
- **Cause**: The model path `/app/model` only exists inside the Docker image. The local fallback searched relative to the folder the app was started from, and could also pick up a notebook model that lacked the feature list the app needs.
- **Fix**: The fallback now searches relative to the code file, considers both `mlruns/` and the bundled model, and skips any model without a feature list.

### `python app.py` did not start anything

- **Symptom**: `can't open file 'app.py': No such file or directory`.
- **Cause**: The app file lives in `src/app/`, and it only defines the app. It does not start a web server.
- **Fix**: Start it through the server, from the project root: `python -m uvicorn src.app.main:app --port 8000`.

### A file path from another computer was hardcoded

- **Symptom**: A test script failed with a file-not-found error.
- **Cause**: The data path was an absolute path from a different machine.
- **Fix**: Replaced it with a path relative to the project root.

### Emoji in log messages crashed scripts on Windows

- **Symptom**: `UnicodeEncodeError` when a script printed status messages.
- **Cause**: The Windows console defaults to an encoding that cannot represent emoji.
- **Fix**: The scripts set their output to UTF-8 at startup.

### Tuning for recall alone made the model worse overall

- **Symptom**: Optuna raised recall from 0.82 to 0.94, but precision fell to 0.42 and F1 dropped.
- **Cause**: The tuning objective rewarded recall only, so the search found a model that simply flags more customers: 46 more churners caught at the cost of 172 more loyal customers flagged.
- **Outcome**: Documented as a trade-off in the notebook. The pipeline uses a more balanced configuration.

## Known limitations

- **`SeniorCitizen` is not collected by the API or the form**, although the model uses it, so it is always treated as 0 at prediction time.
- **Results come from a single train/test split.** Small differences between models could change with a different split; cross-validation would give a firmer comparison.
- **The model bundled for Docker** in `src/serving/model/` comes from the original project, not from a fresh training run.
- **There is no automated test suite.** The scripts in `scripts/test_*.py` are run by hand.
