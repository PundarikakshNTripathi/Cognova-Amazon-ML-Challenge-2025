"""Advanced ensembling of text and image models with Optuna + MLflow.

This script:
- Loads OOF predictions for available models and aligns them with train labels
- Optimizes non-negative weights (softmax) to minimize SMAPE on OOF
- Applies the same weights to test submissions and writes a final blended CSV
- Logs metrics and artifacts to MLflow for reproducibility

Notes:
- Detects and inverse-transforms log1p OOFs automatically (expm1 heuristic)
- Test predictions are assumed to be on the original price scale
"""

import os
import numpy as np
import pandas as pd
import optuna
import mlflow
from datetime import datetime

## Config: point these to your files if names differ
SUB_DIR = "submissions"
TRAIN_CSV = "data/train.csv"

## Known model files (edit if your filenames differ)
## The loader will auto-detect id/pred columns.
MODEL_SPECS = [
    ## Text models
    {"name": "text_cpu", "oof": f"{SUB_DIR}/oof_text_preds_cpu.csv", "sub": f"{SUB_DIR}/submission_text_cpu.csv"},
    {"name": "text_gpu", "oof": f"{SUB_DIR}/oof_text_preds_gpu.csv", "sub": f"{SUB_DIR}/submission_text_gpu.csv"},
    ## CNN image model you just trained
    {"name": "image_cnn", "oof": f"{SUB_DIR}/oof_image_cnn.csv", "sub": f"{SUB_DIR}/submission_image_cnn.csv"},
    ## Optional VLM (if you have OOF ready)
    {"name": "vlm", "oof": f"{SUB_DIR}/oof_vlm_preds.csv", "sub": f"{SUB_DIR}/submission_vlm_only.csv"},
]

EXPERIMENT = "Amazon-Price-Prediction"
RUN_NAME = f"ENSEMBLE_Advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

##      utils     
def smape(y_true, y_pred):
    """Compute SMAPE in percentage; robust to zeros via small epsilon."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true) / (denom + 1e-8)
    diff[denom == 0] = 0.0
    return float(np.mean(diff) * 100.0)

def detect_id_col(cols):
    """Identify the id column among common variants."""
    for c in ["sample_id", "id", "Id", "sampleId"]:
        if c in cols: return c
    raise KeyError("ID column not found (expected one of: sample_id, id, Id, sampleId)")

def detect_pred_col(df, exclude):
    """Choose a reasonable prediction column (price/predicted_price/etc.)."""
    ## Try common names
    for c in ["price", "predicted_price", "prediction", "vlm_pred"]:
        if c in df.columns and c not in exclude: return c
    ## Else pick the first numeric column not in exclude
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise KeyError("No numeric prediction column found.")
    return num_cols[0]

def load_oof(spec, id_key="id"):
    """Load OOF CSV; return DataFrame with standardized id and model-named column.

    If the OOF values look like log1p scale (typical range and median), convert with expm1.
    """
    path = spec["oof"]
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    id_col = detect_id_col(df.columns)
    pred_col = detect_pred_col(df, exclude=[id_col])
    ## Convert to original price scale if OOF appears to be in log1p scale
    s = pd.to_numeric(df[pred_col], errors="coerce")
    ## Heuristic: values typically in [0, ~25] and median < 10 indicate log1p for prices up to ~10k
    if s.notna().mean() > 0.99 and s.median() < 10 and s.max() < 25:
        s = np.expm1(s)
    out = pd.DataFrame({id_key: df[id_col], spec["name"]: s})
    return out

def load_sub(spec, id_key="id"):
    """Load test submission CSV and standardize columns (id + model name)."""
    path = spec["sub"]
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    id_col = detect_id_col(df.columns)
    pred_col = detect_pred_col(df, exclude=[id_col])
    out = df[[id_col, pred_col]].copy()
    out.columns = [id_key, spec["name"]]
    return out

def softmax(x):
    x = np.array(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

##      load data     
train = pd.read_csv(TRAIN_CSV)
id_col = detect_id_col(train.columns)
target_col = "price"
assert target_col in train.columns, "price target not found in train.csv"

## merge OOFs
oof_frames = []
used_specs = []
for spec in MODEL_SPECS:
    oof = load_oof(spec, id_key="id")
    if oof is None:
        continue
    oof_frames.append(oof)
    used_specs.append(spec)

if not oof_frames:
    raise RuntimeError("No OOF files found. Provide at least one OOF to fit ensemble weights.")

oof_merged = oof_frames[0]
for f in oof_frames[1:]:
    oof_merged = oof_merged.merge(f, on="id", how="inner")

## attach ground truth
train_truth = train[[id_col, target_col]].copy()
train_truth.columns = ["id", "price"]
oof_merged = oof_merged.merge(train_truth, on="id", how="inner")

## drop models with constant or near-constant OOF (e.g., zero stubs)
model_cols = [spec["name"] for spec in used_specs if spec["name"] in oof_merged.columns]
kept = []
for c in model_cols:
    if np.nanstd(oof_merged[c].values) < 1e-6:
        print(f"Skipping model '{c}' from training (constant OOF).")
    else:
        kept.append(c)
model_cols = kept
if len(model_cols) == 0:
    raise RuntimeError("All OOFs are constant. Need at least one valid OOF to ensemble.")

## drop rows with any NaN across chosen models or target
oof_train = oof_merged.dropna(subset=model_cols + ["price"]).reset_index(drop=True)

X = oof_train[model_cols].values.astype(np.float64)
y = oof_train["price"].values.astype(np.float64)

##      Optuna objective (non-negative weights, sum=1 via softmax)     
def objective(trial):
    """Optuna objective: softmax-normalized weights to minimize OOF SMAPE."""
    raw = [trial.suggest_float(f"w_{c}", 0.0, 2.0) for c in model_cols]
    w = softmax(raw)
    pred = X.dot(w)
    return smape(y, pred)

##      MLflow + Optuna run     
mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name=RUN_NAME) as run:
    mlflow.log_params({
        "models": ",".join(model_cols),
        "n_samples": len(X),
        "optuna_trials": 80
    })
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=80, show_progress_bar=False)

    best_raw = [study.best_trial.params[f"w_{c}"] for c in model_cols]
    best_w = softmax(best_raw)
    oof_pred = X.dot(best_w)
    best_smape = smape(y, oof_pred)

    ## log
    mlflow.log_metric("ensemble_oof_smape", best_smape)
    for c, w in zip(model_cols, best_w):
        mlflow.log_param(f"weight_{c}", float(w))

    print(f"Selected models: {model_cols}")
    print(f"Best weights: {dict(zip(model_cols, map(float, best_w)))}")
    print(f"OOF SMAPE: {best_smape:.4f}% on {len(y)} samples")

    ##      build test blend     
    ## Collect only the models that were kept during OOF optimization
    sub_frames = []
    for spec in used_specs:
        s = load_sub(spec, id_key="id")
        if s is None or spec["name"] not in model_cols:
            continue
        sub_frames.append(s)
    if not sub_frames:
        raise RuntimeError("No test submissions found for the selected models.")

    sub_merged = sub_frames[0]
    for f in sub_frames[1:]:
        sub_merged = sub_merged.merge(f, on="id", how="outer")

    ## align to test ids present in all available subs; fill missing with row-median
    model_cols_test = [c for c in model_cols if c in sub_merged.columns]
    M = sub_merged[model_cols_test].copy()
    row_med = M.median(axis=1)
    for c in model_cols_test:
        M[c] = M[c].fillna(row_med)
    W = np.array([best_w[model_cols.index(c)] for c in model_cols_test], dtype=np.float64)
    W = W / (W.sum() + 1e-12)

    test_pred = M.values.dot(W)
    out = pd.DataFrame({"id": sub_merged["id"], "price": np.clip(test_pred, 0.01, 10000.0)})
    ## restore original id column name
    out.columns = [id_col, "price"]
    out = out.sort_values(by=id_col).reset_index(drop=True)

    os.makedirs(SUB_DIR, exist_ok=True)
    out_path = os.path.join(SUB_DIR, "submission_ensemble_advanced.csv")
    out.to_csv(out_path, index=False)
    mlflow.log_artifact(out_path)
    print(f"Saved: {out_path}")