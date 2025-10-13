"""CPU text pipeline: SBERT embeddings + LightGBM/XGBoost.

Runs fully on CPU, caches embeddings to speed up re-runs, and writes OOF/test
predictions to the submissions directory for ensembling.
"""

# Optimized for a standard CPU; no special hardware required.

import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer
import mlflow
import os
from utils import smape, start_mlflow_run
from xgboost.callback import EarlyStopping

# --- Configuration ---
EXPERIMENT_NAME = "Amazon-Price-Prediction"
RUN_NAME = "CPU_Text_Ensemble_SBERT_LGBM_XGB"
N_SPLITS = 5
TEXT_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDINGS_FILE = 'data/text_embeddings.npy'

# --- 1. Data Loading and Preprocessing ---
print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
full_df = pd.concat([train_df.drop('price', axis=1), test_df], ignore_index=True)

print("Cleaning and Feature Engineering...")
train_df['price'] = np.log1p(train_df['price'])

def extract_ipq(text):
    match = re.search(r'(?:ipq|pack of|item pack quantity)[:\s]*(\d+)', str(text).lower())
    return int(match.group(1)) if match else 1.0
full_df['ipq'] = full_df['catalog_content'].apply(extract_ipq)

# --- 2. Feature Extraction (Sentence-BERT Embeddings with Caching) ---
if os.path.exists(EMBEDDINGS_FILE):
    print(f"Loading cached embeddings from {EMBEDDINGS_FILE}...")
    text_embeddings = np.load(EMBEDDINGS_FILE)
else:
    print(f"Creating text embeddings with '{TEXT_MODEL_NAME}'... (This will take a while on the first run)")
    text_model = SentenceTransformer(TEXT_MODEL_NAME)
    text_embeddings = text_model.encode(full_df['catalog_content'].astype(str).tolist(), show_progress_bar=True)
    print(f"Saving embeddings to {EMBEDDINGS_FILE} for future runs...")
    os.makedirs('data', exist_ok=True)
    np.save(EMBEDDINGS_FILE, text_embeddings)

X_full = np.hstack([text_embeddings, full_df[['ipq']].values])
X_train = X_full[:len(train_df)]
X_test = X_full[len(train_df):]
y_train = train_df['price'].values

# --- 3. Stratified Sampling for Robust Cross-Validation ---
num_bins = int(np.floor(1 + np.log2(len(train_df))))
train_df['price_bins'] = pd.cut(train_df['price'], bins=num_bins, labels=False)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# --- 4. Model Training (CPU) ---
with start_mlflow_run(EXPERIMENT_NAME, RUN_NAME) as run:
    mlflow.log_params({"n_splits": N_SPLITS, "text_model": TEXT_MODEL_NAME, "device": "cpu"})
    
    oof_preds_lgb, test_preds_lgb = np.zeros(len(train_df)), np.zeros(len(test_df))
    oof_preds_xgb, test_preds_xgb = np.zeros(len(train_df)), np.zeros(len(test_df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, train_df['price_bins'])):
        print(f"--- FOLD {fold+1}/{N_SPLITS} ---")
        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

        # LightGBM (CPU)
        print("  - Training LightGBM on CPU...")
        lgb_model = lgb.LGBMRegressor(random_state=42, n_estimators=2000, learning_rate=0.01, num_leaves=31, n_jobs=-1)
        lgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_preds_lgb[val_idx] = lgb_model.predict(X_val_fold)
        test_preds_lgb += lgb_model.predict(X_test) / N_SPLITS

        # XGBoost (CPU)
        print("  - Training XGBoost on CPU...")
        xgb_model = xgb.XGBRegressor(
            random_state=42, 
            n_estimators=2000, 
            learning_rate=0.01, 
            max_depth=7, 
            tree_method='hist', 
            n_jobs=-1
        )
        
        try:
            # Try the newer callback approach
            xgb_model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[EarlyStopping(rounds=100)],
                verbose=False
            )
        except TypeError:
            # Fallback to parameter-based early stopping
            xgb_model.set_params(early_stopping_rounds=100)
            xgb_model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
        oof_preds_xgb[val_idx] = xgb_model.predict(X_val_fold)
        test_preds_xgb += xgb_model.predict(X_test) / N_SPLITS

    # --- 5. Evaluation and Blending ---
    y_train_orig = np.expm1(y_train)
    smape_lgb = smape(y_train_orig, np.expm1(oof_preds_lgb))
    smape_xgb = smape(y_train_orig, np.expm1(oof_preds_xgb))
    print(f"\nLGBM OOF SMAPE: {smape_lgb:.4f}")
    print(f"XGB OOF SMAPE: {smape_xgb:.4f}")
    mlflow.log_metrics({"lgbm_oof_smape": smape_lgb, "xgb_oof_smape": smape_xgb})

    oof_preds_ensemble = (oof_preds_lgb + oof_preds_xgb) / 2
    test_preds_ensemble = (test_preds_lgb + test_preds_xgb) / 2
    
    ensemble_smape = smape(y_train_orig, np.expm1(oof_preds_ensemble))
    print(f"Text Ensemble OOF SMAPE: {ensemble_smape:.4f}")
    mlflow.log_metric("text_ensemble_oof_smape", ensemble_smape)

    # --- 6. Save Predictions ---
    print("Saving text model predictions...")
    os.makedirs('submissions', exist_ok=True)
    
    pd.DataFrame({'sample_id': train_df['sample_id'], 'text_pred_cpu': oof_preds_ensemble}).to_csv('submissions/oof_text_preds_cpu.csv', index=False)
    
    submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': np.expm1(test_preds_ensemble)})
    submission_df['price'] = submission_df['price'].clip(0)
    submission_df.to_csv('submissions/submission_text_cpu.csv', index=False)
    mlflow.log_artifact('submissions/submission_text_cpu.csv')

print("\nCPU Text model script finished successfully.")