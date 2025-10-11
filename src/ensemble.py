import pandas as pd
import numpy as np
import optuna
from utils import smape
import os

print("Loading OOF predictions from individual models...")
oof_text = pd.read_csv('submissions/oof_text_preds.csv')
oof_vlm = pd.read_csv('submissions/oof_vlm_preds.csv')
train_df = pd.read_csv('data/train.csv')

# Merge predictions with ground truth
oof_df = pd.merge(train_df[['sample_id', 'price']], oof_text, on='sample_id')
oof_df = pd.merge(oof_df, oof_vlm, on='sample_id')

# Inverse log transform to get back to original price scale
oof_df['text_pred'] = np.expm1(oof_df['text_pred'])
oof_df['vlm_pred'] = np.expm1(oof_df['vlm_pred'])

# --- Find Optimal Ensemble Weights using Optuna ---
def objective(trial):
    # We define the search space for the weight of the text model
    w_text = trial.suggest_float('w_text', 0.0, 1.0)
    
    # The VLM weight is simply 1 - w_text
    w_vlm = 1 - w_text
    
    final_pred = w_text * oof_df['text_pred'] + w_vlm * oof_df['vlm_pred']
    return smape(oof_df['price'], final_pred)

print("Optimizing ensemble weights with Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_weight = study.best_params['w_text']
print(f"Optimal weight for text model: {best_weight:.4f}")
print(f"Optimal weight for VLM model: {1 - best_weight:.4f}")
print(f"Best OOF SMAPE from ensemble: {study.best_value:.4f}")

# --- Create Final Submission ---
print("Loading test predictions...")
test_text = pd.read_csv('submissions/submission_text_only.csv')
test_vlm = pd.read_csv('submissions/submission_vlm_only.csv')

# Combine using the optimal weights
final_submission = pd.DataFrame()
final_submission['sample_id'] = test_text['sample_id']
final_submission['price'] = (best_weight * test_text['price']) + ((1 - best_weight) * test_vlm['price'])

# Ensure prices are positive
final_submission['price'] = final_submission['price'].clip(0)

final_submission.to_csv('submissions/final_ensemble_submission.csv', index=False)
print("Final ensemble submission created at 'submissions/final_ensemble_submission.csv'")