import pandas as pd
import numpy as np
import optuna
from utils import smape
import os

print(" - Starting Ensemble Script  -")

##  - 1. Define File Paths  -
OOF_TEXT_CPU_PATH = 'submissions/oof_text_preds_cpu.csv'
OOF_TEXT_GPU_PATH = 'submissions/oof_text_preds_gpu.csv'
OOF_VLM_PATH = 'submissions/oof_vlm_preds.csv'

TEST_TEXT_CPU_PATH = 'submissions/submission_text_cpu.csv'
TEST_TEXT_GPU_PATH = 'submissions/submission_text_gpu.csv'
TEST_VLM_PATH = 'submissions/submission_vlm_only.csv'

##  - 2. Load and Combine Text Model Predictions  -
print("Loading and combining text model predictions...")
oof_text_cpu_exists = os.path.exists(OOF_TEXT_CPU_PATH)
oof_text_gpu_exists = os.path.exists(OOF_TEXT_GPU_PATH)

## Process OOF (validation) files
if oof_text_cpu_exists and oof_text_gpu_exists:
    print("Found both CPU and GPU text OOF predictions. Averaging them.")
    oof_text_cpu = pd.read_csv(OOF_TEXT_CPU_PATH)
    oof_text_gpu = pd.read_csv(OOF_TEXT_GPU_PATH)
    oof_text = pd.merge(oof_text_cpu, oof_text_gpu, on='sample_id')
    oof_text['text_pred'] = (oof_text['text_pred_cpu'] + oof_text['text_pred_gpu']) / 2
    oof_text = oof_text[['sample_id', 'text_pred']]
elif oof_text_gpu_exists:
    print("Found only GPU text OOF predictions.")
    oof_text = pd.read_csv(OOF_TEXT_GPU_PATH).rename(columns={'text_pred_gpu': 'text_pred'})
else:
    print("Found only CPU text OOF predictions.")
    oof_text = pd.read_csv(OOF_TEXT_CPU_PATH).rename(columns={'text_pred_cpu': 'text_pred'})

## Process Test prediction files
if os.path.exists(TEST_TEXT_CPU_PATH) and os.path.exists(TEST_TEXT_GPU_PATH):
    print("Found both CPU and GPU text test predictions. Averaging them.")
    test_text_cpu = pd.read_csv(TEST_TEXT_CPU_PATH)
    test_text_gpu = pd.read_csv(TEST_TEXT_GPU_PATH)
    test_text = pd.merge(test_text_cpu, test_text_gpu, on='sample_id')
    test_text['price'] = (test_text['price_x'] + test_text['price_y']) / 2
    test_text = test_text[['sample_id', 'price']]
elif os.path.exists(TEST_TEXT_GPU_PATH):
    print("Found only GPU text test predictions.")
    test_text = pd.read_csv(TEST_TEXT_GPU_PATH)
else:
    print("Found only CPU text test predictions.")
    test_text = pd.read_csv(TEST_TEXT_CPU_PATH)

## Load VLM predictions
print("Loading VLM predictions...")
oof_vlm = pd.read_csv(OOF_VLM_PATH)
test_vlm = pd.read_csv(TEST_VLM_PATH)

##  - 3. Prepare Data for Optimization  -
train_df = pd.read_csv('data/train.csv')
oof_df = pd.merge(train_df[['sample_id', 'price']], oof_text, on='sample_id')
oof_df = pd.merge(oof_df, oof_vlm, on='sample_id')

## Inverse log transform to get back to original price scale for SMAPE calculation
oof_df['text_pred'] = np.expm1(oof_df['text_pred'])
oof_df['vlm_pred'] = np.expm1(oof_df['vlm_pred'])

##  - 4. Find Optimal Ensemble Weights using Optuna  -
def objective(trial):
    w_text = trial.suggest_float('w_text', 0.0, 1.0)
    w_vlm = 1.0 - w_text
    final_pred = w_text * oof_df['text_pred'] + w_vlm * oof_df['vlm_pred']
    return smape(oof_df['price'], final_pred)

print("\nOptimizing ensemble weights with Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_weight = study.best_params['w_text']
print(f"\nOptimal weight for text model: {best_weight:.4f}")
print(f"Optimal weight for VLM model: {1 - best_weight:.4f}")
print(f"Best OOF SMAPE from ensemble: {study.best_value:.4f}")

##  - 5. Create Final Submission  -
print("\nCreating final submission file...")
final_submission = pd.DataFrame()
final_submission['sample_id'] = test_text['sample_id']
final_submission['price'] = (best_weight * test_text['price']) + ((1.0 - best_weight) * test_vlm['price'])

final_submission['price'] = final_submission['price'].clip(0)

final_submission.to_csv('submissions/final_ensemble_submission.csv', index=False)
print("Final ensemble submission created at 'submissions/final_ensemble_submission.csv'")