# NOTE: This script is designed to be run in a GPU environment like Google Colab.

import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlflow
from utils import smape, start_mlflow_run, download_images

# --- Configuration ---
EXPERIMENT_NAME = "Amazon-Price-Prediction"
RUN_NAME = "VLM_Inference_Moondream2"
IMAGE_DIR = 'images/'
IS_DEBUG = False # Set to False on Colab to run on the full dataset

# --- 1. Model and Data Setup ---
print("Loading VLM model (moondream2)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
moondream_model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16
).to(device)
moondream_model.eval()

print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

if IS_DEBUG:
    train_df = train_df.sample(n=100, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(n=100, random_state=42).reset_index(drop=True)

print("Downloading images...")
download_images(pd.concat([train_df, test_df]), image_dir=IMAGE_DIR)

# --- 2. VLM Inference Function ---
def get_price_from_vlm(image_path, model, tokenizer):
    try:
        image = Image.open(image_path)
        enc_image = model.encode_image(image)
        prompt = "What is the price of this item? Answer with only a single numerical value."
        answer = model.answer_question(enc_image, prompt, tokenizer)
        price_match = re.search(r'[\d\.]+', answer)
        return float(price_match.group(0)) if price_match else np.nan
    except Exception:
        return np.nan

# --- 3. Generate Predictions and Track ---
with start_mlflow_run(EXPERIMENT_NAME, RUN_NAME) as run:
    vlm_test_preds = [get_price_from_vlm(os.path.join(IMAGE_DIR, link), moondream_model, tokenizer) for link in tqdm(test_df['image_link'], desc="VLM Inference on Test Set")]
    vlm_oof_preds = [get_price_from_vlm(os.path.join(IMAGE_DIR, link), moondream_model, tokenizer) for link in tqdm(train_df['image_link'], desc="VLM Inference on Train Set")]

    # Impute missing predictions with the mean of successful ones
    vlm_oof_series = pd.Series(vlm_oof_preds).replace(0, np.nan)
    vlm_test_series = pd.Series(vlm_test_preds).replace(0, np.nan)
    mean_pred = vlm_oof_series.mean()
    vlm_oof_preds = vlm_oof_series.fillna(mean_pred).values
    vlm_test_preds = vlm_test_series.fillna(mean_pred).values
    
    vlm_model_smape = smape(train_df['price'], vlm_oof_preds)
    print(f"VLM Inference Model OOF SMAPE: {vlm_model_smape:.4f}")
    mlflow.log_metric("vlm_oof_smape", vlm_model_smape)

    # --- 4. Save Predictions ---
    print("Saving VLM model predictions...")
    os.makedirs('submissions', exist_ok=True)
    
    # Save OOF predictions in log scale for consistency with the text model
    pd.DataFrame({'sample_id': train_df['sample_id'], 'vlm_pred': np.log1p(vlm_oof_preds)}).to_csv('submissions/oof_vlm_preds.csv', index=False)
    pd.DataFrame({'sample_id': test_df['sample_id'], 'price': vlm_test_preds}).to_csv('submissions/submission_vlm_only.csv', index=False)
    mlflow.log_artifact('submissions/submission_vlm_only.csv')

print("VLM model script finished. Download the 'submissions' folder.")