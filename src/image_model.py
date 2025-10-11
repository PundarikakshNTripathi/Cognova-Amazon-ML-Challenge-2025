# src/image_model.py
# NOTE: This script is designed for a GPU environment (local NVIDIA GPU or Google Colab).

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
BATCH_SIZE = 16 # Process images in batches for massive speedup on GPU
IS_DEBUG = False # Set to False for the full run

# --- 1. Model and Data Setup ---
print("Loading VLM model (moondream2)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == 'cpu':
    print("WARNING: No GPU found. This script will be extremely slow on CPU.")

model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Use torch.float16 for half-precision to reduce memory usage on your GPU
moondream_model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16
).to(device)
moondream_model.eval()

print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

if IS_DEBUG:
    # Use a small sample for quick debugging
    train_df = train_df.sample(n=32, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(n=32, random_state=42).reset_index(drop=True)

print("Downloading images (will skip if already downloaded)...")
download_images(pd.concat([train_df, test_df]), image_dir=IMAGE_DIR)

# --- 2. VLM Inference with Caching and Batching ---
def get_price_from_vlm_batch(image_paths, model, tokenizer):
    """Processes a batch of images and returns price predictions."""
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            # Ensure image is in RGB format, which the model expects
            images.append(Image.open(path).convert("RGB"))
            valid_paths.append(path)
        except Exception:
            # Handle broken or missing images gracefully
            pass

    if not images:
        return {}

    enc_images = model.encode_image(images).to(device)
    prompts = ["What is the price of this product?" for _ in images]
    
    answers = model.answer_question(enc_images, prompts, tokenizer)
    
    results = {}
    for i, answer in enumerate(answers):
        price_match = re.search(r'[\d\.]+', answer)
        price = float(price_match.group(0)) if price_match else np.nan
        results[valid_paths[i]] = price
        
    return results

def process_dataframe(df, desc, cache_file):
    """Generates predictions for a dataframe with caching to resume if interrupted."""
    os.makedirs('submissions', exist_ok=True)
    if os.path.exists(cache_file):
        print(f"Loading cached predictions from {cache_file}")
        preds_df = pd.read_csv(cache_file)
        preds_dict = pd.Series(preds_df['price'].values, index=preds_df['image_path']).to_dict()
    else:
        preds_dict = {}

    image_links = df['image_link'].tolist()
    image_paths = [os.path.join(IMAGE_DIR, os.path.basename(link)) for link in image_links]
    
    paths_to_process = [p for p in image_paths if p not in preds_dict]
    
    if not paths_to_process:
        print("All predictions found in cache.")
    else:
        print(f"Found {len(paths_to_process)} new images to process.")
        for i in tqdm(range(0, len(paths_to_process), BATCH_SIZE), desc=desc):
            batch_paths = paths_to_process
            batch_results = get_price_from_vlm_batch(batch_paths, moondream_model, tokenizer)
            preds_dict.update(batch_results)
            
            # Save progress after each batch - this is your safety net
            temp_df = pd.DataFrame(list(preds_dict.items()), columns=['image_path', 'price'])
            temp_df.to_csv(cache_file, index=False)
            
    final_preds = [preds_dict.get(p, np.nan) for p in image_paths]
    return final_preds

# --- 3. Generate Predictions and Save ---
with start_mlflow_run(EXPERIMENT_NAME, RUN_NAME) as run:
    mlflow.log_params({"batch_size": BATCH_SIZE, "device": device})

    vlm_oof_preds = process_dataframe(train_df, "VLM OOF Inference", "submissions/cache_vlm_oof.csv")
    vlm_test_preds = process_dataframe(test_df, "VLM Test Inference", "submissions/cache_vlm_test.csv")

    # Impute missing predictions with the mean of successful ones
    vlm_oof_series = pd.Series(vlm_oof_preds).replace(0, np.nan)
    vlm_test_series = pd.Series(vlm_test_preds).replace(0, np.nan)
    mean_pred = vlm_oof_series.mean()
    vlm_oof_preds = vlm_oof_series.fillna(mean_pred).values
    vlm_test_preds = vlm_test_series.fillna(mean_pred).values
    
    vlm_model_smape = smape(train_df['price'], vlm_oof_preds)
    print(f"VLM Inference Model OOF SMAPE: {vlm_model_smape:.4f}")
    mlflow.log_metric("vlm_oof_smape", vlm_model_smape)

    # --- 4. Save Final Prediction Files ---
    print("Saving VLM model predictions...")
    
    pd.DataFrame({'sample_id': train_df['sample_id'], 'vlm_pred': np.log1p(vlm_oof_preds)}).to_csv('submissions/oof_vlm_preds.csv', index=False)
    
    submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': vlm_test_preds})
    submission_df['price'] = submission_df['price'].clip(0)
    submission_df.to_csv('submissions/submission_vlm_only.csv', index=False)
    mlflow.log_artifact('submissions/submission_vlm_only.csv')

print("\nVLM model script finished.")