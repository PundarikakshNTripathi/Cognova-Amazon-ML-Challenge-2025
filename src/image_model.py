"""VLM image pipeline (moondream2) with batching and caching.

Optional component for image-based predictions. Heavier than the fast CNN path
and not used in the final run due to runtime constraints, but kept for future work.
"""

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
import warnings
warnings.filterwarnings('ignore')

##  - Configuration  -
EXPERIMENT_NAME = "Amazon-Price-Prediction"
RUN_NAME = "VLM_Inference_Moondream2_CUDA129_Fixed"
IMAGE_DIR = 'images/'
IS_DEBUG = False

##  - Enhanced GPU Configuration  -
print("Loading VLM model (moondream2) with CUDA 12.9...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    cuda_version = torch.version.cuda
    
    print(f"GPU: {gpu_name}")
    print(f"CUDA Version: {cuda_version}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    BATCH_SIZE = 8  # Optimal for RTX 5060
    torch_dtype = torch.float16
    device_map = "auto"
else:
    print("Running on CPU")
    BATCH_SIZE = 4
    torch_dtype = torch.float32
    device_map = None

print(f"Batch size: {BATCH_SIZE}")

##  - Model Loading  -
model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

moondream_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    dtype=torch_dtype,
    device_map=device_map,
    low_cpu_mem_usage=True
)

if device_map is None:
    moondream_model = moondream_model.to(device)

moondream_model.eval()

if device == "cuda":
    torch.cuda.empty_cache()

##  - Data Loading  -
print(" Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

if IS_DEBUG:
    train_df = train_df.head(100)
    test_df = test_df.head(50)

print("⬇️ Downloading images...")
failed_downloads = download_images(pd.concat([train_df, test_df]), image_dir=IMAGE_DIR)
## download_images returns None (logs failures internally); avoid len(None) crash
print(" Image download step completed (see logs for any failures)")

##  - Enhanced Price Extraction  -
def extract_price_from_text(text):
    """Robust price extraction with validation"""
    if not text or pd.isna(text):
        return None
    
    text = str(text).lower().replace(',', '').replace('$', '')
    
    ## Multiple price patterns
    patterns = [
        r'price[:\s]*(\d+\.?\d*)',     # "price: 123.45"
        r'cost[:\s]*(\d+\.?\d*)',      # "cost: 123.45"  
        r'(\d+\.?\d*)\s*dollars?',     # "123.45 dollars"
        r'(\d+\.?\d*)\s*usd',          # "123.45 usd"
        r'(\d{1,4}\.\d{2})',           # XX.XX format
        r'(\d{1,4})',                  # Just integers
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                price = float(matches[0])
                if 0.01 <= price <= 10000:  # Reasonable bounds
                    return price
            except (ValueError, IndexError):
                continue
    
    return None

##  - FIXED VLM Processing  -
def get_price_from_vlm_batch(image_paths, model, tokenizer):
    """FIXED: Correct VLM API usage with error handling"""
    results = {}
    
    for path in image_paths:
        try:
            if not os.path.exists(path):
                results[path] = None
                continue
                
            ## Load image
            img = Image.open(path).convert("RGB")
            
            ## Check minimum size
            if img.size[0] < 50 or img.size[1] < 50:
                results[path] = None
                continue
                
            ## FIXED: Correct moondream2 API
            with torch.no_grad():
                prompt = "What is the price of this product? Please provide only the numerical price value in dollars."
                response = model.answer_question(img, prompt, tokenizer)  # ✅ Correct API
                price = extract_price_from_text(response)
                results[path] = price
                
        except Exception as e:
            results[path] = None
    
    return results

##  - ENHANCED Cache Processing with Outlier Cleaning  -
def clean_cached_predictions(preds_dict, reference_prices=None):
    """Clean existing cached predictions to remove outliers"""
    if not preds_dict:
        return preds_dict
    
    ## Extract valid predictions
    valid_preds = []
    for path, pred in preds_dict.items():
        if pred is not None and not pd.isna(pred) and pred > 0:
            valid_preds.append(pred)
    
    if len(valid_preds) < 10:  # Not enough data for outlier detection
        print("⚠️ Insufficient cached predictions for outlier detection")
        return preds_dict
    
    ## Calculate outlier bounds
    valid_array = np.array(valid_preds)
    q1, q3 = np.percentile(valid_array, [25, 75])
    iqr = q3 - q1
    lower_bound = max(0.01, q1 - 2.0 * iqr)  # More aggressive outlier detection
    upper_bound = min(10000, q3 + 2.0 * iqr)
    median_price = np.median(valid_array)
    
    ## Clean predictions
    cleaned_dict = {}
    outlier_count = 0
    
    for path, pred in preds_dict.items():
        if pred is None or pd.isna(pred) or pred <= 0:
            cleaned_dict[path] = None  # Keep as missing
        elif pred < lower_bound or pred > upper_bound:
            cleaned_dict[path] = median_price  # Replace outlier with median
            outlier_count += 1
        else:
            cleaned_dict[path] = pred  # Keep valid prediction
    
    if outlier_count > 0:
        print(f" Cleaned {outlier_count} outliers from cached predictions")
        print(f" Valid price range: ${lower_bound:.2f} - ${upper_bound:.2f}")
        print(f" Median replacement: ${median_price:.2f}")
    
    return cleaned_dict

def process_dataframe(df, desc, cache_file):
    """ENHANCED: Process with cache cleaning and outlier handling"""
    os.makedirs('submissions', exist_ok=True)
    
    ## Load existing cache
    if os.path.exists(cache_file):
        print(f" Loading cached predictions from {cache_file}")
        try:
            preds_df = pd.read_csv(cache_file)
            preds_dict = pd.Series(preds_df['price'].values, index=preds_df['image_path']).to_dict()
            print(f"✅ Loaded {len(preds_dict)} cached predictions")
            
            ## CRITICAL: Clean existing cached outliers
            print(" Cleaning cached predictions for outliers...")
            preds_dict = clean_cached_predictions(preds_dict)
            
        except Exception as e:
            print(f"⚠️ Cache loading error: {e}, starting fresh")
            preds_dict = {}
    else:
        preds_dict = {}

    ## FIXED: Robust image path generation
    ## Prefer images saved as sample_id.jpg; fall back to basename from image_link if that exists
    def resolve_image_path(row):
        ## candidate 1: sample_id.jpg
        sid = row['sample_id'] if 'sample_id' in row.index else row.get('id')
        cand1 = os.path.join(IMAGE_DIR, f"{sid}.jpg") if sid is not None else None
        ## candidate 2: basename from image_link (strip query params)
        link = str(row.get('image_link', ''))
        basename = os.path.basename(link.split('?')[0]) if link else ''
        cand2 = os.path.join(IMAGE_DIR, basename) if basename else None
        ## optional candidate 3: if basename has no extension, try adding .jpg
        cand3 = os.path.join(IMAGE_DIR, f"{basename}.jpg") if basename and '.' not in basename else None

        for p in [cand1, cand2, cand3]:
            if p and os.path.exists(p):
                return p
        ## default to sample_id.jpg if none exist; downstream will handle missing paths gracefully
        return cand1 or cand2 or cand3

    image_paths = [resolve_image_path(row) for _, row in df.iterrows()]
    
    ## Find paths that need processing
    paths_to_process = [p for p in image_paths if p not in preds_dict]
    
    if not paths_to_process:
        print("✅ All predictions found in cache")
    else:
        print(f" Processing {len(paths_to_process)} new images...")
        
        ## Process in batches
        for i in tqdm(range(0, len(paths_to_process), BATCH_SIZE), desc=desc):
            batch_paths = paths_to_process[i:i+BATCH_SIZE]
            batch_results = get_price_from_vlm_batch(batch_paths, moondream_model, tokenizer)
            
            ## Update predictions
            preds_dict.update(batch_results)
            
            ## Save progress with cleaned cache
            temp_df = pd.DataFrame([
                {'image_path': path, 'price': price} 
                for path, price in preds_dict.items()
            ])
            temp_df.to_csv(cache_file, index=False)
            
            ## GPU cleanup
            if device == "cuda" and i % (BATCH_SIZE * 4) == 0:
                torch.cuda.empty_cache()
    
    ## Final assembly with comprehensive outlier handling
    raw_predictions = [preds_dict.get(p, None) for p in image_paths]
    successful_preds = [p for p in raw_predictions if p is not None and not pd.isna(p) and p > 0]
    
    if successful_preds:
        ## Final outlier detection and replacement
        successful_array = np.array(successful_preds)
        median_price = np.median(successful_array)
        q1, q3 = np.percentile(successful_array, [25, 75])
        iqr = q3 - q1
        lower_bound = max(0.01, q1 - 1.5 * iqr)
        upper_bound = min(10000, q3 + 1.5 * iqr)
        
        print(f" Final price statistics:")
        print(f"  Successful predictions: {len(successful_preds)}/{len(raw_predictions)} ({len(successful_preds)/len(raw_predictions)*100:.1f}%)")
        print(f"  Median price: ${median_price:.2f}")
        print(f"  Valid range: ${lower_bound:.2f} - ${upper_bound:.2f}")
        
        ## Create final predictions
        final_predictions = []
        outlier_count = 0
        missing_count = 0
        
        for pred in raw_predictions:
            if pred is None or pd.isna(pred) or pred <= 0:
                final_predictions.append(median_price)
                missing_count += 1
            elif pred < lower_bound or pred > upper_bound:
                final_predictions.append(median_price)
                outlier_count += 1
            else:
                final_predictions.append(pred)
        
        print(f" Final cleaning:")
        print(f"  Missing filled: {missing_count}")
        print(f"  Outliers replaced: {outlier_count}")
        
    else:
        print("⚠️ No successful predictions - using training fallback")
        fallback_price = np.median(train_df['price']) if 'price' in train_df.columns else 50.0
        final_predictions = [fallback_price] * len(raw_predictions)
    
    return final_predictions

##  - Main Execution  -
print(" Starting VLM inference pipeline...")

with start_mlflow_run(EXPERIMENT_NAME, RUN_NAME) as run:
    ## Log configuration
    mlflow.log_params({
        "batch_size": BATCH_SIZE,
        "device": device,
        "cuda_version": torch.version.cuda if device == "cuda" else "N/A",
        "model_id": model_id,
        "torch_dtype": str(torch_dtype)
    })

    ## Process datasets
    print(" Processing training data...")
    vlm_oof_preds = process_dataframe(train_df, "VLM Training", "submissions/cache_vlm_oof.csv")
    
    print(" Processing test data...")
    vlm_test_preds = process_dataframe(test_df, "VLM Test", "submissions/cache_vlm_test.csv")

    ## Calculate performance
    vlm_model_smape = smape(train_df['price'], vlm_oof_preds)
    print(f"\n VLM Model OOF SMAPE: {vlm_model_smape:.4f}")
    mlflow.log_metric("vlm_oof_smape", vlm_model_smape)
    
    ## Save final predictions
    print(" Saving predictions...")

    ## Align with ensemble.py expectations
    ## OOF: save log1p of prices as 'vlm_pred' with 'sample_id'
    oof_df = pd.DataFrame({
        'sample_id': train_df['sample_id'],
        'vlm_pred': np.log1p(np.maximum(0.0, vlm_oof_preds))
    })
    oof_df.to_csv('submissions/oof_vlm_preds.csv', index=False)

    ## Test: save original price scale as 'price' with 'sample_id'
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': np.maximum(0.0, vlm_test_preds)
    })
    submission_df.to_csv('submissions/submission_vlm_only.csv', index=False)

    ## Log artifacts
    mlflow.log_artifact('submissions/oof_vlm_preds.csv')
    mlflow.log_artifact('submissions/submission_vlm_only.csv')

print("✅ VLM model completed successfully!")
print(" Output files:")
print("  - submissions/oof_vlm_preds.csv")
print("  - submissions/submission_vlm_only.csv")
print("  - submissions/cache_vlm_*.csv (cleaned and updated)")