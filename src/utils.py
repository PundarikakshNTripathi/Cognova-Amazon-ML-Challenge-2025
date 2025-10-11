import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os
import mlflow

def smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).
    This is the official evaluation metric. Lower is better.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    epsilon = 1e-8
    ratio = numerator / (denominator + epsilon)
    return np.mean(ratio) * 100

def download_images(df, image_dir='images/'):
    """
    Downloads images from a dataframe's 'image_link' column.
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    for _, row in tqdm(df.iterrows(), total=df.shape, desc="Downloading Images"):
        link = row['image_link']
        image_name = link.split('/')[-1]
        save_path = os.path.join(image_dir, image_name)
        
        if os.path.exists(save_path):
            continue
        try:
            response = requests.get(link, timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
        except Exception as e:
            print(f"Could not download {link}: {e}")

def start_mlflow_run(experiment_name, run_name):
    """
    Starts an MLflow run and sets tags for easy identification.
    """
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    print(f"MLflow Run Started: {run.info.run_name} ({run.info.run_id})")
    return run