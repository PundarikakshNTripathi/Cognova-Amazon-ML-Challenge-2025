# Cognova - Amazon ML Challenge 2025

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn" />
<img src="https://img.shields.io/badge/LightGBM-FF6D00?style=for-the-badge&logo=lightgbm&logoColor=white" alt="LightGBM" />
<img src="https://img.shields.io/badge/XGBoost-007ACC?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
<img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
</p>

<p align="center">
<strong>Team Cognova's official submission for the Amazon ML Challenge 2025: Smart Product Pricing</strong>
</p>

## 📋 Project Overview

This project implements a state-of-the-art multimodal machine learning solution to predict product prices based on textual descriptions and product images. Our approach is a robust, two-pipeline system where predictions from each modality are intelligently combined using an optimized ensemble. This architecture is designed for high performance, rapid iteration, and reproducibility.

## 🔍 Key Features

1. **Advanced Text Pipeline**: An ensemble of LightGBM and XGBoost models trained on high-quality text features. We utilize Sentence-BERT embeddings (all-MiniLM-L6-v2) to capture the deep semantic meaning of product descriptions, a significant upgrade over traditional methods. The pipeline includes separate, optimized scripts for both CPU and GPU execution, allowing for flexibility and speed.

2. **Fast CNN Image Pipeline**: A production-ready approach using ResNet-50 as a fixed feature extractor, feeding 2048-dimensional pooled features into LightGBM for regression. Features are cached as FP16 memmaps for fast re-runs, and the pipeline supports 5-fold out-of-fold generation for proper ensembling. This path balances speed, reliability, and performance under time constraints. *(Originally, we explored a VLM-based approach using moondream2 for direct price prediction from images, but prioritized the CNN path for the final run due to runtime and resource considerations. The VLM pipeline remains available for future integration.)*

3. **Optimized Ensemble Strategy**: The final predictions are a weighted blend of text and image CNN models. The script intelligently combines predictions from both CPU and GPU text models when available, and automatically optimizes non-negative weights using the Optuna hyperparameter optimization framework to directly minimize the competition's SMAPE metric on our local out-of-fold validation set. It also auto-detects and corrects log-scale mismatches in OOF predictions.

4. **MLOps and Reproducibility**: The entire workflow is instrumented with MLflow for comprehensive experiment tracking of parameters, metrics, and artifacts. Caching mechanisms for embeddings, CNN features, and intermediate predictions are implemented to drastically reduce runtimes after the initial execution. Progress logs and resume-friendly behavior ensure long-running processes can be interrupted and restarted cleanly.

## � End-to-end process overview

Here’s how the full pipeline fits together:

1. Inputs
   - `data/train.csv`: training rows with `sample_id`, `catalog_content`, `image_link`, `price`.
   - `data/test.csv`: test rows with `sample_id`, `catalog_content`, `image_link` (no `price`).
2. Text features
   - Scripts: `src/text_model_cpu.py`, `src/text_model_gpu.py`.
   - Steps: clean/catalog parsing, SBERT embeddings (cached to `data/text_embeddings.npy`), LightGBM + XGBoost training with stratified CV on log(price), OOF/test predictions saved to `submissions/`.
3. Image features (fast CNN)
   - Script: `src/image_cnn_fast.py`.
   - Steps: resolve image paths, extract ResNet-50 pooled features (cached, FP16 memmap), train LightGBM, produce OOF/test predictions. Use `--cv` for 5-fold OOF.
4. Advanced Ensemble
   - Script: `src/ensemble_advanced.py`.
   - Steps: load available OOFs, auto-fix log-scale mismatches (OOF), optimize non-negative weights with Optuna (minimize SMAPE), log to MLflow, blend test predictions, write `submissions/submission_ensemble_advanced.csv`.
5. Submission copy
   - Copy final blend to repository root as `test_out.csv` for portal upload.

## 📁 Directory and file guide

- `data/`
  - `train.csv`, `test.csv`: official inputs.
  - `sample_test.csv`, `sample_test_out.csv`: format examples from the challenge.
  - `features_*_resnet50_fp16.npy`: cached CNN features (if present).
  - `text_embeddings.npy`: cached SBERT embeddings.
- `images/`
  - Downloaded images (if using image pipelines); resolved by `src/image_cnn_fast.py` and `src/image_model.py`.
- `src/`
  - `utils.py`: utility functions (SMAPE, MLflow helpers, image download).
  - `text_model_cpu.py`: text-only pipeline on CPU; caches embeddings; writes OOF/test CSVs.
  - `text_model_gpu.py`: text-only pipeline using GPU for embedding generation/XGBoost; writes OOF/test CSVs.
  - `image_cnn_fast.py`: fast image pipeline (ResNet50 features + LightGBM), feature caching, optional `--cv` for OOF.
  - `image_model.py`: optional VLM image pipeline (moondream2) with batching/caching; not used in final run.
  - `ensemble_advanced.py`: Optuna-weighted ensemble with MLflow logging; outputs final blended CSV.
- `submissions/`
  - OOF files: `oof_text_preds_cpu.csv`, `oof_text_preds_gpu.csv`, `oof_image_cnn.csv`.
  - Test predictions: `submission_text_cpu.csv`, `submission_text_gpu.csv`, `submission_image_cnn.csv`.
  - Final blend: `submission_ensemble_advanced.csv` (source for `test_out.csv`).
- `experiments/mlruns/`
  - MLflow runs, params, metrics, and artifacts for traceability.
- `archive/`
  - Non-essential or auxiliary assets (e.g., colab_upload) kept here to reduce clutter.
- Top-level files
  - `README.md`: quickstart, approach, results, troubleshooting.
  - `Documentation.md`: 1-page formal report aligned with the challenge template.
  - `requirements.txt`: dependencies; adjust torch/torchvision wheels for your platform/CUDA if needed.
  - `LICENSE`: licensing information (Apache-2.0).
  - `test_out.csv`: final submission file for the portal.

## �🚀 Workflow & How to Run

### 1. Setup

#### Environment Setup
It is highly recommended to use a virtual environment. We recommend Python 3.11 for stability and compatibility with all libraries.

**Using Conda:**
```bash
# Create a new conda environment
conda create -n amc2025 python=3.11

# Activate the environment
conda activate amc2025
```

**Using venv (alternative):**
```bash
# Create a new virtual environment
python -m venv amc2025

# Activate the environment (Windows)
.\amc2025\Scripts\activate

# Activate the environment (macOS/Linux)
source amc2025/bin/activate
```

#### Project Setup
```bash
# Clone the repository
git clone https://github.com/PundarikakshNTripathi/Cognova-Amazon-ML-Challenge-2025.git

# Navigate to the project directory
cd Cognova-Amazon-ML-Challenge-2025

# Install dependencies
pip install -r requirements.txt
```

#### GPU Configuration (Highly Recommended)
For local GPU execution, ensure your environment is correctly configured:
1. **NVIDIA Driver**: Install the NVIDIA Studio Driver for optimal performance in computational tasks.
2. **GPU Mode**: If using a gaming laptop with a MUX switch (like NVIDIA Advanced Optimus), set the mode to "GPU Only" or "dGPU Mode" in your system's control software (e.g., OMEN Gaming Hub) and reboot. This guarantees that all scripts have access to the dedicated GPU.

### 2. Run Text Models (Local Machine)

These scripts generate text-based predictions. The first run will be long as it generates and caches embeddings. Subsequent runs will be much faster. You can run either or both.

To run the CPU version:
```bash
python src/text_model_cpu.py
```

To run the GPU version (recommended for speed):
```bash
python src/text_model_gpu.py
```

These scripts will create prediction files (e.g., `submission_text_cpu.csv`, `oof_text_preds_gpu.csv`) in the `submissions/` folder.

### 3. Run Fast Image CNN (GPU recommended)

This path is fast and reliable: it extracts CNN features once and caches them, then trains LightGBM.

```
```bash
python src/image_cnn_fast.py
```

- Generate proper 5-fold OOF and overwrite image CSVs (after features exist):
```bash
python src/image_cnn_fast.py --cv
```

Outputs in `submissions/`:
- `oof_image_cnn.csv` (predicted_price)
- `submission_image_cnn.csv` (price)

### 3. Run VLM Model (Local GPU or Colab)

This script uses the VLM for image-based inference. It is heavily optimized for a GPU.

**Local Execution (Recommended):**
Ensure your GPU is configured as described in the setup.
```bash
python src/image_model.py
```

### 4b. Run Advanced Ensemble (recommended)

This optimized ensemble blends text (CPU/GPU) and image CNN predictions with Optuna-tuned weights and MLflow logging.

```bash
python src/ensemble_advanced.py
```

Artifacts:
- OOF blend score (SMAPE) logged to MLflow
- Final blended submission: `submissions/submission_ensemble_advanced.csv`

Tip: The script auto-detects and fixes log-scale mismatches in OOF files (applies expm1 when needed). Test files are assumed to be on original price scale.



## 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| [Python](https://www.python.org/downloads/) | Core programming language |
| [PyTorch](https://pytorch.org/) | Deep learning framework for VLM & Embeddings |
| [Transformers](https://huggingface.co/docs/transformers/index) | Hugging Face library for VLM models |
| [Torchvision](https://pytorch.org/vision/stable/index.html) | ResNet-50 feature extractor for fast CNN pipeline |
| [LightGBM](https://lightgbm.readthedocs.io/en/latest/) | Gradient boosting framework for text model |
| [XGBoost](https://xgboost.ai/) | Gradient boosting framework for text model |
| [Sentence-BERT](https://www.sbert.net/) | State-of-the-art sentence embeddings |
| [Scikit-learn](https://scikit-learn.org/) | Machine learning utilities & validation |
| [Optuna](https://optuna.org/) | Hyperparameter optimization for ensembling |
| [MLflow](https://mlflow.org/) | Experiment tracking and MLOps |

## 📊 Model Architecture

Our architecture consists of two parallel pipelines whose outputs are fed into a final, intelligently weighted ensemble model.

```mermaid
flowchart TD
  subgraph text[Text Pipeline]
    A[Catalog Content] --> B[Sentence-BERT Embeddings]
    B --> C[LightGBM + XGBoost]
    C --> D[Text Prediction]
  end

  subgraph vision[Vision Pipeline - CNN]
    E[Product Image] --> F[ResNet50 Features + LightGBM]
    F --> G[Image Prediction - CNN]
  end

  subgraph ensemble[Final Ensemble]
    D --> H[Optuna-Optimized Ensemble]
    G --> H
    H --> I[Final Price Prediction]
  end
```
## Performance Metrics (Out-of-Fold)

- **CPU Text Ensemble SMAPE:** 60.0676
  - LightGBM CPU: 60.8804
  - XGBoost CPU: 59.4842

- **GPU Text Ensemble SMAPE:** ~60.0 (LightGBM CPU + XGBoost GPU)
  - LightGBM CPU: 60.8804 (CPU fallback for stability)
  - XGBoost GPU: 59.2832

**Technical Notes:**
- GPU script uses CPU for LightGBM due to numerical precision issues on GPU
- XGBoost GPU provides reliable acceleration without stability concerns
- This hybrid approach maintains ensemble diversity while ensuring model quality

Additional highlights:
- Fast CNN (ResNet50+LGBM): strong image-only baseline with proper 5-fold OOF
- Advanced Ensemble (text_cpu + text_gpu + image_cnn): OOF SMAPE improved to ~58.02 after aligning OOF scales

## 🏆 Results

The performance of each model component is tracked via a robust 5-fold stratified cross-validation strategy. The final scores are based on the out-of-fold (OOF) predictions.

| Model | SMAPE Score (OOF) |
|-------|-------------------|
| Text Model (CPU) | 60.0676 |
| Text Model (GPU) | 59.9404 |
| Image CNN (ResNet50 + LGBM) | 59.3695 |
| VLM Model | N/A (not used in final) |
| Final Ensemble (Advanced) | ~58.02 |

Note: Scores are from current OOF files and may vary slightly with different seeds or environments.

## What changed vs. the original plan

- Image model: We prioritized a fast CNN path (ResNet50 feature extractor + LightGBM) over the VLM for the final run due to runtime/resource constraints, while keeping the VLM pipeline available for future integration.
- Ensembling: We moved to an Optuna-tuned ensemble with MLflow logging and automatic OOF log-scale alignment (expm1 when detected) to avoid scale mismatches.
- Caching and resilience: Added aggressive caching (embeddings, image features, intermediate predictions), clearer stage logs, and resume-friendly behavior.

### Future work (not in the final run)
- Integrate VLM OOF/test predictions (e.g., cached VLM OOF when available) into the advanced ensemble.
- Add lightweight structured features (e.g., explicit IPQ parsing, brand) to augment text/image pipelines.
- Consider calibration (e.g., isotonic regression) if leaderboard feedback shows bias.

## 🏁 Final Output to Submit

- Required file name: `test_out.csv`
- Required format: header `sample_id,price`, positive floats, and predictions for all 75k test IDs.
- We keep the original blended file at `submissions/submission_ensemble_advanced.csv` and provide a copy at the repo root as `test_out.csv` for portal upload.

## ⚙️ Troubleshooting and Notes

- If image extraction appears stuck, look for "[Stage]" logs and tqdm progress bars; the script uses memmap caching and resumes cleanly.
- LightGBM GPU can be inconsistent across versions; our code attempts GPU and falls back to CPU automatically.
- OOF vs test scale: we auto-detect log1p OOFs and convert with expm1; test predictions are on original scale and should not be transformed.
- Image download hiccups are retried; missing/corrupt images are handled with a neutral placeholder image.
- Large CSVs: read selective columns/dtypes when possible to control memory.

Platform notes:
- PyTorch/Torchvision wheels in requirements use CUDA 12.9 index URLs. If you're on CPU-only or a different CUDA version, install matching wheels from the PyTorch site, or remove the index-url suffixes for CPU.
- All scripts auto-detect CUDA and fall back to CPU when not available.

Requirements: No changes are needed for this repo’s environment. Only adjust torch/torchvision lines to match your platform if necessary.

## 📄 Documentation

The final 1-page report for the judges can be found in `Documentation.md`.

## 👥 Team Cognova

- Pundarikaksh Narayan Tripathi
- Ahmad Abdullah
- Yash Raj

## 📝 License

This project is licensed under the Apache-2.0 License.

