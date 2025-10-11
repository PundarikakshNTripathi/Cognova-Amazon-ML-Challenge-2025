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

This project implements a state-of-the-art multimodal machine learning solution to predict product prices based on textual descriptions and product images. Our approach is a robust, two-pipeline system where predictions from each modality are intelligently combined using an optimized ensemble.

### 🔍 Key Features

1. **Advanced Text Model**: An ensemble of LightGBM and XGBoost models trained on high-quality text features. We moved beyond traditional methods by using Sentence-BERT embeddings to capture the deep semantic meaning of the product catalog_content.

2. **Vision Language Model (VLM)**: A cutting-edge approach using the moondream2 VLM for direct price prediction from product images. This model performs inference via carefully crafted prompts to extract price information visually.

3. **Optimized Ensemble**: The final predictions are a weighted average of the two models. The optimal weights are determined automatically using the Optuna hyperparameter optimization framework to directly minimize the competition's SMAPE metric on our robust local validation set.

## 🚀 Workflow & How to Run

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/PundarikakshNTripathi/Cognova-Amazon-ML-Challenge-2025.git

# Navigate to the project directory
cd Cognova-ML-Challenge-2025

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Text Model (Local Machine)

This script trains the text model ensemble and generates its predictions.

```bash
python src/text_model.py
```

This will create `submission_text_only.csv` and `oof_text_preds.csv` in a new `submissions/` folder.

### 3. Run VLM Model (Colab - GPU Required)

This script uses a VLM for inference and must be run in a GPU environment.

- Upload the `src/image_model.py`, `src/utils.py` scripts and the `data/` folder to a Google Colab instance.
- Ensure the runtime is set to use a GPU (e.g., T4).
- Run the script in a Colab cell: `!python image_model.py`
- After execution, download the generated `submissions/` folder from Colab and place its contents into your local `submissions/` folder.

### 4. Run Ensemble (Local Machine)

This script finds the optimal blend of the text and VLM models and generates the final submission file.

```bash
python src/ensemble.py
```

This will create the final `final_ensemble_submission.csv` in the `submissions/` folder.

### 5. Sanity Check (Local Machine)

Run this script to ensure the final submission file is correctly formatted before uploading.

```bash
python src/sanity.py
```

## 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| [Python](https://www.python.org/) | Core programming language |
| [PyTorch](https://pytorch.org/) | Deep learning framework for VLM |
| [Transformers](https://huggingface.co/docs/transformers/index) | Hugging Face library for VLM models |
| [LightGBM](https://lightgbm.readthedocs.io/en/latest/) | Gradient boosting framework for text model |
| [XGBoost](https://xgboost.ai/) | Gradient boosting framework for text model |
| [Sentence Transformers](https://www.sbert.net/) | State-of-the-art sentence embeddings |
| [Scikit-learn](https://scikit-learn.org/) | Machine learning utilities |
| [Optuna](https://optuna.org/) | Hyperparameter optimization framework |
| [MLflow](https://mlflow.org/) | Experiment tracking |

## 📊 Model Architecture

Our architecture consists of two parallel pipelines whose outputs (out-of-fold predictions on a log scale) are fed into a final weighted-average ensemble model.

```
[Catalog Content] -> [Sentence-BERT Embeddings] -> [LightGBM + XGBoost Ensemble] -> [Text Prediction] ----\
                                                                                                       \
                                                                                                        -> [Optuna-Optimized Ensemble] -> [Final Price]
[Product Image] -> [VLM (moondream2) Inference] -> [Image Prediction] ---------------------------------/
```

## 🏆 Results

| Model | SMAPE Score |
|-------|-------------|
| Text Model (OOF) | TBD |
| VLM Model (OOF) | TBD |
| Final Ensemble (OOF) | TBD |

## 📄 Documentation

See [Documentation_template.md](Documentation_template.md) for the 1-page document for the judges. It's written to be concise, professional, and highlight the innovative aspects of your solution.

## 👥 Team Cognova

- Pundarikaksh Narayan Tripathi
- Ahmad Abdullah
- Yash Raj

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.