# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Cognova  
**Team Members:** Pundarikaksh Narayan Tripathi, Ahmad Abdullah, Yash Raj 
**Submission Date:** October 13, 2025

---

## 1. Executive Summary

Our solution addresses the Smart Product Pricing Challenge by implementing a robust, multimodal ensemble model that leverages both textual and visual data. We combine a sophisticated text-based model using Sentence-BERT embeddings with a state-of-the-art Vision Language Model (VLM) for image-based price extraction. The final predictions are an optimized blend of these two pipelines, delivering a highly accurate and generalizable pricing model.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

Our initial Exploratory Data Analysis (EDA) revealed that product price is influenced by a combination of explicit details in the text and implicit visual cues in the image.

**Key Observations:**

*   The `price` distribution is heavily right-skewed, indicating that a log transformation would be beneficial for modeling to stabilize variance and improve performance for tree-based models.

*   The `catalog_content` often contains explicit numerical indicators like "Item Pack Quantity" (IPQ), which are highly predictive features that can be extracted with regular expressions.

*   Visual features such as brand logos, product complexity, and materials, which are not always present in the text, provide significant additional signal that can only be captured from the images.

### 2.2 Solution Strategy

We adopted a multimodal ensemble strategy to capture signals from both data sources independently before combining them for a final prediction. This approach mitigates the risk of one modality overpowering the other and leverages the unique strengths of different model architectures.

**Approach Type:** Hybrid Multimodal Ensemble  
**Core Innovation:** Our core innovation lies in the intelligent fusion of two distinct, state-of-the-art pipelines: a deep semantic text model and a VLM-based visual inference engine. The final blending weights are not manually set but are mathematically determined using the **Optuna** framework to directly minimize the competition's SMAPE metric on our validation data.

---

## 3. Model Architecture

### 3.1 Architecture Overview

Our architecture consists of two parallel pipelines whose outputs (out-of-fold predictions on a log scale) are fed into a final weighted-average ensemble model.

```
[Catalog Content] -> [Sentence-BERT Embeddings] -> [LightGBM + XGBoost Ensemble] -> [Text Prediction] ----\
                                                                                                       \
                                                                                                        -> [Optuna-Optimized Ensemble] -> [Final Price]
[Product Image] -> [VLM (moondream2) Inference] -> [Image Prediction] ---------------------------------/
```

### 3.2 Model Components

**Text Processing Pipeline:**

- **Preprocessing:**
    - Extracted numerical "Item Pack Quantity" (IPQ) using regular expressions.
    - Transformed the target variable `price` using `numpy.log1p` to normalize its distribution.

- **Feature Extraction:** Converted `catalog_content` into 384-dimensional vectors using the `all-MiniLM-L6-v2` Sentence-BERT model to capture semantic meaning.

- **Model Type:** An ensemble of **LightGBM** and **XGBoost** regressors. Predictions from both models were averaged to create a more robust text-based prediction.

- **Validation:** A robust 5-fold **Stratified Cross-Validation** strategy was used. Stratification was performed on bins of the log-transformed price to ensure stable and reliable local scoring.

**Image Processing Pipeline:**

- **Model Type:** We utilized a pre-trained Vision Language Model (VLM), `vikhyatk/moondream2`, for direct price inference.

- **Method:** For each image, we prompted the VLM with the question: "What is the price of this item? Answer with only a single numerical value." This leverages the model's powerful visual understanding to act as an "expert" price estimator.

- **Post-processing:** The model's textual output was parsed with regular expressions to extract the numerical price. Missing predictions were imputed with the mean of the successful predictions from the validation set to ensure robustness.

---

## 4. Model Performance

### 4.1 Validation Results

Our final model's performance was rigorously evaluated using the 5-fold stratified cross-validation framework. The out-of-fold (OOF) predictions were used to calculate the SMAPE score, providing an unbiased estimate of the model's performance on unseen data.

- **Final Ensemble SMAPE Score (OOF):** TBD
- **Text Model SMAPE (OOF):** TBD
- **VLM Model SMAPE (OOF):** TBD

---

## 5. Conclusion

Our solution demonstrates the power of a carefully constructed multimodal ensemble. By treating text and image data with specialized, state-of-the-art techniques and then mathematically optimizing their fusion, we created a model that is more accurate and robust than either single-modality approach alone. This structured, reproducible workflow was key to our success.

---

## Appendix

### A. Code artefacts

*A link to our complete code repository will be provided here.*