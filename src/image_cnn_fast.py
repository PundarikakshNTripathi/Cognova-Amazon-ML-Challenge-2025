"""Fast image pipeline: ResNet-50 features + LightGBM regressor.

- Extracts and caches FP16 features via memmap (resumable)
- Trains LightGBM with early stopping; attempts GPU and falls back to CPU
- Optional 5-fold OOF generation via --cv for proper ensembling
"""

import os
import re
import math
import gc
import warnings
warnings.filterwarnings("ignore")
import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import KFold

# -----------------------------
# Config
# -----------------------------
IMAGE_DIR = "images"
FEATURES_DTYPE = np.float16
BATCH_SIZE = 96         # adjust if OOM: try 64 or 48
NUM_WORKERS = 4         # increase if disk can keep up
IMG_SIZE = 224
BACKBONE = "resnet50"   # torchvision backbone
SEED = 42

# Default normalization (fallback if weights.transforms() not available)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

os.makedirs("submissions", exist_ok=True)
os.makedirs("data", exist_ok=True)

torch.backends.cudnn.benchmark = True

# -----------------------------
# Helpers
# -----------------------------
def get_id_col(df: pd.DataFrame) -> str:
    for c in ["sample_id", "id", "Id", "sampleId"]:
        if c in df.columns:
            return c
    raise KeyError("No ID column found. Expected one of: sample_id, id, Id, sampleId")

def resolve_image_path(row, id_col: str) -> str:
    sid = str(row[id_col])
    # Try images/{sample_id}.jpg first (what most downloaders use)
    cand1 = os.path.join(IMAGE_DIR, f"{sid}.jpg")
    if os.path.exists(cand1):
        return cand1
    # Fallback: derive basename from image_link if present
    link = str(row.get("image_link", "")) or str(row.get("image_url", ""))
    base = os.path.basename(link)
    if base:
        cand2 = os.path.join(IMAGE_DIR, base)
        if os.path.exists(cand2):
            return cand2
        # if base has no extension, try adding .jpg
        if "." not in base:
            cand3 = os.path.join(IMAGE_DIR, base + ".jpg")
            if os.path.exists(cand3):
                return cand3
    # Last resort: non-existent path; loader will handle gracefully
    return cand1

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100.0 * np.mean(np.where(denom == 0, 0, np.abs(y_pred - y_true) / (denom + 1e-12)))

# -----------------------------
# Dataset
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id_col: str, transform):
        self.df = df
        self.id_col = id_col
        self.transform = transform
        self.paths = [resolve_image_path(r, id_col) for _, r in df.iterrows()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
        except Exception:
            # Missing/corrupt -> return plain gray image
            im = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
        if self.transform:
            im = self.transform(im)
        return im

# -----------------------------
# Feature extractor
# -----------------------------
class ResNet50Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Support both new (weights=...) and older (pretrained=True) APIs
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            m = models.resnet50(weights=weights)
        except Exception:
            m = models.resnet50(pretrained=True)
        # keep everything except final fc
        self.feature_extractor = nn.Sequential(*list(m.children())[:-1])  # -> (N, 2048, 1, 1)
        self.out_dim = 2048

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # (N, 2048)
        return x

def build_transforms():
    # Prefer official weight transforms for correct preprocessing
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        return weights.transforms()
    except Exception:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

@torch.no_grad()
def extract_features(df: pd.DataFrame, id_col: str, out_path: str, device: str) -> str:
    # Robust cache check based on expected file size
    n = len(df)
    d = 2048  # resnet50 penultimate feature dim
    expected_bytes = n * d * np.dtype(FEATURES_DTYPE).itemsize
    if os.path.exists(out_path):
        actual = os.path.getsize(out_path)
        if actual == expected_bytes:
            print(f"✅ Using cached features: {out_path}")
            return out_path
        else:
            print(f"⚠️ Incomplete/old cache detected ({actual} bytes, expected {expected_bytes}). Rebuilding: {out_path}")
            try:
                os.remove(out_path)
            except Exception:
                pass

    transform = build_transforms()
    ds = ImageDataset(df, id_col, transform)
    extra_loader_kwargs = {"persistent_workers": True, "prefetch_factor": 2} if NUM_WORKERS > 0 else {}
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        **extra_loader_kwargs,
    )

    model = ResNet50Extractor().to(device).eval()
    # memory-map to avoid huge RAM usage
    mmap = np.memmap(out_path, mode="w+", dtype=FEATURES_DTYPE, shape=(n, d))

    idx = 0
    pbar = tqdm(dl, total=len(dl), desc=f"Extracting {os.path.basename(out_path)}")
    for batch in pbar:
        batch = batch.to(device, non_blocking=True)
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats = model(batch)  # (bs, d)
        else:
            feats = model(batch)
        feats = feats.detach().to("cpu").numpy().astype(FEATURES_DTYPE)
        bs = feats.shape[0]
        mmap[idx:idx+bs] = feats
        idx += bs

    # flush to disk
    del mmap
    gc.collect()
    return out_path

# -----------------------------
# Train LightGBM and predict
# -----------------------------
def _make_lgbm_gpu_regressor():
    base = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        n_estimators=5000,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=40,
        reg_lambda=1.0,
        n_jobs=-1,
        verbose=-1,
    )
    # Try modern and legacy GPU configs, else CPU
    try:
        return lgb.LGBMRegressor(**{**base, "device_type": "gpu"})
    except TypeError:
        pass
    try:
        return lgb.LGBMRegressor(**{**base, "device": "gpu"})
    except TypeError:
        pass
    return lgb.LGBMRegressor(**base)

def _smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100.0 * np.mean(np.where(denom == 0, 0, np.abs(y_pred - y_true) / (denom + 1e-8)))

def run_kfold_oof(feat_train_path, feat_test_path, train_df, test_df, id_col, n_splits=5):
    """Generate proper OOF via KFold using cached features and overwrite CSVs."""
    print(f"[Stage] OOF CV with KFold={n_splits} using cached features")
    if not (os.path.exists(feat_train_path) and os.path.exists(feat_test_path)):
        raise FileNotFoundError("Cached features not found. Run the script without --cv first to extract.")

    n_train = len(train_df)
    n_test = len(test_df)
    d = 2048  # ResNet-50 pooled feature size

    X = np.memmap(feat_train_path, mode="r", dtype=FEATURES_DTYPE, shape=(n_train, d))
    T = np.memmap(feat_test_path,  mode="r", dtype=FEATURES_DTYPE, shape=(n_test,  d))
    y = train_df["price"].values.astype(np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        y_tr_log = np.log1p(y_tr)
        y_va_log = np.log1p(y_va)

        reg = _make_lgbm_gpu_regressor()
        print(f"[Fold {fold}] Training LightGBM ({reg.__class__.__name__}) ...")
        reg.fit(
            X_tr, y_tr_log,
            eval_set=[(X_va, y_va_log)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
        )

        va_pred = np.expm1(reg.predict(X_va, num_iteration=getattr(reg, "best_iteration_", None)))
        oof[va_idx] = va_pred.astype(np.float32)

        t_pred = np.expm1(reg.predict(T, num_iteration=getattr(reg, "best_iteration_", None)))
        test_preds += t_pred / n_splits

        print(f"[Fold {fold}] SMAPE: {_smape(y_va, va_pred):.4f}% | best_iter={getattr(reg, 'best_iteration_', None)}")

    oof_smape = _smape(y, oof)
    print(f"[OOF] SMAPE: {oof_smape:.4f}% on {n_train} rows")

    # Clip and save
    oof = np.clip(oof, 0.01, 10000.0).astype(np.float32)
    test_preds = np.clip(test_preds, 0.01, 10000.0).astype(np.float32)

    os.makedirs("submissions", exist_ok=True)
    oof_path = os.path.join("submissions", "oof_image_cnn.csv")
    sub_path = os.path.join("submissions", "submission_image_cnn.csv")
    pd.DataFrame({id_col: train_df[id_col], "predicted_price": oof}).to_csv(oof_path, index=False)
    pd.DataFrame({id_col: test_df[id_col],  "price": test_preds}).to_csv(sub_path, index=False)
    print("Saved (overwritten if existed):")
    print(f" - {oof_path}")
    print(f" - {sub_path}")
def train_lgbm_and_predict(X_train: np.ndarray, y: np.ndarray, X_test: np.ndarray):
    print("[Stage] Starting LightGBM training (with early stopping)...")
    # log1p target
    y_log = np.log1p(y)

    # simple split for speed
    n = X_train.shape[0]
    val_size = max(10000, int(0.1 * n))
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_log[tr_idx], y_log[val_idx]

    base_params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        n_estimators=3000,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=40,
        reg_lambda=1.0,
        n_jobs=-1,
        verbose=-1,
    )
    model = None
    trainer = ""
    # 1) Try LightGBM GPU (newer API)
    try:
        print("[Stage] Attempting LightGBM GPU (device_type='gpu')...")
        params = {**base_params, "device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=150, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        trainer = "lgbm_gpu_device_type"
    except Exception as e_gpu1:
        # 2) Try older LightGBM GPU API
        try:
            print("[Stage] Falling back: LightGBM GPU (device='gpu')...")
            params = {**base_params, "device": "gpu"}
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=150, verbose=False),
                    lgb.log_evaluation(period=100)
                ]
            )
            trainer = "lgbm_gpu_device"
        except Exception as e_gpu2:
            # 3) CPU LightGBM as default
            try:
                print("[Stage] LightGBM GPU unavailable; using LightGBM CPU.")
                params = {**base_params}
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=150, verbose=False),
                        lgb.log_evaluation(period=100)
                    ]
                )
                trainer = "lgbm_cpu"
            except Exception as e_cpu:
                # 4) Optional XGBoost GPU fallback
                try:
                    print("[Stage] Falling back to XGBoost (GPU)...")
                    import xgboost as xgb
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=3000,
                        learning_rate=0.05,
                        max_depth=8,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        tree_method="gpu_hist",
                        predictor="gpu_predictor",
                        n_jobs=-1,
                        random_state=SEED,
                    )
                    xgb_model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                        callbacks=[xgb.callback.EarlyStopping(rounds=150, save_best=True)]
                    )
                    class _XGBWrap:
                        def __init__(self, mdl):
                            self.mdl = mdl
                            self.best_iteration_ = getattr(mdl, "best_iteration", None)
                        def predict(self, X, num_iteration=None):
                            return self.mdl.predict(X)
                    model = _XGBWrap(xgb_model)
                    trainer = "xgb_gpu"
                except Exception as e_xgb:
                    raise RuntimeError(f"All trainers failed (LGBM GPU/CPU, XGB GPU). Last error: {e_xgb}")

    # OOF
    if trainer.startswith("lgbm"):
        y_val_pred = model.predict(X_val, num_iteration=getattr(model, "best_iteration_", None))
    else:
        y_val_pred = model.predict(X_val)
    y_val_pred = np.expm1(y_val_pred)
    y_val_true = np.expm1(y_val)
    print(f"Validation SMAPE: {smape(y_val_true, y_val_pred):.4f}% (on {len(y_val_true)} samples)")

    # Fit on full data
    if trainer.startswith("lgbm"):
        final_n_estimators = getattr(model, "best_iteration_", None)
        if final_n_estimators is None:
            final_n_estimators = base_params["n_estimators"]
        final_params = {**base_params, "n_estimators": int(final_n_estimators)}
        if trainer == "lgbm_gpu_device_type":
            final_params.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
        elif trainer == "lgbm_gpu_device":
            final_params.update({"device": "gpu"})
        print(f"[Stage] Training final LightGBM model on full data ({trainer})...")
        model_full = lgb.LGBMRegressor(**final_params)
        model_full.fit(X_train, y_log)
        model = model_full
    else:
        import xgboost as xgb
        print("[Stage] Training final XGBoost model on full data (gpu_hist)...")
        best_iter = getattr(model, "best_iteration_", None)
        best_iter = int(best_iter) if best_iter is not None else 3000
        model_full = xgb.XGBRegressor(
            n_estimators=best_iter,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            n_jobs=-1,
            random_state=SEED,
        )
        model_full.fit(X_train, y_log)
        class _XGBWrap2:
            def __init__(self, mdl):
                self.mdl = mdl
            def predict(self, X):
                return self.mdl.predict(X)
        model = _XGBWrap2(model_full)

    # Predict
    print("[Stage] Predicting on test features...")
    y_pred_test = model.predict(X_test)
    y_pred_test = np.expm1(y_pred_test)
    y_pred_test = np.clip(y_pred_test, 0.01, 10000.0)
    return model, y_pred_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, default=0, help="Run K-fold OOF with this number of folds (>=2). 0 disables.")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_path = os.path.join("data", "train.csv")
    test_path  = os.path.join("data", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    id_col = get_id_col(train_df)
    assert id_col in test_df.columns, f"ID column {id_col} missing in test.csv"

    # Feature cache paths
    feat_train_path = os.path.join("data", "features_train_resnet50_fp16.npy")
    feat_test_path  = os.path.join("data", "features_test_resnet50_fp16.npy")

    # Extract (or reuse cached) features
    print(f"[Stage] Preparing TRAIN features -> {feat_train_path}")
    feat_train_path = extract_features(train_df, id_col, feat_train_path, device)
    print("[Stage] TRAIN features ready.")
    print(f"[Stage] Preparing TEST features  -> {feat_test_path}")
    feat_test_path  = extract_features(test_df, id_col, feat_test_path, device)
    print("[Stage] TEST features ready.")

    # If CV requested, generate proper OOF + submission and exit
    if args.cv and args.cv >= 2:
        run_kfold_oof(feat_train_path, feat_test_path, train_df, test_df, id_col, n_splits=args.cv)
        return

    # Load features
    print("[Stage] Loading feature memmaps into X_train/X_test...")
    X_train = np.memmap(feat_train_path, mode="r", dtype=FEATURES_DTYPE, shape=(len(train_df), 2048))
    X_test  = np.memmap(feat_test_path,  mode="r", dtype=FEATURES_DTYPE, shape=(len(test_df),  2048))

    y = train_df["price"].values.astype(np.float32)

    # Train regressor and predict
    print("[Stage] Launching training + prediction...")
    _, test_preds = train_lgbm_and_predict(X_train, y, X_test)

    # Save outputs
    print("[Stage] Writing output CSVs...")
    oof_stub = np.zeros_like(y)  # keeping simple; full OOF would need CV
    pd.DataFrame({id_col: train_df[id_col], "predicted_price": oof_stub}).to_csv(
        "submissions/oof_image_cnn.csv", index=False
    )
    pd.DataFrame({id_col: test_df[id_col], "price": test_preds}).to_csv(
        "submissions/submission_image_cnn.csv", index=False
    )

    print("Saved:")
    print(" - submissions/oof_image_cnn.csv")
    print(" - submissions/submission_image_cnn.csv")

if __name__ == "__main__":
    main()