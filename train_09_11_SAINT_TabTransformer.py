import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load TabMini dataset
tabmini_path = os.path.abspath("./TabMini")
sys.path.append(tabmini_path)
import tabmini

from pytorch_tabnet.tab_model import TabNetClassifier  # TabTransformer
from pytorch_tabnet.pretraining import TabNetPretrainer  # SAINT

# Thư mục lưu model
MODEL_SAVE_PATH = "saved_models/saint_tabtransformer"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs("results/saint_tabtransformer", exist_ok=True)

# Hàm train và lưu best model
def train_tab_model(model, X_train, y_train, X_valid, y_valid, model_name, epochs=50):
    model_path = os.path.join(MODEL_SAVE_PATH, model_name)

    if os.path.exists(model_path):  # Load model nếu đã được train trước đó
        model.load_model(model_path)
        print(f"✅ Loaded existing model: {model_path}")

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=["valid"],
        eval_metric=["auc"],
        max_epochs=epochs,
        patience=10,  # Early stopping nếu không cải thiện trong 10 epoch
        batch_size=256,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False
    )

    # Lưu best model
    model.save_model(model_path)
    print(f"✅ Saved best model: {model_path}")

    return model

# Load dataset
print("Loading TabMini dataset...")
dataset_obj = tabmini.load_dataset(reduced=False)
dataset_names = list(dataset_obj.keys())
print("Dataset loaded.")

results = []
for dataset_name in dataset_names:
    print(f"\n=== Processing dataset: {dataset_name} ===")
    X, y = dataset_obj[dataset_name]

    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    # Encode label nếu cần
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train TabTransformer
    model_tab = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params={"lr": 2e-2}, verbose=0)
    model_tab = train_tab_model(model_tab, X_train, y_train, X_test, y_test, f"TabTransformer_{dataset_name}.zip")

    # Train SAINT (TabNet pretraining)
    model_saint = TabNetPretrainer(verbose=0)
    model_saint = train_tab_model(model_saint, X_train, y_train, X_test, y_test, f"SAINT_{dataset_name}.zip")

    # Đánh giá mô hình
    for model, model_name in [(model_tab, "TabTransformer"), (model_saint, "SAINT")]:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        metrics = {
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob)
        }
        results.append(metrics)

# Lưu kết quả
results_df = pd.DataFrame(results)
results_df.to_csv("results/saint_tabtransformer/tabmini_model_comparison.csv", index=False)
print("\n✅ Finished evaluation. Results saved to tabmini_model_comparison.csv")
