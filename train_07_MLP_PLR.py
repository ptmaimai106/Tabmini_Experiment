import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mlp_plr import MLPPLR  # Thư viện MLP-PLR

# Thư mục lưu mô hình
os.makedirs("saved_models/mlp", exist_ok=True)


def train_mlp_plr(X_train, X_test, y_train, y_test, dataset_name):
    print(f"Training MLP-PLR on {dataset_name}...")

    model_filename = f"saved_models/mlp/{dataset_name}_mlpplr.pkl"
    if os.path.exists(model_filename):
        print(f"Loading saved model: {model_filename}")
        mlp_plr_model = joblib.load(model_filename)
    else:
        mlp_plr_model = MLPPLR(input_dim=X_train.shape[1], hidden_layers=[64, 32])
        mlp_plr_model.fit(X_train, y_train, epochs=50, batch_size=32)
        joblib.dump(mlp_plr_model, model_filename)

    y_pred = mlp_plr_model.predict(X_test)
    y_prob = mlp_plr_model.predict_proba(X_test)[:, 1]

    results = {
        "dataset": dataset_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
    }

    return results
