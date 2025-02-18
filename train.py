import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from train_07_MLP_PLR import train_mlp_plr
from train_09_TabTransformer import train_tab_transformer
from train_11_saint import train_saint

tabmini_path = os.path.abspath("./TabMini")
sys.path.append(tabmini_path)
import tabmini

print("Loading TabMini dataset...")
dataset_obj = tabmini.load_dataset(reduced=False)
print("Dataset loaded.")

dataset_names = list(dataset_obj.keys())

# Tạo thư mục lưu kết quả
os.makedirs("results", exist_ok=True)

# Danh sách kết quả cho từng model
mlpplr_results = []
tabtransformer_results = []
saint_results = []

# Duyệt qua từng dataset
for dataset_name in dataset_names:
    print(f"\n=== Processing dataset: {dataset_name} ===")
    X, y = dataset_obj[dataset_name]

    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train từng model
    # mlpplr_results.append(train_mlp_plr(X_train, X_test, y_train, y_test, dataset_name))
    tabtransformer_results.append(train_tab_transformer(X_train, X_test, y_train, y_test, dataset_name))
    saint_results.append(train_saint(X_train, X_test, y_train, y_test, dataset_name))

# Lưu kết quả
# pd.DataFrame(mlpplr_results).to_csv("results/tabmini_mlpplr_results.csv", index=False)
pd.DataFrame(tabtransformer_results).to_csv("results/tabmini_tabtransformer_results.csv", index=False)
pd.DataFrame(saint_results).to_csv("results/tabmini_saint_results.csv", index=False)

print("\nFinished training. Results saved in 'results/' folder.")
