import os
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib


def load_tsv_datasets(folder_path):
    """Đọc tất cả các tệp .tsv trong thư mục đã giải nén và trả về datasets"""
    datasets = {}
    for file in os.listdir(folder_path):
        if file.endswith(".tsv"):
            dataset_name = os.path.splitext(file)[0]
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, sep="\t")

            if df.shape[1] < 2:
                print(f"⚠️ Bỏ qua {file}, số cột không đủ!")
                continue

            X = df.iloc[:, :-1]  # Các cột đầu là features
            y = df.iloc[:, -1]  # Cột cuối là label
            datasets[dataset_name] = (X, y)
    return datasets


print("Loading datasets from extracted TSV files...")
dataset_folder = "datasets_decompressed"  # Đặt thư mục chứa file .tsv
os.makedirs(dataset_folder, exist_ok=True)
dataset_obj = load_tsv_datasets(dataset_folder)
dataset_names = list(dataset_obj.keys())
print("Datasets loaded.")


OUTPUT_MODEL_PATH = "saved_models/lightgbm"
RESULT_PATH = "results/lightgbm"


os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)


model_configs = {
    "lightgbm": {
        "model_class": lgb.LGBMClassifier,
        "params": {
            'n_estimators': [50, 100, 200],
            'max_depth':  [-1, 6, 12, 20], # [-1, 6, 9], max_depth=-1 cho phép cây phát triển tự do mà không bị giới hạn
            'learning_rate': [0.01, 0.1, 0.2],
            'min_gain_to_split': [0.0, 0.1, 0.2],
            'min_data_in_leaf': [1, 10, 20]
        },
        "save_path": "saved_models/lightgbm_new/{dataset}_lightgbm.pkl"
    }
}

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    for model_name, config in model_configs.items():
        model_filename = config["save_path"].format(dataset=dataset_name)

        if os.path.exists(model_filename):
            print(f"✅ Loading model {model_name} for {dataset_name} from {model_filename}...")
            model = joblib.load(model_filename)
        else:
            print(f"🚀 Training new {model_name} model for {dataset_name}...")

            # Khởi tạo model và GridSearchCV
            model = config["model_class"]()
            grid_search = GridSearchCV(model, config["params"], cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Lưu model tốt nhất
            model = grid_search.best_estimator_
            joblib.dump(model, model_filename)
            print(f"💾 Model saved: {model_filename}")

        # Đánh giá mô hình
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Lấy xác suất của lớp 1

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

# Lưu kết quả để so sánh
results_df = pd.DataFrame(results)
results_df.to_csv("results/lightgbm/result.csv", index=False)
print("\n✅ Finished evaluation. Results saved to result.csv")