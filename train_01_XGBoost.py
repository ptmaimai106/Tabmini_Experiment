import os
import sys
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib


os.makedirs("saved_models/xgboost", exist_ok=True)
tabmini_path = os.path.abspath("./TabMini")
sys.path.append(tabmini_path)
import tabmini

print("Loading TabMini dataset...")
dataset_obj = tabmini.load_dataset(reduced=False)
print("Dataset loaded.")


dataset_names = list(dataset_obj.keys())
results = []

def evaluate_model(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob)
    }

for dataset_name in dataset_names:
    print(f"\n=== Processing dataset: {dataset_name} ===")

    X, y = dataset_obj[dataset_name]
    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values\

    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_filename = f"saved_models/xgboost/{dataset_name}_xgboost.pkl"
    if os.path.exists(model_filename):
        print(f"Loading model from {model_filename}")
        best_model = joblib.load(model_filename)
    else:
        print("No saved model found, training a new one...")

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        joblib.dump(best_model, model_filename)
        print(f"Model saved: {model_filename}")

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics["dataset"] = dataset_name
    results.append(metrics)

results_df = pd.DataFrame(results)
results_df.to_csv("tabmini_xgboost_results.csv", index=False)
print("\nFinished processing all datasets. Results saved to tabmini_xgboost_results.csv")
