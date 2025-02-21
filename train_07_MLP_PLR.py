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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve

# Load TabMini dataset
tabmini_path = os.path.abspath("./TabMini")
sys.path.append(tabmini_path)
import tabmini


os.makedirs("saved_models/mlp", exist_ok=True)
os.makedirs("results/mlp", exist_ok=True)

MODEL_SAVE_PATH = "saved_models/mlp"
RESULTS_CSV = "results/mlp/tabmini_model_comparison.csv"


def save_results(metrics, csv_path):
    df = pd.DataFrame([metrics])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(model, train_loader, val_loader, criterion, optimizer, epochs=50, model_path=None):
    best_loss = float("inf")
    early_stop_counter = 0  # Đếm số epochs không cải thiện
    early_stop_patience = 20  # Số epochs tối đa không cải thiện trước khi dừng

    # Scheduler để giảm learning rate nếu validation loss không giảm
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"✅ Loaded existing model: {model_path}")

    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            targets = targets.view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Tính validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_targets = val_targets.view(-1, 1)
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_targets.float())
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        # Giảm Learning Rate nếu cần
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Lưu model tốt nhất
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model: {model_path}")
            early_stop_counter = 0  # Reset bộ đếm early stopping
        else:
            early_stop_counter += 1

        # Dừng sớm nếu validation loss không cải thiện
        if early_stop_counter >= early_stop_patience:
            print("⏹ Early stopping triggered. Training stopped.")
            break


dataset_obj = tabmini.load_dataset(reduced=False)
results = []
for dataset_name, (X, y) in dataset_obj.items():
    X, y = X.values, LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)

    model = MLP(X.shape[1])
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.float32)),
                              batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.float32)),
                            batch_size=2, shuffle=False, drop_last=False)

    model_path = f"{MODEL_SAVE_PATH}/MLP_{dataset_name}.pth"
    # train_mlp(model, train_loader, nn.BCELoss(), optim.Adam(model.parameters(), lr=0.001), epochs=500,
    #           model_path=model_path)
    train_mlp(model, train_loader, val_loader, nn.BCELoss(), optim.Adam(model.parameters(), lr=0.001),
              epochs=500, model_path=model_path)

    # ✅ Đảm bảo load state_dict thay vì toàn bộ mô hình
    model = MLP(X.shape[1])  # Khởi tạo model trước khi load trọng số
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_prob = model(X_test_tensor).detach().numpy().flatten()
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    best_threshold = thresholds[np.argmax(precision * recall)]
    y_pred = (y_prob > best_threshold).astype(int)

    metrics = {
        "dataset": dataset_name,
        "model": "MLP",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=1),
        "recall": recall_score(y_test, y_pred, zero_division=1),
        "f1_score": f1_score(y_test, y_pred, zero_division=1),
        "auc": roc_auc_score(y_test, y_prob)
    }
    results.append(metrics)
  # print(f"✅ Saved results for {dataset_name} to {RESULTS_CSV}")

# Save final results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_CSV, index=False)
print("\n✅ Finished evaluation. Results saved to tabmini_model_comparison.csv")
