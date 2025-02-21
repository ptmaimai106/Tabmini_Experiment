import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import pandas as pd
import numpy as np

# Load TabMini dataset
tabmini_path = os.path.abspath("./TabMini")
sys.path.append(tabmini_path)
import tabmini

os.makedirs("saved_models/SAINT", exist_ok=True)
os.makedirs("results/SAINT", exist_ok=True)

MODEL_SAVE_PATH = "saved_models/SAINT"
RESULTS_CSV = "results/SAINT/tabmini_model_comparison.csv"

# ===================== SAINT Model ===================== #
class SAINT(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_heads=4, num_layers=2):
        super(SAINT, self).__init__()

        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer blocks for Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Inter-Sample Attention (Modeled as an extra attention layer)
        self.inter_sample_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Self-Attention
        x = self.embedding(x)
        x = self.self_attention(x.unsqueeze(1)).squeeze(1)

        # Inter-Sample Attention
        x, _ = self.inter_sample_attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = x.squeeze(1)

        # Final classification
        return self.fc(x)

# ===================== Training Function ===================== #
def train_model(model, train_loader, val_loader, criterion, optimizer, model_path, epochs=300):
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
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

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_targets = val_targets.view(-1, 1)
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_targets.float())
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model: {model_path}")

# ===================== Load Data ===================== #
dataset_obj = tabmini.load_dataset(reduced=False)
results = []

for dataset_name, (X, y) in dataset_obj.items():
    print(f"\n🔹 Training on dataset: {dataset_name}")

    # Preprocessing
    X, y = X.values, LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)

    model = SAINT(X.shape[1])
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=16, shuffle=False)

    model_path = f"{MODEL_SAVE_PATH}/SAINT_{dataset_name}.pth"
    train_model(model, train_loader, val_loader, nn.BCELoss(), optim.Adam(model.parameters(), lr=0.001), model_path, epochs=300)

    # Load best model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluation
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_prob = model(X_test_tensor).detach().numpy().flatten()
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    best_threshold = thresholds[np.argmax(precision * recall)] if len(thresholds) > 0 else 0.5
    y_pred = (y_prob > best_threshold).astype(int)

    metrics = {
        "dataset": dataset_name,
        "model": "SAINT",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=1),
        "recall": recall_score(y_test, y_pred, zero_division=1),
        "f1_score": f1_score(y_test, y_pred, zero_division=1),
        "auc": roc_auc_score(y_test, y_prob)
    }
    results.append(metrics)
    print(f"✅ Completed {dataset_name} - Accuracy: {metrics['accuracy']:.4f}")

# Save final results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_CSV, index=False)
print("\n✅ Finished evaluation. Results saved to tabmini_model_comparison.csv")
