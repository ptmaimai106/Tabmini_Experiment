import os
import torch
import torch.nn as nn
import torch.optim as optim

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mlp_plr import MLPPLR  # Thư viện MLP-PLR

# Thư mục lưu mô hình
os.makedirs("saved_models/mlp", exist_ok=True)




class MLP_PLR(nn.Module):
    def __init__(self, input_dim):
        super(MLP_PLR, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Binary classification

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

model = MLP_PLR(input_dim=X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(50):  # Train for 50 epochs
    optimizer.zero_grad()
    output = model(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion(output.squeeze(), torch.tensor(y_train, dtype=torch.float32))
    loss.backward()
    optimizer.step()
