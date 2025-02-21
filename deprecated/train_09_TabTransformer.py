import os
import torch
import pandas as pd
from pytorch_widedeep.models import TabTransformer
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

os.makedirs("saved_models/tab_transformer", exist_ok=True)


def train_tab_transformer(X_train, X_test, y_train, y_test, dataset_name):
    print(f"Training TabTransformer on {dataset_name}...")

    model_filename = f"saved_models/tab_transformer/{dataset_name}_tabtransformer.pth"
    if os.path.exists(model_filename):
        print(f"Loading saved model: {model_filename}")
        tab_transformer = torch.load(model_filename)
    else:
        tab_transformer = TabTransformer(input_dim=X_train.shape[1], mlp_hidden_dims=[64, 32])
        trainer = Trainer(model=tab_transformer, objective="binary", callbacks=[EarlyStopping(patience=3)])
        trainer.fit(X_train, y_train, X_test, y_test, n_epochs=50, batch_size=32)
        torch.save(tab_transformer, model_filename)

    y_pred = trainer.predict(X_test).argmax(axis=1)
    y_prob = trainer.predict_proba(X_test)[:, 1]

    results = {
        "dataset": dataset_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
    }

    return results
