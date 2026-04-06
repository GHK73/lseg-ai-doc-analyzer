# backend/ml/service.py

import torch
import numpy as np
import os
from .model import DocumentClassifier
from .utils import save_model, load_model


class ClassifierService:
    def __init__(self, input_dim, num_classes, lr=1e-3, model_path="ml_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        self.model = DocumentClassifier(input_dim, num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        if os.path.exists(self.model_path):
            load_model(self.model, self.model_path)

        self.input_dim = input_dim

    # 🔴 critical for FAISS consistency
    def _normalize(self, x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)

    def train(self, x, y, epochs=5, batch_size=32):
        x = np.array(x)
        y = np.array(y)

        # validation
        assert x.shape[1] == self.input_dim, "Input dimension mismatch"

        x = self._normalize(x)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()

        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                logits = self.model(xb)
                loss = self.criterion(logits, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        save_model(self.model, self.model_path)
        return loss.item()

    def predict(self, x):
        self.model.eval()

        x = np.array(x)

        # handle single sample
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # validation
        assert x.shape[1] == self.input_dim, "Input dimension mismatch"

        x = self._normalize(x)

        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        return {
            "predictions": preds.cpu().tolist(),
            "probabilities": probs.cpu().tolist()
        }