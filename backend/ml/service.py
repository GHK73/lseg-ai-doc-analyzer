# backend/ml/service.py

import torch
import numpy as np
import os

from .model import DocumentClassifier
from .utils import save_model, load_model


class ClassifierService:
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        model_path: str = "ml_model.pt"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.input_dim = input_dim

        self.model = DocumentClassifier(input_dim, num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        # ---- Load existing model if available ----
        if os.path.exists(self.model_path):
            print("✅ Loading existing model...")
            load_model(self.model, self.model_path)
        else:
            print("🚀 Initializing new model...")

    # -------- Normalize (important for consistency with embeddings) --------
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)

    # -------- Training --------
    def train(self, x, y, epochs=5, batch_size=32):
        x = np.array(x)
        y = np.array(y)

        if x.shape[1] != self.input_dim:
            raise ValueError("Input dimension mismatch")

        x = self._normalize(x)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        final_loss = 0

        for epoch in range(epochs):
            epoch_loss = 0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                logits = self.model(xb)
                loss = self.criterion(logits, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            final_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch+1}: Loss = {final_loss:.4f}")

        save_model(self.model, self.model_path)

        return {
            "final_loss": final_loss,
            "epochs": epochs
        }

    # -------- Prediction --------
    def predict(self, x):
        self.model.eval()

        x = np.array(x)

        # Handle single sample
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != self.input_dim:
            raise ValueError("Input dimension mismatch")

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

    # -------- Optional: Load model manually --------
    def load(self):
        if os.path.exists(self.model_path):
            load_model(self.model, self.model_path)
            print("✅ Model loaded")
        else:
            print("⚠️ No saved model found")

    # -------- Optional: Save manually --------
    def save(self):
        save_model(self.model, self.model_path)
        print("💾 Model saved")