# backend/ml/service.py

import torch
from .model import DocumentClassifier


class ClassifierService:
    def __init__(self, input_dim, num_classes, lr=1e-3):
        self.device = torch.device("cpu")

        self.model = DocumentClassifier(input_dim, num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, x, y, epochs=5):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)

        for _ in range(epochs):
            self.model.train()

            logits = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def predict(self, x):
        self.model.eval()

        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1)

        return preds.cpu().tolist()