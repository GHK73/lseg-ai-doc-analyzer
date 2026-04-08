# backend/ml/utils.py

import torch
import os


def save_model(model, path: str):
    try:
        dir_name = os.path.dirname(path)

        if dir_name:  # avoid "" issue
            os.makedirs(dir_name, exist_ok=True)

        torch.save(model.state_dict(), path)
        print(f"💾 Model saved at {path}")

    except Exception as e:
        print(f"❌ Error saving model: {e}")


def load_model(model, path: str, device=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")

    try:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        print(f"✅ Model loaded from {path} on {device}")

        return model

    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")