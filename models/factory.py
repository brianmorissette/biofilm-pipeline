# src/models/factory.py
from .cnn import CNN

def build_model(architecture: str, **kwargs):
    if architecture == "cnn":
        return CNN(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
