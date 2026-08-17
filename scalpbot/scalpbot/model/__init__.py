"""Modelado: CV purgada, entrenamiento walk-forward y registro de modelos."""
from .registry import ModelBundle
from .train import TrainResult, train_walk_forward

__all__ = ["ModelBundle", "TrainResult", "train_walk_forward"]
