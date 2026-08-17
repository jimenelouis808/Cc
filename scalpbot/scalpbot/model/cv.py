"""Validacion cruzada temporal purgada con embargo.

Un K-Fold normal es catastrofico en series financieras: mezcla futuro y pasado
y produce Sharpes de fantasia. Aqui:

  * Walk-forward: el test siempre esta DESPUES del train.
  * Purga: se eliminan del train las muestras cuya barrera (t1) invade el test.
  * Embargo: se descartan barras extra tras el test para cortar la
    autocorrelacion serial residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Split:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


class PurgedWalkForward:
    """Genera splits expansivos (anchored) o rodantes."""

    def __init__(self, n_splits: int = 6, embargo: int = 30,
                 min_train: int = 1000, expanding: bool = True):
        if n_splits < 2:
            raise ValueError("n_splits debe ser >= 2")
        self.n_splits = n_splits
        self.embargo = embargo
        self.min_train = min_train
        self.expanding = expanding

    def split(self, n_samples: int, t1: np.ndarray | None = None) -> Iterator[Split]:
        """t1: indice entero de la barra en que expira cada etiqueta."""
        if t1 is None:
            t1 = np.arange(n_samples)
        t1 = np.asarray(t1, dtype=np.int64)

        usable = n_samples - self.min_train
        if usable <= self.n_splits:
            raise ValueError(
                f"historico insuficiente: {n_samples} barras con min_train="
                f"{self.min_train} y n_splits={self.n_splits}")
        fold_size = usable // self.n_splits

        for k in range(self.n_splits):
            test_start = self.min_train + k * fold_size
            test_end = test_start + fold_size if k < self.n_splits - 1 else n_samples
            if test_end <= test_start:
                continue

            train_end = test_start
            train_start = 0 if self.expanding else max(0, train_end - self.min_train)
            train_idx = np.arange(train_start, train_end)

            # Purga: fuera las muestras cuyo horizonte entra en el test.
            keep = t1[train_idx] < (test_start - self.embargo)
            train_idx = train_idx[keep]
            if len(train_idx) < self.min_train // 2:
                continue

            yield Split(fold=k, train_idx=train_idx,
                        test_idx=np.arange(test_start, test_end))
