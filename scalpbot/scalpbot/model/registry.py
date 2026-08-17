"""Persistencia de modelos entrenados y sus metadatos."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils import ensure_dir, get_logger

log = get_logger("model.registry")


@dataclass
class ModelBundle:
    """Todo lo necesario para predecir en produccion."""
    model: Any
    features: list[str]
    classes: list[int]                       # orden de columnas de predict_proba
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        ensure_dir(p.parent)
        with p.open("wb") as fh:
            pickle.dump({"model": self.model, "features": self.features,
                         "classes": self.classes, "meta": self.meta}, fh)
        p.with_suffix(".json").write_text(json.dumps(
            {"features": self.features, "classes": self.classes, "meta": self.meta},
            indent=2, default=str))
        log.info("modelo guardado en %s", p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"no existe el modelo {p}; ejecuta 'train' primero")
        with p.open("rb") as fh:
            d = pickle.load(fh)
        return cls(model=d["model"], features=d["features"],
                   classes=d["classes"], meta=d.get("meta", {}))
