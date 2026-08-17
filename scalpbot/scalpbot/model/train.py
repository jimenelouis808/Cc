"""Entrenamiento con walk-forward purgado y generacion de predicciones OOS.

Regla de oro: el backtest SOLO puede consumir las predicciones out-of-sample
que produce esta funcion. Backtestear con probabilidades in-sample da curvas
de equity preciosas y perdidas reales.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..features.builder import feature_columns
from ..labeling import Labels
from ..utils import get_logger
from .cv import PurgedWalkForward
from .registry import ModelBundle

log = get_logger("model.train")

CLASSES = [-1, 0, 1]
_CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:  # pragma: no cover - depende del entorno
    HAS_LGB = False
    log.warning("lightgbm no disponible; usando HistGradientBoosting de sklearn")


@dataclass
class TrainResult:
    bundle: ModelBundle
    oos: pd.DataFrame          # index=ts, columnas p_dn/p_flat/p_up + fold
    fold_report: pd.DataFrame
    importances: pd.Series


def _make_model(cfg: Config, n_classes: int):
    m = cfg.model
    if HAS_LGB:
        return lgb.LGBMClassifier(
            objective="multiclass", num_class=n_classes,
            learning_rate=m.learning_rate, num_leaves=m.num_leaves,
            max_depth=m.max_depth, n_estimators=m.n_estimators,
            min_child_samples=m.min_child_samples, subsample=m.subsample,
            subsample_freq=1, colsample_bytree=m.colsample_bytree,
            reg_lambda=m.reg_lambda, random_state=m.seed, n_jobs=-1, verbosity=-1,
        )
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=m.learning_rate, max_iter=m.n_estimators,
        max_leaf_nodes=m.num_leaves, max_depth=m.max_depth,
        min_samples_leaf=m.min_child_samples, l2_regularization=m.reg_lambda,
        random_state=m.seed,
    )


def _fit(model, X: pd.DataFrame, y: np.ndarray, w: np.ndarray):
    try:
        model.fit(X, y, sample_weight=w)
    except TypeError:  # algun estimador sin sample_weight
        model.fit(X, y)
    return model


def _proba_full(model, X: pd.DataFrame) -> np.ndarray:
    """predict_proba alineado siempre a las 3 clases de CLASSES."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(CLASSES)))
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, col]
    return out


def select_features(cfg: Config, X: pd.DataFrame, y: np.ndarray,
                    w: np.ndarray) -> list[str]:
    """Preselecciona features por importancia usando solo el primer tramo.

    Se hace UNA vez sobre la ventana de entrenamiento inicial para no filtrar
    informacion de los folds posteriores.
    """
    k = cfg.model.max_features
    if k <= 0 or k >= X.shape[1]:
        return list(X.columns)
    cut = min(len(X), max(cfg.model.min_train_bars, len(X) // 3))
    model = _make_model(cfg, len(CLASSES))
    _fit(model, X.iloc[:cut], y[:cut], w[:cut])
    imp = _importances(model, list(X.columns))
    chosen = list(imp.nlargest(k).index)
    log.info("seleccionadas %d/%d features por importancia", len(chosen), X.shape[1])
    return chosen


def _importances(model, names: list[str]) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=names, dtype=float)
    return pd.Series(1.0, index=names, dtype=float)


def train_walk_forward(cfg: Config, feats: pd.DataFrame, labels: Labels) -> TrainResult:
    """Entrena por folds, acumula predicciones OOS y ajusta el modelo final."""
    cols = feature_columns(feats)
    if not cols:
        raise ValueError("no hay columnas de features en el dataset")

    X_all = feats[cols].astype(np.float32)
    y_raw = labels.y.reindex(feats.index)
    w_all = labels.weights.reindex(feats.index).fillna(0.0).to_numpy()
    t1_ts = labels.t1.reindex(feats.index)

    # t1 viene como indice entero del dataframe ORIGINAL; lo reconvertimos a
    # posiciones dentro de `feats` (que empieza mas tarde por el warmup).
    pos = pd.Series(np.arange(len(feats)), index=feats.index)
    orig_index = labels.y.index
    t1_ts_dt = orig_index[np.clip(t1_ts.to_numpy(), 0, len(orig_index) - 1)]
    t1_pos = pos.reindex(t1_ts_dt).to_numpy()
    t1_pos = np.where(np.isnan(t1_pos), len(feats) - 1, t1_pos).astype(np.int64)

    y = y_raw.map(_CLASS_TO_IDX).to_numpy()
    mask = np.isfinite(y.astype(float)) & (w_all > 0)
    if mask.sum() < cfg.model.min_train_bars:
        raise ValueError(f"solo {int(mask.sum())} muestras validas; necesitas mas historico")

    cols = select_features(cfg, X_all[mask], y[mask], w_all[mask])
    X_all = X_all[cols]

    cv = PurgedWalkForward(n_splits=cfg.model.n_splits, embargo=cfg.model.embargo,
                           min_train=cfg.model.min_train_bars, expanding=True)

    oos_rows: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    imp_acc = pd.Series(0.0, index=cols)

    for sp in cv.split(len(feats), t1_pos):
        tr = sp.train_idx[mask[sp.train_idx]]
        te = sp.test_idx
        if len(tr) < 200 or len(np.unique(y[tr])) < 2:
            log.warning("fold %d omitido (train insuficiente)", sp.fold)
            continue

        model = _make_model(cfg, len(CLASSES))
        _fit(model, X_all.iloc[tr], y[tr], w_all[tr])
        proba = _proba_full(model, X_all.iloc[te])

        oos_rows.append(pd.DataFrame(
            proba, index=feats.index[te], columns=["p_dn", "p_flat", "p_up"]
        ).assign(fold=sp.fold))

        imp_acc += _importances(model, cols).reindex(cols).fillna(0.0)
        reports.append(_fold_metrics(sp.fold, y[te], proba, len(tr), len(te),
                                     feats.index[te]))
        log.info("fold %d | train=%d test=%d | acc_dir=%.3f logloss=%.4f",
                 sp.fold, len(tr), len(te), reports[-1]["dir_accuracy"],
                 reports[-1]["logloss"])

    if not oos_rows:
        raise RuntimeError("ningun fold pudo entrenarse; reduce min_train_bars o n_splits")

    oos = pd.concat(oos_rows).sort_index()
    fold_report = pd.DataFrame(reports)

    # Modelo final sobre todo el historico: es el que se usa en vivo.
    final = _make_model(cfg, len(CLASSES))
    _fit(final, X_all[mask], y[mask], w_all[mask])
    importances = (imp_acc / max(len(reports), 1)).sort_values(ascending=False)

    bundle = ModelBundle(
        model=final, features=cols, classes=CLASSES,
        meta={
            "symbol": cfg.data.symbol, "interval": cfg.data.interval,
            "market": cfg.data.market, "n_train": int(mask.sum()),
            "backend": "lightgbm" if HAS_LGB else "sklearn",
            "label_cfg": cfg.labels.__dict__, "model_cfg": cfg.model.__dict__,
            "oos_logloss": float(fold_report["logloss"].mean()),
            "oos_dir_accuracy": float(fold_report["dir_accuracy"].mean()),
            "top_features": list(importances.head(20).index),
        },
    )
    return TrainResult(bundle=bundle, oos=oos, fold_report=fold_report,
                       importances=importances)


def _fold_metrics(fold: int, y_true: np.ndarray, proba: np.ndarray,
                  n_train: int, n_test: int, index: pd.Index) -> dict[str, Any]:
    eps = 1e-9
    ll = float(-np.mean(np.log(np.clip(proba[np.arange(len(y_true)), y_true], eps, 1.0))))

    # Precision direccional: solo sobre las muestras que NO expiraron por tiempo,
    # comparando p_up vs p_dn. Es la metrica que de verdad importa para operar.
    directional = y_true != _CLASS_TO_IDX[0]
    if directional.sum() > 0:
        pred_up = proba[directional, 2] > proba[directional, 0]
        true_up = y_true[directional] == _CLASS_TO_IDX[1]
        dir_acc = float((pred_up == true_up).mean())
    else:
        dir_acc = float("nan")

    return {
        "fold": fold, "n_train": n_train, "n_test": n_test,
        "start": index[0], "end": index[-1],
        "logloss": ll, "dir_accuracy": dir_acc,
        "mean_p_up": float(proba[:, 2].mean()), "mean_p_dn": float(proba[:, 0].mean()),
    }
