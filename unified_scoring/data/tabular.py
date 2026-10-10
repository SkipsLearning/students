"""Предобработка табличного источника.

* Категориальные признаки -> целочисленные индексы для Entity Embeddings
  (0 — редкое/неизвестное значение, 1 — пропуск, 2.. — частые значения).
* Числовые признаки -> (опционально signed log1p) -> импутация медианой ->
  StandardScaler или min-max. Все параметры считаются только на train.
* Состояние сериализуется в JSON и восстанавливается на инференсе.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import TabularConfig

UNKNOWN, MISSING, OFFSET = 0, 1, 2


def entity_embedding_dim(cardinality: int, max_dim: int) -> int:
    """Эвристика размерности Entity Embedding (fast.ai): 1.6 * card^0.56."""
    return int(max(2, min(max_dim, round(1.6 * cardinality ** 0.56), (cardinality + 1) // 2)))


class TabularPreprocessor:
    def __init__(self, cfg: TabularConfig) -> None:
        self.cfg = cfg
        self.categorical: list[str] = []
        self.numerical: list[str] = []
        self.vocab: dict[str, dict[str, int]] = {}
        self.log_cols: list[str] = []
        self.medians: dict[str, float] = {}
        self.loc: dict[str, float] = {}
        self.scale: dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        exclude = set(self.cfg.exclude)
        if self.cfg.categorical is not None:
            self.categorical = list(self.cfg.categorical)
        else:
            self.categorical = [
                c for c in df.columns
                if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])
            ]
        if self.cfg.numerical is not None:
            self.numerical = list(self.cfg.numerical)
        else:
            self.numerical = [
                c for c in df.columns
                if c not in exclude and c not in self.categorical
                and pd.api.types.is_numeric_dtype(df[c])
            ]
        for c in self.categorical:
            counts = df[c].dropna().astype(str).value_counts()
            frequent = counts[counts >= self.cfg.min_category_freq].index
            self.vocab[c] = {v: i + OFFSET for i, v in enumerate(sorted(frequent))}
        self.log_cols = [
            c for c in self.numerical if any(p in c for p in self.cfg.log_patterns)
        ]
        num = self._numeric_raw(df)
        for c in self.numerical:
            col = num[c]
            med = float(np.nanmedian(col)) if np.isfinite(col).any() else 0.0
            filled = np.where(np.isfinite(col), col, med)
            if self.cfg.scaler == "minmax":
                lo, hi = float(filled.min()), float(filled.max())
                self.loc[c], self.scale[c] = lo, (hi - lo) or 1.0
            else:
                self.loc[c], self.scale[c] = float(filled.mean()), float(filled.std()) or 1.0
            self.medians[c] = med
        return self

    # ------------------------------------------------------------ transform
    def _numeric_raw(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for c in self.numerical:
            if c in df.columns:
                col = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
            else:
                col = np.full(len(df), np.nan)
            col = np.where(np.isinf(col), np.nan, col)
            if c in self.log_cols:
                col = np.sign(col) * np.log1p(np.abs(col))
            out[c] = col
        return out

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        cat = np.zeros((n, len(self.categorical)), dtype=np.int64)
        for j, c in enumerate(self.categorical):
            if c not in df.columns:
                cat[:, j] = MISSING
                continue
            col = df[c]
            mapped = col.astype(str).map(self.vocab[c]).fillna(UNKNOWN).to_numpy(dtype=np.int64).copy()
            mapped[col.isna().to_numpy()] = MISSING
            cat[:, j] = mapped
        num = np.zeros((n, len(self.numerical)), dtype=np.float32)
        raw = self._numeric_raw(df)
        for j, c in enumerate(self.numerical):
            col = np.where(np.isfinite(raw[c]), raw[c], self.medians[c])
            num[:, j] = np.clip((col - self.loc[c]) / self.scale[c], -10, 10)
        return cat, num

    # ------------------------------------------------------------ metadata
    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + OFFSET for c in self.categorical]

    @property
    def embedding_dims(self) -> list[int]:
        return [entity_embedding_dim(k, self.cfg.max_emb_dim) for k in self.cardinalities]

    def state_dict(self) -> dict[str, Any]:
        return {
            "categorical": self.categorical,
            "numerical": self.numerical,
            "vocab": self.vocab,
            "log_cols": self.log_cols,
            "medians": self.medians,
            "loc": self.loc,
            "scale": self.scale,
            "max_emb_dim": self.cfg.max_emb_dim,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], cfg: TabularConfig) -> "TabularPreprocessor":
        obj = cls(cfg)
        for k in ("categorical", "numerical", "vocab", "log_cols", "medians", "loc", "scale"):
            setattr(obj, k, state[k])
        obj.cfg.max_emb_dim = state["max_emb_dim"]
        return obj
