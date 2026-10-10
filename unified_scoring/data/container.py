"""Единый контейнер данных и разбиение train/val/test."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import Config, SplitConfig

SPLIT_NAMES = ("domain_fit", "train", "val", "test")


@dataclass
class ScoringData:
    """Данные на уровне клиента (одна строка = одна заявка).

    embeddings[name] — float32 (n, dim), нули там, где источника нет;
    masks[name]      — bool (n,), наличие источника (включая табличный);
    tabular          — сырые табличные признаки (обрабатываются TabularPreprocessor);
    extra_scores     — готовые логиты внешних моделей (например, доменной GRU) для сравнения;
    targets          — {"main": K-дефолт, "short": короткое определение (если есть)};
    period           — номер периода выдачи (для out-of-time), -1 если неизвестен.
    """

    client_id: np.ndarray
    period: np.ndarray
    targets: dict[str, np.ndarray]
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    tabular: pd.DataFrame | None = None
    extra_scores: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.client_id)

    def subset(self, idx: np.ndarray) -> "ScoringData":
        return ScoringData(
            client_id=self.client_id[idx],
            period=self.period[idx],
            targets={k: v[idx] for k, v in self.targets.items()},
            embeddings={k: v[idx] for k, v in self.embeddings.items()},
            masks={k: v[idx] for k, v in self.masks.items()},
            tabular=None if self.tabular is None else self.tabular.iloc[idx].reset_index(drop=True),
            extra_scores={k: v[idx] for k, v in self.extra_scores.items()},
        )

    def availability(self) -> pd.Series:
        """Доля клиентов, у которых есть каждый источник."""
        return pd.Series({k: float(v.mean()) for k, v in self.masks.items()})


def time_splits(period: np.ndarray, cfg: SplitConfig) -> dict[str, np.ndarray]:
    """Out-of-time: интервалы периодов [from, to] включительно."""
    bounds = {
        "domain_fit": cfg.domain_fit_periods,
        "train": cfg.train_periods,
        "val": cfg.val_periods,
        "test": cfg.test_periods,
    }
    out = {}
    for name, (lo, hi) in bounds.items():
        out[name] = np.flatnonzero((period >= lo) & (period <= hi))
    # контроль отсутствия пересечений и порядка во времени
    for a, b in zip(SPLIT_NAMES[:-1], SPLIT_NAMES[1:]):
        if bounds[a][1] >= bounds[b][0]:
            raise ValueError(f"Периоды {a} и {b} пересекаются или нарушен порядок времени")
    return out


def random_splits(y: np.ndarray, fractions: dict[str, float], seed: int) -> dict[str, np.ndarray]:
    """Стратифицированное случайное разбиение (когда дат выдачи нет)."""
    total = sum(fractions.get(k, 0.0) for k in SPLIT_NAMES)
    if not np.isclose(total, 1.0):
        raise ValueError(f"Сумма долей split.fractions должна быть 1, получено {total}")
    rest = np.arange(len(y))
    out: dict[str, np.ndarray] = {}
    remaining = 1.0
    for name in SPLIT_NAMES[:-1]:
        frac = fractions.get(name, 0.0)
        if frac <= 0:
            out[name] = np.array([], dtype=np.int64)
            continue
        take, rest = train_test_split(
            rest, train_size=frac / remaining, stratify=y[rest], random_state=seed
        )
        out[name] = np.sort(take)
        remaining -= frac
    out[SPLIT_NAMES[-1]] = np.sort(rest)
    return out


def make_splits(data: ScoringData, cfg: Config) -> dict[str, np.ndarray]:
    if cfg.split.scheme == "time":
        if (data.period < 0).any():
            raise ValueError("Для split.scheme=time нужны периоды выдачи")
        return time_splits(data.period, cfg.split)
    return random_splits(data.targets["main"], cfg.split.fractions, cfg.experiment.seed)
