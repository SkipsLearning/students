"""Базлайны для сравнения с единой моделью.

1. ``ScoreVectorBaseline`` — классическая схема «ансамбль скоров»: по каждому
   источнику доменная модель выдаёт ОДНО число (скор), затем логистическая
   регрессия на векторе скоров + флагах наличия. Скор доменной модели
   эмулируется логистической регрессией поверх её эмбеддинга — это ровно
   последний линейный слой доменной модели. Доменные скореры обучаются на
   отдельной части ``domain_fit``, чтобы скоры на train были out-of-sample.
2. ``EmbeddingLRBaseline`` — логистическая регрессия на тех же
   конкатенированных эмбеддингах (показывает вклад нелинейности MLP).
3. ``BoostingBaseline`` — градиентный бустинг (HistGradientBoosting) на том же
   входе, что у линейного базлайна: на векторе скоров или на эмбеддингах.
   Сильный нелинейный конкурент: проверяет, что прирост единой модели не
   объясняется только тем, что LR — слабый смеситель.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SourceConfig
from .data.container import ScoringData
from .data.tabular import TabularPreprocessor
from .metrics import gini


class _TabularDesign:
    """Табличные признаки для линейных моделей: one-hot категорий + числа."""

    def __init__(self, src: SourceConfig) -> None:
        self.pre = TabularPreprocessor(src.tabular)
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)

    def fit(self, data: ScoringData) -> "_TabularDesign":
        self.pre.fit(data.tabular)
        cat, _ = self.pre.transform(data.tabular)
        self.ohe.fit(cat if cat.shape[1] else np.zeros((len(cat), 1), dtype=np.int64))
        return self

    def transform(self, data: ScoringData) -> np.ndarray:
        cat, num = self.pre.transform(data.tabular)
        oh = self.ohe.transform(cat if cat.shape[1] else np.zeros((len(cat), 1), dtype=np.int64))
        return np.hstack([oh, num]).astype(np.float32)


def _fit_lr(X: np.ndarray, y: np.ndarray, C: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=3000).fit(scaler.transform(X), y)
    return scaler, lr


class DomainScorers:
    """Скалярные скоры источников (logit), NaN — если источника нет."""

    def __init__(self, sources: list[SourceConfig], C: float = 0.5) -> None:
        self.sources = sources
        self.C = C
        self.models: dict[str, tuple[StandardScaler, LogisticRegression]] = {}
        self.tab: _TabularDesign | None = None

    def _features(self, src: SourceConfig, data: ScoringData) -> np.ndarray:
        if src.type == "tabular":
            assert self.tab is not None
            return self.tab.transform(data)
        return data.embeddings[src.name]

    def fit(self, data: ScoringData, target: str) -> "DomainScorers":
        y_all = data.targets[target]
        for src in self.sources:
            if src.type == "tabular":
                self.tab = _TabularDesign(src).fit(data)
            m = data.masks[src.name]
            self.models[src.name] = _fit_lr(self._features(src, data)[m], y_all[m], self.C)
        return self

    def score(self, data: ScoringData) -> dict[str, np.ndarray]:
        out = {}
        for src in self.sources:
            scaler, lr = self.models[src.name]
            s = lr.decision_function(scaler.transform(self._features(src, data)))
            out[src.name] = np.where(data.masks[src.name], s, np.nan)
        return out


class ScoreVectorBaseline:
    name = "LR на векторе скоров"

    def __init__(self, sources: list[SourceConfig], target: str, logger: logging.Logger) -> None:
        self.sources = sources
        self.target = target
        self.logger = logger
        self.scorers = DomainScorers(sources)
        self.fill: dict[str, float] = {}

    def design(self, data: ScoringData) -> np.ndarray:
        scores = self.scorers.score(data)
        cols = []
        for src in self.sources:
            cols.append(np.where(np.isnan(scores[src.name]), self.fill[src.name], scores[src.name]))
        for src in self.sources:
            cols.append(data.masks[src.name].astype(np.float64))
        return np.column_stack(cols)

    def fit(self, domain: ScoringData, train: ScoringData) -> "ScoreVectorBaseline":
        if len(domain) == 0:
            self.logger.warning("domain_fit пуст: доменные скоры обучаются на train (in-sample, утечка)")
            domain = train
        self.scorers.fit(domain, self.target)
        train_scores = self.scorers.score(train)
        self.fill = {k: float(np.nanmean(v)) for k, v in train_scores.items()}
        self.scaler, self.lr = _fit_lr(self.design(train), train.targets[self.target], C=1.0)
        return self

    def predict_pd(self, data: ScoringData) -> np.ndarray:
        return self.lr.predict_proba(self.scaler.transform(self.design(data)))[:, 1]


class EmbeddingLRBaseline:
    name = "LR на эмбеддингах"

    def __init__(self, sources: list[SourceConfig], target: str, C: float = 0.05) -> None:
        self.sources = sources
        self.target = target
        self.C = C
        self.means: dict[str, np.ndarray] = {}
        self.tab: _TabularDesign | None = None

    def design(self, data: ScoringData) -> np.ndarray:
        blocks = []
        for src in self.sources:
            m = data.masks[src.name][:, None]
            x = self.tab.transform(data) if src.type == "tabular" else data.embeddings[src.name]
            blocks.append(np.where(m, x, self.means[src.name]))
        blocks.append(np.column_stack([data.masks[s.name] for s in self.sources]).astype(np.float32))
        return np.hstack(blocks)

    def fit(self, train: ScoringData) -> "EmbeddingLRBaseline":
        for src in self.sources:
            if src.type == "tabular":
                self.tab = _TabularDesign(src).fit(train)
                x = self.tab.transform(train)
            else:
                x = train.embeddings[src.name]
            m = train.masks[src.name]
            self.means[src.name] = x[m].mean(axis=0) if m.any() else np.zeros(x.shape[1], np.float32)
        self.scaler, self.lr = _fit_lr(self.design(train), train.targets[self.target], self.C)
        return self

    def predict_pd(self, data: ScoringData) -> np.ndarray:
        return self.lr.predict_proba(self.scaler.transform(self.design(data)))[:, 1]


class BoostingBaseline:
    """HistGradientBoosting на заданной матрице признаков.

    Число деревьев подбирается по Джини на val — так же, как early stopping
    единой модели, чтобы сравнение было честным. Деревья достраиваются
    порциями, пока лучшая итерация не окажется заметно раньше конца:
    иначе на больших данных бустинг был бы недообучен.
    """

    STEP_TREES = 400
    MAX_TREES = 4000

    def __init__(self, name: str, design: Callable[[ScoringData], np.ndarray], target: str, seed: int,
                 learning_rate: float = 0.05) -> None:
        self.name = name
        self.design = design
        self.target = target
        self.params = dict(learning_rate=learning_rate, max_leaf_nodes=15, min_samples_leaf=50,
                           l2_regularization=1.0, early_stopping=False, random_state=seed)
        self.best_trees = 0

    def fit(self, train: ScoringData, val: ScoringData) -> "BoostingBaseline":
        X_tr, y_tr = self.design(train), train.targets[self.target]
        X_va, y_va = self.design(val), val.targets[self.target]
        clf = HistGradientBoostingClassifier(max_iter=self.STEP_TREES, warm_start=True, **self.params)
        val_gini: list[float] = []
        while True:
            clf.fit(X_tr, y_tr)  # warm_start: дообучает уже построенный ансамбль
            for i, p in enumerate(clf.staged_predict_proba(X_va)):
                if i >= len(val_gini):
                    val_gini.append(gini(y_va, p[:, 1]))
            self.best_trees = int(np.argmax(val_gini)) + 1
            if self.best_trees <= 0.9 * clf.max_iter or clf.max_iter >= self.MAX_TREES:
                break
            clf.set_params(max_iter=clf.max_iter + self.STEP_TREES)
        self.model = clf
        return self

    def predict_pd(self, data: ScoringData) -> np.ndarray:
        """Прогноз ансамбля из первых best_trees деревьев (лучшая итерация по val)."""
        for i, p in enumerate(self.model.staged_predict_proba(self.design(data)), start=1):
            if i == self.best_trees:
                return p[:, 1]
        raise RuntimeError("best_trees больше числа построенных деревьев")
