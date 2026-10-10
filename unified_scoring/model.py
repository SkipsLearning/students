"""Единая нейросетевая модель кредитного скоринга.

Схема::

    эмбеддинг источника 1 ─┐ (нет источника -> обучаемый «средний» эмбеддинг)
        ... tower_1: Linear-ReLU-Dropout-Linear-ReLU ─┐
    эмбеддинг источника k ─┤                          ├─ concat ─┬─ [флаги наличия]
    табличные признаки ─ EntityEmb ─ tower_tab ───────┘          │
                                                                 ▼
                         head: Linear-ReLU-Dropout-Linear-ReLU-Linear(1) -> logit -> σ -> PD

Модель строится по списку ``SourceSpec``: новый источник = новый элемент списка
(новый tower), код модели не меняется.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

Batch = dict[str, Any]


@dataclass
class SourceSpec:
    name: str
    kind: str                 # embedding | tabular
    dim: int                  # размерность входного эмбеддинга источника
    tower_hidden: list[int]
    tower_dropout: float
    # только для tabular:
    cardinalities: list[int] = field(default_factory=list)
    cat_emb_dims: list[int] = field(default_factory=list)
    n_numeric: int = 0


def mlp(in_dim: int, hidden: list[int], dropout: float) -> nn.Sequential:
    """Linear -> ReLU -> Dropout -> Linear -> ReLU -> ... (без Dropout после последнего слоя)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for i, h in enumerate(hidden):
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if i < len(hidden) - 1 and dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    return nn.Sequential(*layers)


class TabularEncoder(nn.Module):
    """Entity Embeddings категорий + нормализованные числа -> вектор размерности dim."""

    def __init__(self, cardinalities: list[int], emb_dims: list[int], n_numeric: int, out_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(c, d) for c, d in zip(cardinalities, emb_dims))
        in_dim = sum(emb_dims) + n_numeric
        if in_dim == 0:
            raise ValueError("Табличный источник без признаков")
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        parts = [emb(cat[:, j]) for j, emb in enumerate(self.embeddings)]
        if num.shape[1] > 0:
            parts.append(num)
        return self.proj(torch.cat(parts, dim=1))


class UnifiedScoringModel(nn.Module):
    def __init__(
        self,
        specs: list[SourceSpec],
        head_hidden: list[int],
        head_dropout: float,
        use_presence_flags: bool = True,
        source_dropout: float = 0.0,
        missing_embedding: str = "learned_mean",
    ) -> None:
        super().__init__()
        if not specs:
            raise ValueError("Нужен хотя бы один источник")
        self.specs = specs
        self.names = [s.name for s in specs]
        self.use_presence_flags = use_presence_flags
        self.source_dropout = source_dropout
        self.missing_embedding = missing_embedding

        self.encoders = nn.ModuleDict()
        self.towers = nn.ModuleDict()
        self.default_emb = nn.ParameterDict()
        for s in specs:
            if s.kind == "tabular":
                self.encoders[s.name] = TabularEncoder(s.cardinalities, s.cat_emb_dims, s.n_numeric, s.dim)
            # константный эмбеддинг-заглушка для клиентов без источника;
            # табличный всегда обучаемый (среднее выхода энкодера заранее неизвестно)
            trainable = missing_embedding == "learned_mean" or s.kind == "tabular"
            self.default_emb[s.name] = nn.Parameter(torch.zeros(s.dim), requires_grad=trainable)
            self.towers[s.name] = mlp(s.dim, s.tower_hidden, s.tower_dropout)

        head_in = sum(s.tower_hidden[-1] for s in specs) + (len(specs) if use_presence_flags else 0)
        self.head = nn.Sequential(mlp(head_in, head_hidden, head_dropout), nn.Linear(head_hidden[-1], 1))

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def init_default_embeddings(self, means: dict[str, np.ndarray]) -> None:
        """Инициализирует заглушки средним эмбеддингом источника по train (где он есть)."""
        for name, vec in means.items():
            if name in self.default_emb:
                self.default_emb[name].copy_(torch.as_tensor(vec, dtype=torch.float32))

    def _apply_source_dropout(self, masks: torch.Tensor) -> torch.Tensor:
        """При обучении случайно «прячет» доступные источники (но оставляет хотя бы один),
        чтобы модель была устойчива к любому набору источников на инференсе."""
        drop = torch.rand_like(masks, dtype=torch.float32) < self.source_dropout
        new = masks & ~drop
        empty = new.sum(dim=1) == 0
        return torch.where(empty.unsqueeze(1), masks, new)

    def forward(self, batch: Batch) -> torch.Tensor:
        masks = torch.stack([batch["mask"][n] for n in self.names], dim=1).bool()
        if self.training and self.source_dropout > 0:
            masks = self._apply_source_dropout(masks)
        outs = []
        for i, s in enumerate(self.specs):
            if s.kind == "tabular":
                x = self.encoders[s.name](batch["cat"], batch["num"])
            else:
                x = batch["emb"][s.name]
            m = masks[:, i : i + 1].to(x.dtype)
            x = m * x + (1.0 - m) * self.default_emb[s.name]
            outs.append(self.towers[s.name](x))
        if self.use_presence_flags:
            outs.append(masks.float())
        return self.head(torch.cat(outs, dim=1)).squeeze(-1)

    @torch.no_grad()
    def predict_pd(self, batch: Batch) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(batch))

    # ----------------------------------------------------------------------
    def build_kwargs(self) -> dict[str, Any]:
        return {
            "specs": [asdict(s) for s in self.specs],
            "head_hidden": [m.out_features for m in self.head[0] if isinstance(m, nn.Linear)],
            "head_dropout": next((m.p for m in self.head[0] if isinstance(m, nn.Dropout)), 0.0),
            "use_presence_flags": self.use_presence_flags,
            "source_dropout": self.source_dropout,
            "missing_embedding": self.missing_embedding,
        }

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> "UnifiedScoringModel":
        kw = dict(kwargs)
        kw["specs"] = [SourceSpec(**s) for s in kw["specs"]]
        return cls(**kw)


def load_matching_weights(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Warm start: переносит веса с совпадающими именами и формами (например, towers
    старых источников при добавлении нового). Возвращает список перенесённых ключей."""
    own = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    own.update(matched)
    model.load_state_dict(own)
    return sorted(matched)
