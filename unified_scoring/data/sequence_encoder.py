"""Доменные модели для последовательностей (этап 1 двухэтапной схемы).

Для каждого источника-последовательности (кредиты в БКИ, месяцы по карте,
платежи по кредитам, ...) обучается GRU-модель на PD. Эмбеддинг клиента
снимается с предпоследнего полносвязного слоя — ровно так, как описано в
постановке: «эмбеддинги с последних полносвязных слоёв обученных доменных
моделей». Эти эмбеддинги и подаются в единую модель.
"""
from __future__ import annotations

import copy
import logging

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..metrics import gini


class CategoricalInput(nn.Module):
    """Событие из категориальных кодов: у каждого признака свой обучаемый эмбеддинг,
    их конкатенация проецируется в вектор размерности hidden."""

    def __init__(self, cardinalities: list[int], hidden: int, max_emb_dim: int = 4) -> None:
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(c, min(max_emb_dim, c)) for c in cardinalities)
        self.proj = nn.Sequential(nn.Linear(sum(e.embedding_dim for e in self.embs), hidden), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        return self.proj(torch.cat([emb(x[..., j]) for j, emb in enumerate(self.embs)], dim=-1))


class SequenceEncoder(nn.Module):
    def __init__(self, n_features: int, emb_dim: int, hidden: int = 64,
                 cardinalities: list[int] | None = None) -> None:
        super().__init__()
        # числовые признаки событий -> Linear; категориальные коды -> эмбеддинги
        self.inp = (CategoricalInput(cardinalities, hidden) if cardinalities
                    else nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU()))
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        # предпоследний слой -> эмбеддинг размерности emb_dim
        self.proj = nn.Sequential(
            nn.Linear(3 * hidden, max(emb_dim, hidden)), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(max(emb_dim, hidden), emb_dim), nn.ReLU(),
        )
        self.head = nn.Linear(emb_dim, 1)

    def embed(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, last = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        mask = (torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        mean = (out * mask).sum(1) / lengths[:, None].clamp(min=1)
        mx = out.masked_fill(~mask, -1e4).max(1).values
        return self.proj(torch.cat([mean, mx, last[-1]], dim=1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x, lengths)).squeeze(-1)


def _batches(n: int, batch_size: int, rng: np.random.Generator | None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i : i + batch_size]


@torch.no_grad()
def encode(model: SequenceEncoder, X: np.ndarray, lengths: np.ndarray,
           batch_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (эмбеддинги, логиты доменной модели)."""
    model.eval()
    embs, logits = [], []
    for b in _batches(len(X), batch_size, None):
        e = model.embed(torch.as_tensor(X[b]), torch.as_tensor(lengths[b]))
        embs.append(e.numpy())
        logits.append(model.head(e).squeeze(-1).numpy())
    return np.concatenate(embs).astype(np.float32), np.concatenate(logits)


def fit_sequence_encoder(
    X: np.ndarray,
    lengths: np.ndarray,
    y: np.ndarray,
    fit_rows: np.ndarray,
    emb_dim: int,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    logger: logging.Logger,
    name: str,
    cardinalities: list[int] | None = None,
) -> SequenceEncoder:
    """Обучает доменную модель на строках ``fit_rows`` (внутренний holdout 15% для отбора эпохи)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, ho = train_test_split(fit_rows, test_size=0.15, stratify=y[fit_rows], random_state=seed)
    model = SequenceEncoder(X.shape[2], emb_dim, hidden, cardinalities)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    best, best_state = -np.inf, copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        model.train()
        for b in _batches(len(tr), batch_size, rng):
            rows = tr[b]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(torch.as_tensor(X[rows]), torch.as_tensor(lengths[rows])),
                        torch.as_tensor(y[rows], dtype=torch.float32))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        _, ho_logits = encode(model, X[ho], lengths[ho])
        g = gini(y[ho], ho_logits)
        logger.info("  [%s] доменная модель, эпоха %d: holdout Gini %.2f%%", name, epoch, g)
        if g > best:
            best, best_state = g, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model
