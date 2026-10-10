"""Torch-датасет и даталоадер.

Датасет поддерживает выборку сразу батчем индексов (``__getitem__(list[int])``),
а DataLoader получает BatchSampler — это на порядок быстрее поэлементной
сборки для табличных/векторных данных.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from .container import ScoringData
from .tabular import TabularPreprocessor


class ScoringTorchDataset(Dataset):
    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        cat: np.ndarray | None,
        num: np.ndarray | None,
        y: np.ndarray | None,
    ) -> None:
        self.emb = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in embeddings.items()}
        self.mask = {k: torch.as_tensor(v, dtype=torch.bool) for k, v in masks.items()}
        n = len(next(iter(self.mask.values())))
        self.cat = torch.as_tensor(cat if cat is not None else np.zeros((n, 0)), dtype=torch.long)
        self.num = torch.as_tensor(num if num is not None else np.zeros((n, 0)), dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: list[int] | torch.Tensor) -> dict[str, Any]:
        idx = torch.as_tensor(idx, dtype=torch.long)
        batch: dict[str, Any] = {
            "emb": {k: v[idx] for k, v in self.emb.items()},
            "mask": {k: v[idx] for k, v in self.mask.items()},
            "cat": self.cat[idx],
            "num": self.num[idx],
        }
        if self.y is not None:
            batch["y"] = self.y[idx]
        return batch


def build_dataset(
    data: ScoringData,
    preprocessor: TabularPreprocessor | None,
    target: str | None = "main",
) -> ScoringTorchDataset:
    cat = num = None
    if preprocessor is not None and data.tabular is not None:
        cat, num = preprocessor.transform(data.tabular)
    y = None if target is None else data.targets[target]
    return ScoringTorchDataset(data.embeddings, data.masks, cat, num, y)


def make_loader(ds: ScoringTorchDataset, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    if shuffle:
        sampler = RandomSampler(ds, generator=torch.Generator().manual_seed(seed))
    else:
        sampler = SequentialSampler(ds)
    return DataLoader(ds, sampler=BatchSampler(sampler, batch_size, drop_last=False), batch_size=None)


def batch_to(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out
