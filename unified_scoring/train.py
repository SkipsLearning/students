"""Тренировочный цикл: BCEWithLogitsLoss + Adam, early stopping по Джини на val."""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from .config import TrainConfig
from .data.torch_dataset import ScoringTorchDataset, batch_to, make_loader
from .metrics import gini
from .model import UnifiedScoringModel


@dataclass
class TrainResult:
    model: UnifiedScoringModel
    best_epoch: int
    best_val_gini: float
    history: list[dict[str, float]] = field(default_factory=list)


def resolve_pos_weight(value: float | str | None, y: np.ndarray) -> float | None:
    if value is None:
        return None
    if value == "auto":
        return float((1 - y).sum() / max(y.sum(), 1))
    return float(value)


@torch.no_grad()
def predict_logits(model: nn.Module, ds: ScoringTorchDataset, device: torch.device,
                   batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = [model(batch_to(b, device)).cpu() for b in make_loader(ds, batch_size, shuffle=False)]
    return torch.cat(out).numpy().astype(np.float64)


def train_model(
    model: UnifiedScoringModel,
    train_ds: ScoringTorchDataset,
    val_ds: ScoringTorchDataset,
    cfg: TrainConfig,
    device: torch.device,
    seed: int,
    logger: logging.Logger,
) -> TrainResult:
    model.to(device)
    y_train = train_ds.y.numpy()
    y_val = val_ds.y.numpy()
    pw = resolve_pos_weight(cfg.pos_weight, y_train)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=None if pw is None else torch.tensor(pw, device=device)
    )
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=cfg.scheduler_patience
    )
    loader = make_loader(train_ds, cfg.batch_size, shuffle=True, seed=seed)
    logger.info("Обучение: %d train / %d val, pos_weight=%s, params=%d",
                len(train_ds), len(val_ds), pw,
                sum(p.numel() for p in model.parameters() if p.requires_grad))

    best_state = copy.deepcopy(model.state_dict())
    best_gini, best_epoch, bad_epochs = -np.inf, 0, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = batch_to(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch["y"])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch["y"])
            count += len(batch["y"])
        val_logits = predict_logits(model, val_ds, device)
        val_loss = float(criterion(torch.as_tensor(val_logits, device=device, dtype=torch.float32),
                                   torch.as_tensor(y_val, device=device)).item())
        val_gini = gini(y_val, val_logits)
        scheduler.step(val_gini)
        rec = {"epoch": epoch, "train_loss": total / count, "val_loss": val_loss,
               "val_gini": val_gini, "lr": optimizer.param_groups[0]["lr"], "sec": time.time() - t0}
        history.append(rec)
        logger.info("epoch %3d | train_loss %.4f | val_loss %.4f | val Gini %.2f%% | lr %.1e | %.1fs",
                    epoch, rec["train_loss"], val_loss, val_gini, rec["lr"], rec["sec"])
        if val_gini > best_gini + 1e-3:
            best_gini, best_epoch, bad_epochs = val_gini, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                logger.info("Early stopping: лучшая эпоха %d (val Gini %.2f%%)", best_epoch, best_gini)
                break
    model.load_state_dict(best_state)
    return TrainResult(model=model, best_epoch=best_epoch, best_val_gini=best_gini, history=history)


@dataclass
class PlattCalibrator:
    """PD = σ(a * logit + b). Нужен, если обучали с pos_weight (PD смещены вверх)."""

    a: float = 1.0
    b: float = 0.0

    @classmethod
    def fit(cls, logits: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logits.reshape(-1, 1), y)
        return cls(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * logits + self.b)))
