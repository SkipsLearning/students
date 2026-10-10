"""Загрузка данных: синтетика, Kaggle Home Credit или соревнование Альфа-Банка по БКИ."""
from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from .container import ScoringData, make_splits


def prepare_data(
    cfg: Config, logger: logging.Logger
) -> tuple[ScoringData, dict[str, np.ndarray], dict[str, dict]]:
    """Возвращает (данные, индексы разбиений, доп. артефакты — веса доменных энкодеров)."""
    if cfg.data.kind == "synthetic":
        from .synthetic import generate_synthetic

        data = generate_synthetic(cfg)
        return data, make_splits(data, cfg), {}
    if cfg.data.kind == "home_credit":
        from .home_credit import prepare_home_credit

        return prepare_home_credit(cfg, logger)
    if cfg.data.kind == "alfa_bki":
        from .alfa_bki import prepare_alfa_bki

        return prepare_alfa_bki(cfg, logger)
    raise ValueError(f"Неизвестный data.kind={cfg.data.kind}")


__all__ = ["ScoringData", "prepare_data"]
