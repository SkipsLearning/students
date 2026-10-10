"""Построение таргета «дефолт = просрочка K+ дней в пределах N месяцев».

Функция ``default_flag`` применима к любым помесячным траекториям DPD
(days past due), в т.ч. к реальным платёжным историям. Для синтетики
траектории порождаются марковской цепочкой просрочки.
"""
from __future__ import annotations

import numpy as np


def default_flag(dpd_paths: np.ndarray, horizon_months: int, dpd_threshold: int) -> np.ndarray:
    """1, если в первые ``horizon_months`` месяцев DPD достигал ``dpd_threshold``.

    dpd_paths: (n_clients, n_months) — DPD на конец каждого месяца после выдачи.
    """
    if horizon_months > dpd_paths.shape[1]:
        raise ValueError("Горизонт N больше длины траекторий DPD")
    return (dpd_paths[:, :horizon_months] >= dpd_threshold).any(axis=1).astype(np.int8)


def simulate_dpd_paths(
    monthly_hazard: np.ndarray,
    n_months: int,
    cure_prob: float,
    u_enter: np.ndarray,
    u_cure: np.ndarray,
) -> np.ndarray:
    """Марковская модель просрочки с шагом 30 дней.

    * DPD=0: с вероятностью hazard клиент пропускает платёж -> DPD=30;
    * DPD>0: с вероятностью cure_prob гасит долг -> 0, иначе DPD += 30.
    Случайные числа передаются снаружи (common random numbers), чтобы
    калибровка уровня дефолтов была гладкой по параметру.
    """
    n = len(monthly_hazard)
    dpd = np.zeros(n, dtype=np.int32)
    paths = np.empty((n, n_months), dtype=np.int32)
    for m in range(n_months):
        delinquent = dpd > 0
        enter = ~delinquent & (u_enter[:, m] < monthly_hazard)
        cure = delinquent & (u_cure[:, m] < cure_prob)
        dpd = np.where(enter, 30, np.where(cure, 0, np.where(delinquent, dpd + 30, 0)))
        paths[:, m] = dpd
    return paths
