"""Метрики качества скоринговой модели, реализованные на numpy «с нуля».

* ROC AUC — через статистику Манна–Уитни (средние ранги при совпадениях);
* Джини = 100 * (2 * AUC - 1), в %;
* KS = max |TPR - FPR| по всем порогам, в %;
* PR-AUC — average precision;
* Brier score — калибровка PD;
* PSI — стабильность распределения скора (бины по квантилям «ожидаемой» выборки);
* бутстрэп-интервал для разницы Джини двух моделей (значимость прироста).
"""
from __future__ import annotations

import numpy as np


def _check(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    s = np.asarray(score, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("y_true и score разной длины")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true должен быть бинарным")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Нужны оба класса для расчёта метрики")
    return y, s


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ранги 1..n, при совпадениях — средний ранг группы."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(xs) != 0) + 1, len(x)]
    starts, ends = boundaries[:-1], boundaries[1:]
    # средний ранг группы совпадающих значений: среднее чисел start+1..end
    ranks_sorted = np.repeat((starts + 1 + ends) / 2.0, ends - starts)
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y, s = _check(y_true, score)
    ranks = _average_ranks(s)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def gini(y_true: np.ndarray, score: np.ndarray) -> float:
    """Коэффициент Джини в процентах."""
    return 100.0 * (2.0 * roc_auc(y_true, score) - 1.0)


def _curve_points(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TP/FP-кривые по порогам, проходящим через различные значения скора (по убыванию)."""
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    last_in_group = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    return tp[last_in_group], fp[last_in_group], last_in_group + 1.0


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    """Колмогоров–Смирнов в процентах."""
    y, s = _check(y_true, score)
    tp, fp, _ = _curve_points(y, s)
    tpr = tp / y.sum()
    fpr = fp / (len(y) - y.sum())
    return float(100.0 * np.max(np.abs(tpr - fpr)))


def pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision: sum_k (R_k - R_{k-1}) * P_k."""
    y, s = _check(y_true, score)
    tp, _, n_pred = _curve_points(y, s)
    precision = tp / n_pred
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def brier_score(y_true: np.ndarray, pd_: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(pd_, dtype=np.float64).ravel()
    if p.min() < 0 or p.max() > 1:
        raise ValueError("Brier считается по вероятностям из [0, 1]")
    return float(np.mean((p - y) ** 2))


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index. <0.1 — стабильно, 0.1–0.25 — умеренный сдвиг, >0.25 — сильный."""
    e = np.asarray(expected, dtype=np.float64).ravel()
    a = np.asarray(actual, dtype=np.float64).ravel()
    edges = np.unique(np.quantile(e, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e_share = np.histogram(e, edges)[0] / len(e) + eps
    a_share = np.histogram(a, edges)[0] / len(a) + eps
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def bootstrap_gini_diff(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
    n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> dict[str, float]:
    """Парный бутстрэп разницы Джини (a − b) на одной выборке.

    Клиенты пересэмплируются с возвращением, обе модели оцениваются на одной
    и той же подвыборке. Возвращает Δ, (1 − alpha) доверительный интервал
    (перцентильный) и одностороннее p-value = доля бутстрэп-Δ ≤ 0.
    """
    y, a = _check(y_true, score_a)
    _, b = _check(y_true, score_b)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() in (0, n):
            continue
        deltas.append(gini(yb, a[idx]) - gini(yb, b[idx]))
    d = np.asarray(deltas)
    return {
        "delta": gini(y, a) - gini(y, b),
        "ci_low": float(np.quantile(d, alpha / 2)),
        "ci_high": float(np.quantile(d, 1 - alpha / 2)),
        "p_value": float(np.mean(d <= 0)),
        "n_boot": int(len(d)),
    }


def classification_report(y_true: np.ndarray, pd_: np.ndarray) -> dict[str, float]:
    """Полный набор метрик для вектора PD."""
    return {
        "gini": gini(y_true, pd_),
        "auc": roc_auc(y_true, pd_),
        "ks": ks_statistic(y_true, pd_),
        "pr_auc": pr_auc(y_true, pd_),
        "brier": brier_score(y_true, pd_),
        "default_rate": float(np.mean(y_true)),
        "mean_pd": float(np.mean(pd_)),
        "n": int(len(y_true)),
    }
