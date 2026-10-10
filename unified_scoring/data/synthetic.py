"""Синтетический генератор эмбеддингов источников с реалистичной структурой.

Идея генератора
---------------
1. У клиента есть скрытые факторы риска F (latent_dim штук). Истинный риск —
   нелинейная функция F, включающая ВЗАИМОДЕЙСТВИЯ факторов, которые видны
   разным источникам (например F1 видят карты и БКИ, F2 — расчётный счёт и БКИ).
   Именно такая структура делает объединение на уровне эмбеддингов выгоднее
   объединения скалярных скоров: скалярный скор источника «сжимает» факторы
   в одно число, и межисточниковые взаимодействия теряются.
2. Каждый источник «видит» своё подмножество факторов с шумом и выдаёт
   эмбеддинг заданной размерности, как предпоследний слой доменной модели:
   ReLU(линейная проекция [факторы + нерелевантные признаки]).
3. Наличие источника случайно, причём может зависеть от риска (MNAR):
   например, у рискованных клиентов чаще нет кредитной истории.
4. Таргет строится честно: моделируются помесячные траектории DPD,
   дефолт = DPD >= K в пределах N месяцев (основное K=90 и короткое K=30).
5. Во времени есть дрейф популяции, чтобы PSI был информативен.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from ..config import Config, SourceConfig
from .container import ScoringData
from .target import default_flag, simulate_dpd_paths

_RISK_WEIGHTS = np.array([0.7, 0.5, 0.5, 0.6, 0.45, 0.35, 0.35, 0.3, 0.25, 0.5])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def _latent_risk(F: np.ndarray, interaction: float) -> np.ndarray:
    k = F.shape[1]
    w = np.zeros(k)
    w[: min(k, len(_RISK_WEIGHTS))] = _RISK_WEIGHTS[: min(k, len(_RISK_WEIGHTS))]
    linear = F @ w
    cross = interaction * (F[:, 1] * F[:, 2] + 0.7 * np.tanh(F[:, 3]) * F[:, 6] + 0.5 * F[:, 0] * F[:, 9])
    nonlinear = 0.3 * np.maximum(F[:, 4], 0.0) ** 2
    risk = linear + cross + nonlinear
    return (risk - risk.mean()) / risk.std()


def _embedding_source(
    F: np.ndarray, src: SourceConfig, rng: np.random.Generator
) -> np.ndarray:
    p = src.synthetic
    factors = list(p.get("factors", [0, 1]))
    noise = float(p.get("noise", 0.5))
    n_nuisance = int(p.get("nuisance", 4))
    n = F.shape[0]
    observed = F[:, factors] + noise * rng.standard_normal((n, len(factors)))
    Z = np.hstack([observed, rng.standard_normal((n, n_nuisance))])
    A = rng.standard_normal((Z.shape[1], src.dim)) / np.sqrt(Z.shape[1])
    b = rng.normal(0.0, 0.5, src.dim)
    emb = np.maximum(Z @ A + b, 0.0) + 0.05 * rng.standard_normal((n, src.dim))
    return emb.astype(np.float32)


def _binned_category(x: np.ndarray, n_levels: int, prefix: str, rng: np.random.Generator) -> np.ndarray:
    """Категория, несущая информацию о факторе x, но с перемешанными метками
    (порядок уровней не совпадает с риском — нужен Entity Embedding, а не ordinal)."""
    edges = np.quantile(x, np.linspace(0, 1, n_levels + 1)[1:-1])
    level = np.digitize(x, edges)
    perm = rng.permutation(n_levels)
    return np.array([f"{prefix}_{i:02d}" for i in perm[level]], dtype=object)


def _tabular_source(F: np.ndarray, src: SourceConfig, rng: np.random.Generator) -> pd.DataFrame:
    p = src.synthetic
    noise = float(p.get("noise", 0.5))
    n = F.shape[0]
    eps = lambda: noise * rng.standard_normal(n)  # noqa: E731
    df = pd.DataFrame(
        {
            "region": _binned_category(F[:, 9] + eps(), 60, "REG", rng),
            "occupation": _binned_category(F[:, 4] + eps(), 15, "OCC", rng),
            "education": _binned_category(F[:, 0] + eps(), 5, "EDU", rng),
            "channel": _binned_category(0.5 * F[:, 5] + eps(), 6, "CH", rng),
            "product": rng.choice(["cash", "card", "pos", "auto"], size=n),
            "income": np.exp(10.8 - 0.35 * F[:, 0] + 0.3 * rng.standard_normal(n)),
            "age": np.clip(42 - 9 * F[:, 4] + 6 * rng.standard_normal(n), 18, 80).round(),
            "credit_amount": np.exp(11.5 + 0.6 * rng.standard_normal(n)),
            "debt_to_income": _sigmoid(0.8 * F[:, 9] + eps()),
            "months_at_job": np.clip(np.exp(3.5 - 0.5 * F[:, 0] + 0.6 * rng.standard_normal(n)), 0, 600).round(),
            "n_dependents": rng.poisson(0.8, n).astype(float),
        }
    )
    for col in ("income", "months_at_job", "occupation"):
        df.loc[rng.random(n) < 0.03, col] = np.nan  # пропуски в отдельных полях анкеты
    return df


def generate_synthetic(cfg: Config) -> ScoringData:
    sc = cfg.data.synthetic
    if sc.latent_dim < 10:
        raise ValueError("synthetic.latent_dim должен быть >= 10")
    rng = np.random.default_rng(cfg.experiment.seed)
    n = sc.n_clients

    period = rng.integers(0, sc.n_months, size=n)
    F = rng.standard_normal((n, sc.latent_dim))
    # дрейф популяции: со временем растёт доля клиентов с повышенным F0
    F[:, 0] += sc.drift * (period / max(sc.n_months - 1, 1) - 0.5)
    risk = _latent_risk(F, sc.interaction_strength)

    # случайные числа для траекторий DPD берём сразу из основного генератора,
    # а каждый источник получает СВОЙ генератор (seed, имя источника): добавление
    # нового источника в конфиг не меняет ни таргет, ни остальные источники
    tc = cfg.target
    n_sim = max(tc.horizon_months, 12)
    u_enter, u_cure = rng.random((n, n_sim)), rng.random((n, n_sim))
    macro = 0.15 * np.sin(2 * np.pi * period / 12)

    sources = cfg.enabled_sources
    embeddings: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    tabular: pd.DataFrame | None = None
    base_avail = {}
    for src in sources:
        src_rng = np.random.default_rng([cfg.experiment.seed, zlib.crc32(src.name.encode())])
        p = src.synthetic
        avail_p = float(p.get("availability", 0.7))
        base_avail[src.name] = avail_p
        logit_avail = _logit(avail_p) + float(p.get("risk_coef", 0.0)) * risk
        masks[src.name] = src_rng.random(n) < _sigmoid(logit_avail)
        if src.type == "embedding":
            embeddings[src.name] = _embedding_source(F, src, src_rng)
        else:
            tabular = _tabular_source(F, src, src_rng)

    # гарантируем хотя бы один источник: включаем самый распространённый
    none = ~np.any(np.stack(list(masks.values()), axis=1), axis=1)
    masks[max(base_avail, key=base_avail.get)][none] = True
    for name, emb in embeddings.items():
        emb[~masks[name]] = 0.0  # у отсутствующих источников эмбеддинга нет

    # таргет через траектории DPD; уровень дефолтов калибруется сдвигом
    def simulate(shift: float) -> np.ndarray:
        hazard = _sigmoid(shift + sc.risk_scale * risk + macro)
        return simulate_dpd_paths(hazard, n_sim, sc.cure_prob, u_enter, u_cure)

    lo, hi = -12.0, 2.0
    for _ in range(40):  # бисекция по сдвигу под целевой default_rate
        mid = (lo + hi) / 2
        rate = default_flag(simulate(mid), tc.horizon_months, tc.dpd_threshold).mean()
        lo, hi = (mid, hi) if rate < sc.default_rate else (lo, mid)
    paths = simulate((lo + hi) / 2)

    targets = {
        "main": default_flag(paths, tc.horizon_months, tc.dpd_threshold),
        "short": default_flag(paths, tc.horizon_months, tc.short_dpd_threshold),
    }
    return ScoringData(
        client_id=np.arange(n),
        period=period,
        targets=targets,
        embeddings=embeddings,
        masks=masks,
        tabular=tabular,
    )
