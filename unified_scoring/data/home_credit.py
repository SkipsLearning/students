"""Адаптер датасета Kaggle «Home Credit Default Risk».

Соответствие таблиц Home Credit источникам единой модели:

=====================  ==========================  ================================
источник (config)      таблица(ы)                  смысл
=====================  ==========================  ================================
tabular                application_train           анкета заявки (Entity Embeddings)
bki                    bureau + bureau_balance     кредитная история в бюро
card                   credit_card_balance         помесячная история карт
account                installments_payments       история платежей (аналог р/с)
pos                    POS_CASH_balance            помесячная история POS/cash
prev_app               previous_application        прошлые заявки в Home Credit
graph                  —                           в датасете нет, источник выключен
=====================  ==========================  ================================

Двухэтапная схема (как в постановке):
1. Для каждого источника-последовательности обучается доменная GRU-модель
   на части ``domain_fit``; эмбеддинг снимается с её предпоследнего слоя.
2. Единая модель обучается на train по этим эмбеддингам.
Так эмбеддинги на train/val/test — out-of-sample для доменных моделей.

Ограничения датасета (важно для интерпретации результатов):
* дат выдачи нет -> честный out-of-time невозможен, используется
  стратифицированное случайное разбиение (``split.scheme: random``);
* таргет TARGET задан организаторами («проблемы с оплатой») и не равен
  определению 90+ за 12 месяцев; короткое определение K=30 построить нельзя.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..config import Config
from .container import ScoringData, random_splits
from .sequence_encoder import encode, fit_sequence_encoder

ANOMALY_DAYS = 365243  # «заглушка» вместо даты в Home Credit
Frame = tuple[pd.DataFrame, str, list[str]]  # (таблица, колонка сортировки, признаки)


# --------------------------------------------------------------------------- helpers
def _read(path: Path, name: str, usecols: list[str], float_cols: list[str]) -> pd.DataFrame:
    file = path / f"{name}.csv"
    if not file.exists():
        raise FileNotFoundError(f"Нет файла {file}")
    return pd.read_csv(file, usecols=usecols, dtype={c: np.float32 for c in float_cols})


def _one_hot(df: pd.DataFrame, col: str, values: list[str]) -> list[str]:
    names = []
    for v in values:
        name = f"{col}={v}"
        df[name] = (df[col] == v).astype(np.float32)
        names.append(name)
    df[f"{col}=other"] = (~df[col].isin(values)).astype(np.float32)
    return names + [f"{col}=other"]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).astype(np.float32)


# --------------------------------------------------------------------------- tables
BUREAU_NUM = ["DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT",
              "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
              "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE", "DAYS_CREDIT_UPDATE", "AMT_ANNUITY"]


def build_bureau(path: Path, ids: np.ndarray) -> Frame:
    b = _read(path, "bureau", ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_TYPE", *BUREAU_NUM],
              BUREAU_NUM)
    b = b[b.SK_ID_CURR.isin(ids)].copy()
    feats = list(BUREAU_NUM)
    if (path / "bureau_balance.csv").exists():
        bb = pd.read_csv(path / "bureau_balance.csv", dtype={"STATUS": "category"})
        bb = bb[bb.SK_ID_BUREAU.isin(b.SK_ID_BUREAU)]
        status = bb.STATUS.astype(str)
        st = pd.to_numeric(status, errors="coerce")  # 0..5 — бакеты просрочки; C/X -> NaN
        agg = (
            pd.DataFrame({"SK_ID_BUREAU": bb.SK_ID_BUREAU, "m": bb.MONTHS_BALANCE, "st": st,
                          "late": (st >= 1).astype(np.float32), "unk": (status == "X").astype(np.float32)})
            .groupby("SK_ID_BUREAU")
            .agg(bb_months=("m", "size"), bb_last_month=("m", "max"), bb_max_status=("st", "max"),
                 bb_late_share=("late", "mean"), bb_unknown_share=("unk", "mean"))
            .reset_index()
        )
        b = b.merge(agg, on="SK_ID_BUREAU", how="left")
        feats += ["bb_months", "bb_last_month", "bb_max_status", "bb_late_share", "bb_unknown_share"]
    feats += _one_hot(b, "CREDIT_ACTIVE", ["Active", "Closed", "Sold", "Bad debt"])
    feats += _one_hot(b, "CREDIT_TYPE", ["Consumer credit", "Credit card", "Car loan", "Mortgage", "Microloan"])
    b["debt_ratio"] = _safe_div(b.AMT_CREDIT_SUM_DEBT, b.AMT_CREDIT_SUM)
    return b, "DAYS_CREDIT", feats + ["debt_ratio"]


CARD_NUM = ["MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_CURRENT",
            "AMT_DRAWINGS_ATM_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT", "AMT_INST_MIN_REGULARITY",
            "CNT_DRAWINGS_CURRENT", "SK_DPD", "SK_DPD_DEF"]


def build_card(path: Path, ids: np.ndarray) -> Frame:
    c = _read(path, "credit_card_balance", ["SK_ID_CURR", *CARD_NUM], CARD_NUM)
    c = c[c.SK_ID_CURR.isin(ids)].copy()
    c["utilization"] = _safe_div(c.AMT_BALANCE, c.AMT_CREDIT_LIMIT_ACTUAL)
    c["pay_to_min"] = _safe_div(c.AMT_PAYMENT_TOTAL_CURRENT, c.AMT_INST_MIN_REGULARITY)
    return c, "MONTHS_BALANCE", CARD_NUM + ["utilization", "pay_to_min"]


INST_NUM = ["NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT", "AMT_PAYMENT"]


def build_installments(path: Path, ids: np.ndarray) -> Frame:
    p = _read(path, "installments_payments", ["SK_ID_CURR", *INST_NUM], INST_NUM)
    p = p[p.SK_ID_CURR.isin(ids)].copy()
    delay = p.DAYS_ENTRY_PAYMENT - p.DAYS_INSTALMENT
    p["dpd"] = delay.clip(lower=0)
    p["dbd"] = (-delay).clip(lower=0)
    p["pay_ratio"] = _safe_div(p.AMT_PAYMENT, p.AMT_INSTALMENT)
    p["underpaid"] = (p.AMT_INSTALMENT - p.AMT_PAYMENT.fillna(0)).astype(np.float32)
    p["no_payment"] = p.AMT_PAYMENT.isna().astype(np.float32)
    return p, "DAYS_INSTALMENT", INST_NUM + ["dpd", "dbd", "pay_ratio", "underpaid", "no_payment"]


POS_NUM = ["MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"]


def build_pos(path: Path, ids: np.ndarray) -> Frame:
    p = _read(path, "POS_CASH_balance", ["SK_ID_CURR", "NAME_CONTRACT_STATUS", *POS_NUM], POS_NUM)
    p = p[p.SK_ID_CURR.isin(ids)].copy()
    feats = POS_NUM + _one_hot(p, "NAME_CONTRACT_STATUS", ["Active", "Completed", "Signed"])
    p["remaining_share"] = _safe_div(p.CNT_INSTALMENT_FUTURE, p.CNT_INSTALMENT)
    return p, "MONTHS_BALANCE", feats + ["remaining_share"]


PREV_NUM = ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE",
            "DAYS_DECISION", "CNT_PAYMENT", "RATE_DOWN_PAYMENT", "DAYS_FIRST_DUE", "DAYS_LAST_DUE",
            "DAYS_TERMINATION", "HOUR_APPR_PROCESS_START"]


def build_previous(path: Path, ids: np.ndarray) -> Frame:
    cols = ["SK_ID_CURR", "NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE", "NAME_YIELD_GROUP", *PREV_NUM]
    p = _read(path, "previous_application", cols, PREV_NUM)
    p = p[p.SK_ID_CURR.isin(ids)].copy()
    for c in ("DAYS_FIRST_DUE", "DAYS_LAST_DUE", "DAYS_TERMINATION"):
        p.loc[p[c] == ANOMALY_DAYS, c] = np.nan
    feats = list(PREV_NUM)
    feats += _one_hot(p, "NAME_CONTRACT_STATUS", ["Approved", "Refused", "Canceled", "Unused offer"])
    feats += _one_hot(p, "NAME_CONTRACT_TYPE", ["Cash loans", "Consumer loans", "Revolving loans"])
    feats += _one_hot(p, "NAME_YIELD_GROUP", ["low_normal", "middle", "high", "low_action"])
    p["app_credit_ratio"] = _safe_div(p.AMT_APPLICATION, p.AMT_CREDIT)
    return p, "DAYS_DECISION", feats + ["app_credit_ratio"]


TABLE_BUILDERS: dict[str, Callable[[Path, np.ndarray], Frame]] = {
    "bureau": build_bureau,
    "credit_card_balance": build_card,
    "installments_payments": build_installments,
    "POS_CASH_balance": build_pos,
    "previous_application": build_previous,
}


# --------------------------------------------------------------------------- padding
def to_padded(
    frame: Frame, all_ids: np.ndarray, fit_ids: np.ndarray, max_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Таблица событий -> (строки клиентов с данными, X (n_present, L, F), длины).

    Берутся последние ``max_len`` событий по колонке сортировки, в хронологическом
    порядке. Признаки: signed log1p -> стандартизация по клиентам domain_fit -> NaN=0.
    """
    df, sort_col, feats = frame
    df = df.sort_values(["SK_ID_CURR", sort_col], kind="mergesort")
    values = df[feats].to_numpy(dtype=np.float32)
    values = np.sign(values) * np.log1p(np.abs(values))
    fit_mask = np.isin(df.SK_ID_CURR.to_numpy(), fit_ids)
    mu = np.nanmean(values[fit_mask], axis=0)
    sd = np.nanstd(values[fit_mask], axis=0)
    mu = np.nan_to_num(mu)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    values = np.clip(np.nan_to_num((values - mu) / sd), -5, 5).astype(np.float32)

    rank_from_end = df.groupby("SK_ID_CURR").cumcount(ascending=False).to_numpy()
    keep = rank_from_end < max_len
    sk = df.SK_ID_CURR.to_numpy()[keep]
    values, rank_from_end = values[keep], rank_from_end[keep]
    present_ids, inv = np.unique(sk, return_inverse=True)
    lengths = np.bincount(inv).astype(np.int64)
    pos = lengths[inv] - 1 - rank_from_end
    X = np.zeros((len(present_ids), max_len, len(feats)), dtype=np.float32)
    X[inv, pos] = values
    rows = pd.Index(all_ids).get_indexer(present_ids)
    return rows, X, lengths


# --------------------------------------------------------------------------- main
def _load_application(path: Path, nrows: int | None) -> pd.DataFrame:
    file = path / "application_train.csv"
    if not file.exists():
        raise FileNotFoundError(
            f"Не найден {file}. Скачайте данные соревнования (нужно принять правила на Kaggle):\n"
            "  kaggle competitions download -c home-credit-default-risk\n"
            f"  unzip home-credit-default-risk.zip -d {path}"
        )
    app = pd.read_csv(file, nrows=nrows)
    app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].replace(ANOMALY_DAYS, np.nan)
    if "CODE_GENDER" in app:
        app.loc[app.CODE_GENDER == "XNA", "CODE_GENDER"] = np.nan
    ratios = pd.DataFrame({
        "CREDIT_TO_INCOME": _safe_div(app.AMT_CREDIT, app.AMT_INCOME_TOTAL),
        "ANNUITY_TO_INCOME": _safe_div(app.AMT_ANNUITY, app.AMT_INCOME_TOTAL),
        "CREDIT_TO_GOODS": _safe_div(app.AMT_CREDIT, app.AMT_GOODS_PRICE),
        "ANNUITY_TO_CREDIT": _safe_div(app.AMT_ANNUITY, app.AMT_CREDIT),
    })
    return pd.concat([app, ratios], axis=1).reset_index(drop=True)


def prepare_home_credit(
    cfg: Config, logger: logging.Logger
) -> tuple[ScoringData, dict[str, np.ndarray], dict[str, dict]]:
    hc = cfg.data.home_credit
    path = Path(hc.path)
    if cfg.split.scheme != "random":
        raise ValueError("В Home Credit нет дат выдачи: используйте split.scheme: random")
    app = _load_application(path, hc.nrows)
    ids = app.SK_ID_CURR.to_numpy()
    y = app.TARGET.to_numpy().astype(np.int8)
    n = len(app)
    logger.info("Home Credit: %d заявок, уровень TARGET=1: %.2f%%", n, 100 * y.mean())

    splits = random_splits(y, cfg.split.fractions, cfg.experiment.seed)
    fit_rows = splits["domain_fit"]
    if len(fit_rows) == 0:
        raise ValueError("Для обучения доменных энкодеров нужна доля split.fractions.domain_fit > 0")
    fit_ids = ids[fit_rows]
    fit_flag = np.zeros(n, dtype=bool)
    fit_flag[fit_rows] = True

    embeddings: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    encoders: dict[str, dict] = {}
    for src in cfg.enabled_sources:
        if src.type == "tabular":
            masks[src.name] = np.ones(n, dtype=bool)
            continue
        table = src.home_credit.get("table")
        if table not in TABLE_BUILDERS:
            raise ValueError(
                f"Источник {src.name}: для Home Credit нужна home_credit.table из {sorted(TABLE_BUILDERS)}"
            )
        logger.info("Источник %s: таблица %s", src.name, table)
        frame = TABLE_BUILDERS[table](path, ids)
        rows, X, lengths = to_padded(frame, ids, fit_ids, int(src.home_credit.get("max_len", 32)))
        del frame
        local_fit = np.flatnonzero(fit_flag[rows])
        enc = fit_sequence_encoder(
            X, lengths, y[rows], local_fit, emb_dim=src.dim, hidden=hc.encoder_hidden,
            epochs=hc.encoder_epochs, lr=hc.encoder_lr, batch_size=hc.encoder_batch_size,
            seed=cfg.experiment.seed, logger=logger, name=src.name,
        )
        emb_present, _ = encode(enc, X, lengths)
        emb = np.zeros((n, src.dim), dtype=np.float32)
        emb[rows] = emb_present
        mask = np.zeros(n, dtype=bool)
        mask[rows] = True
        embeddings[src.name], masks[src.name] = emb, mask
        encoders[src.name] = {k: v.cpu() for k, v in enc.state_dict().items()}
        logger.info("  %s: покрытие %.1f%% клиентов, признаков %d", src.name, 100 * mask.mean(), X.shape[2])
        del X

    data = ScoringData(
        client_id=ids,
        period=np.full(n, -1),
        targets={"main": y},
        embeddings=embeddings,
        masks=masks,
        tabular=app.drop(columns=["SK_ID_CURR", "TARGET"]),
    )
    return data, splits, encoders
