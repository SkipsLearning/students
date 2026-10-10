"""Адаптер данных соревнования Альфа-Банка «DL Fintech: кредитный скоринг по данным БКИ».

Постановка банка: стандартный подход — модели на агрегатах кредитной истории
(логистическая регрессия, бустинг); альтернативный — рекуррентная нейросеть
на последовательности кредитов «как есть». Адаптер готовит оба представления
одной и той же истории как два источника единой модели:

=========  =================================================================
источник   содержимое
=========  =================================================================
bki_seq    последовательность кредитов → доменная GRU (этап 1) → эмбеддинг
bki_agg    агрегаты той же истории: доли значений каждого признака (как у организаторов)
=========  =================================================================

Дополнительно в сравнение попадают прогнозы GRU как самостоятельной модели —
«нейросеть на сырых последовательностях» из постановки банка: одна доменная GRU
и ансамбль GRU, обученных по фолдам (как в решении 7-го места соревнования).

Время. Номер заявки растёт со временем подачи (признаки плавно дрейфуют с ростом
id, а test соревнования — самые поздние заявки), поэтому id служит часами:
``split.scheme: time`` делит клиентов на 24 «периода» по порядку id, и проверка
идёт на самых поздних — как в постановке банка (test — следующие месяцы).

Ограничение: у test_data соревнования нет ответов, поэтому качество меряется на
отложенной (самой поздней) части train_target.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from ..config import AlfaBkiConfig, Config
from .container import ScoringData, make_splits
from .sequence_encoder import encode, fit_sequence_encoder

GRU_MODEL = "GRU на сырой кредитной истории"
GRU_CV_MODEL = "GRU по фолдам (ансамбль)"
ID_PERIODS = 24  # число «периодов» по порядку id для разбиения по времени


def _chunk_files(path: Path) -> list[Path]:
    folder = path / "train_data"
    files = sorted(folder.glob("train_data_*.pq"), key=lambda f: int(re.findall(r"\d+", f.stem)[-1]))
    if not files:
        raise FileNotFoundError(
            f"Не найдены файлы {folder}/train_data_*.pq. Скачайте data_for_competition "
            "с https://ods.ai/competitions/dl-fintech-bki и распакуйте в data/data_for_competition"
        )
    return files


def load_histories(path: Path, n_clients: int | None, logger: logging.Logger,
                   take: str = "latest") -> pd.DataFrame:
    """Читает части train_data, пока не наберётся n_clients клиентов.

    take="latest" — самые поздние заявки (наибольшие id; ближе всего к test
    соревнования), "earliest" — самые ранние. Значения признаков — небольшие
    целые коды, поэтому храним их в int8 (в 8 раз меньше памяти, чем int64).
    """
    if take not in {"latest", "earliest"}:
        raise ValueError("alfa_bki.take: latest | earliest")
    files = _chunk_files(path)
    frames, seen = [], 0
    for f in (reversed(files) if take == "latest" else files):
        df = pd.read_parquet(f)
        feats = df.columns.drop(["id", "rn"])
        df = df.astype({"id": np.int32, "rn": np.int16, **{c: np.int8 for c in feats}})
        frames.append(df)
        seen += df["id"].nunique()
        logger.info("  прочитан %s: всего клиентов %d", f.name, seen)
        if n_clients is not None and seen >= n_clients:
            break
    hist = pd.concat(frames, ignore_index=True)
    if n_clients is not None:
        uniq = np.sort(hist["id"].unique())
        keep = uniq[-n_clients:] if take == "latest" else uniq[:n_clients]
        hist = hist[hist["id"].isin(keep)]
    return hist.sort_values(["id", "rn"], kind="mergesort").reset_index(drop=True)


def feature_columns(hist: pd.DataFrame) -> list[str]:
    return [c for c in hist.columns if c not in ("id", "rn")]


def build_aggregates(hist: pd.DataFrame, block_clients: int = 250_000) -> pd.DataFrame:
    """Классические агрегаты кредитной истории клиента (индекс — id), как в базовом
    решении организаторов: число кредитов и доля кредитов с каждым значением
    каждого признака (one-hot, усреднённый по кредитам клиента).

    Считается блоками клиентов, чтобы one-hot 26 млн кредитов не занимал память целиком.
    """
    feats = feature_columns(hist)
    cards = {c: int(hist[c].max()) + 1 for c in feats}
    columns = ["n_credits"] + [f"{c}={v}" for c in feats for v in range(cards[c])]
    ids = hist["id"].to_numpy()
    uniq = np.unique(ids)
    blocks = []
    for k in range(0, len(uniq), block_clients):
        part = uniq[k:k + block_clients]
        lo, hi = np.searchsorted(ids, part[0], "left"), np.searchsorted(ids, part[-1], "right")
        inv = np.searchsorted(part, ids[lo:hi])
        n_credits = np.bincount(inv, minlength=len(part)).astype(np.float32)
        out = [n_credits[:, None]]
        for c in feats:
            K = cards[c]
            codes = hist[c].to_numpy()[lo:hi].astype(np.int64)
            counts = np.bincount(inv * K + codes, minlength=len(part) * K).reshape(len(part), K)
            out.append((counts / n_credits[:, None]).astype(np.float32))
        blocks.append(pd.DataFrame(np.hstack(out), index=part, columns=columns))
    return pd.concat(blocks)


def to_sequences(hist: pd.DataFrame, ids: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """История -> (строки клиентов с историей, X int8 (n, L, F), длины).

    Берутся последние ``max_len`` кредитов в порядке rn.
    """
    feats = feature_columns(hist)
    rank_from_end = hist.groupby("id").cumcount(ascending=False).to_numpy()
    keep = rank_from_end < max_len
    sk = hist["id"].to_numpy()[keep]
    values = hist[feats].to_numpy(dtype=np.int8)[keep]
    rank_from_end = rank_from_end[keep]
    present_ids, inv = np.unique(sk, return_inverse=True)
    lengths = np.bincount(inv).astype(np.int64)
    pos = lengths[inv] - 1 - rank_from_end
    X = np.zeros((len(present_ids), max_len, len(feats)), dtype=np.int8)
    X[inv, pos] = values
    rows = pd.Index(ids).get_indexer(present_ids)
    return rows, X, lengths


def id_periods(ids: np.ndarray, n_periods: int = ID_PERIODS) -> np.ndarray:
    """«Период» заявки по порядку id (id растёт со временем подачи): 0 — самые ранние."""
    rank = np.argsort(np.argsort(ids, kind="mergesort"), kind="mergesort")
    return (rank * n_periods // len(ids)).astype(np.int64)


def _fit_gru(X, lengths, y, fit_rows, ac: AlfaBkiConfig, dim: int, seed: int, logger, name: str, cards):
    return fit_sequence_encoder(
        X, lengths, y, fit_rows, emb_dim=dim, hidden=ac.encoder_hidden, epochs=ac.encoder_epochs,
        lr=ac.encoder_lr, batch_size=ac.encoder_batch_size, seed=seed, logger=logger, name=name,
        cardinalities=cards,
    )


def cross_validated_gru(X, lengths, y, pool: np.ndarray, ac: AlfaBkiConfig, dim: int, seed: int,
                        logger: logging.Logger, cards: list[int]) -> np.ndarray:
    """GRU как самостоятельная модель, обученная по фолдам (как в решении 7-го места).

    Клиенты ``pool`` делятся на cv_folds частей; модель k учится на остальных частях.
    Клиенты пула получают прогноз модели, которая их не видела (out-of-fold), остальные
    клиенты (val/test) — среднее прогнозов всех фолдовых моделей. Возвращает логиты.
    """
    logits = np.zeros(len(X), dtype=np.float64)
    rest = np.setdiff1d(np.arange(len(X)), pool)
    folds = StratifiedKFold(ac.cv_folds, shuffle=True, random_state=seed)
    for k, (fit_idx, oof_idx) in enumerate(folds.split(pool, y[pool]), start=1):
        model = _fit_gru(X, lengths, y, pool[fit_idx], ac, dim, seed + k, logger,
                         f"GRU фолд {k}/{ac.cv_folds}", cards)
        oof = pool[oof_idx]
        logits[oof] = encode(model, X[oof], lengths[oof])[1]
        logits[rest] += encode(model, X[rest], lengths[rest])[1] / ac.cv_folds
    return logits


def _scores_for_all(n: int, rows: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Логиты клиентов с историей -> массив на всех клиентов (без истории — средний логит)."""
    out = np.full(n, float(np.mean(present)))
    out[rows] = present
    return out


def _gru_sources(hist, ids, y, fit_flag, pool_flag, cfg: Config, logger) -> dict:
    """Всё, что строится из последовательностей: эмбеддинги доменной GRU, её прогноз
    и out-of-fold прогноз ансамбля GRU по фолдам."""
    ac = cfg.data.alfa_bki
    n = len(ids)
    seq_src = next((s for s in cfg.enabled_sources if s.type == "embedding" and s.name != ac.cv_score_source), None)
    rows, X, lengths = to_sequences(hist, ids, ac.max_len)
    cards = (X.max(axis=(0, 1)).astype(np.int64) + 1).tolist()
    y_rows = y[rows]
    out: dict = {"rows": rows, "extra": {}, "encoders": {}}
    if seq_src is not None:
        logger.info("Источник %s: последовательности кредитов → доменная GRU", seq_src.name)
        enc = _fit_gru(X, lengths, y_rows, np.flatnonzero(fit_flag[rows]), ac, seq_src.dim,
                       cfg.experiment.seed, logger, seq_src.name, cards)
        emb_present, logits_present = encode(enc, X, lengths)
        emb = np.zeros((n, seq_src.dim), dtype=np.float32)
        emb[rows] = emb_present
        out["seq"] = (seq_src.name, emb)
        out["extra"][GRU_MODEL] = _scores_for_all(n, rows, logits_present)
        out["encoders"][seq_src.name] = {"state_dict": {k: v.cpu() for k, v in enc.state_dict().items()},
                                         "cardinalities": cards}
    if ac.cv_folds > 1:
        logger.info("GRU по фолдам: %d фолдов на domain_fit + train", ac.cv_folds)
        cv = cross_validated_gru(X, lengths, y_rows, np.flatnonzero(pool_flag[rows]), ac,
                                 seq_src.dim if seq_src else 64, cfg.experiment.seed, logger, cards)
        out["extra"][GRU_CV_MODEL] = _scores_for_all(n, rows, cv)
    return out


def prepare_alfa_bki(
    cfg: Config, logger: logging.Logger
) -> tuple[ScoringData, dict[str, np.ndarray], dict[str, dict]]:
    ac = cfg.data.alfa_bki
    path = Path(ac.path)
    hist = load_histories(path, ac.n_clients, logger, ac.take)
    target = pd.read_csv(path / "train_target.csv").set_index("id")["flag"]
    ids = np.sort(hist["id"].unique())
    y = target.reindex(ids).to_numpy()
    if np.isnan(y).any():
        raise ValueError("Для части клиентов нет flag в train_target.csv")
    y = y.astype(np.int8)
    n = len(ids)
    logger.info("Альфа-Банк БКИ: %d клиентов (id %d–%d), %d кредитов, доля дефолтов %.2f%%",
                n, ids[0], ids[-1], len(hist), 100 * y.mean())

    base = ScoringData(client_id=ids, period=id_periods(ids), targets={"main": y})
    splits = make_splits(base, cfg)
    if len(splits["domain_fit"]) == 0:
        raise ValueError("Для обучения доменной GRU нужна непустая часть domain_fit")
    fit_flag = np.zeros(n, dtype=bool)
    fit_flag[splits["domain_fit"]] = True
    pool_flag = fit_flag.copy()
    pool_flag[splits["train"]] = True  # domain_fit + train: данные для GRU по фолдам

    gru = _gru_sources(hist, ids, y, fit_flag, pool_flag, cfg, logger)
    has_history = np.zeros(n, dtype=bool)
    has_history[gru["rows"]] = True
    embeddings: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    tabular = None
    for src in cfg.enabled_sources:
        if src.type == "tabular":
            logger.info("Источник %s: агрегаты кредитной истории", src.name)
            tabular = build_aggregates(hist).reindex(ids).reset_index(drop=True)
            masks[src.name] = tabular.notna().any(axis=1).to_numpy()
            logger.info("  %s: признаков %d", src.name, tabular.shape[1])
        elif src.name == ac.cv_score_source:
            # стекинг: out-of-fold прогноз ансамбля GRU как одномерный «эмбеддинг»
            if GRU_CV_MODEL not in gru["extra"]:
                raise ValueError(f"Источник {src.name} требует data.alfa_bki.cv_folds > 1")
            if src.dim != 1:
                raise ValueError(f"Источник {src.name}: dim должен быть 1 (это один прогноз)")
            embeddings[src.name] = gru["extra"][GRU_CV_MODEL][:, None].astype(np.float32)
            masks[src.name] = has_history
        else:
            name, emb = gru["seq"]
            embeddings[name], masks[name] = emb, has_history
    logger.info("Клиентов с историей: %.1f%%", 100 * has_history.mean())

    data = ScoringData(client_id=ids, period=base.period, targets={"main": y}, embeddings=embeddings,
                       masks=masks, tabular=tabular, extra_scores=gru["extra"])
    return data, splits, gru["encoders"]
