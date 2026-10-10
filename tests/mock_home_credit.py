"""Мини-копия схемы Home Credit (те же имена таблиц и колонок, случайные значения)
для smoke-теста адаптера без скачивания реальных данных."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_mock_home_credit(path: Path, n: int = 3000, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    ids = np.arange(100001, 100001 + n)
    risk = rng.standard_normal(n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.6 + 1.2 * risk)))).astype(int)
    app = pd.DataFrame({
        "SK_ID_CURR": ids, "TARGET": y,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
        "CODE_GENDER": rng.choice(["M", "F", "XNA"], n, p=[0.45, 0.549, 0.001]),
        "OCCUPATION_TYPE": rng.choice(["Laborers", "Sales staff", "Drivers", None], n),
        "ORGANIZATION_TYPE": rng.choice([f"Org{i}" for i in range(30)], n),
        "AMT_INCOME_TOTAL": np.exp(11.9 + 0.5 * rng.standard_normal(n)),
        "AMT_CREDIT": np.exp(13 + 0.6 * rng.standard_normal(n)),
        "AMT_ANNUITY": np.exp(10 + 0.4 * rng.standard_normal(n)),
        "AMT_GOODS_PRICE": np.exp(12.9 + 0.6 * rng.standard_normal(n)),
        "DAYS_BIRTH": -rng.integers(7000, 25000, n),
        "DAYS_EMPLOYED": np.where(rng.random(n) < 0.15, 365243, -rng.integers(0, 15000, n)),
        "EXT_SOURCE_2": 1 / (1 + np.exp(risk + 0.5 * rng.standard_normal(n))),
        "CNT_CHILDREN": rng.poisson(0.5, n),
    })
    app.to_csv(path / "application_train.csv", index=False)

    def events(cover: float, max_k: int) -> tuple[np.ndarray, np.ndarray]:
        who = ids[rng.random(n) < cover]
        k = rng.integers(1, max_k, len(who))
        return np.repeat(who, k), np.concatenate([np.arange(c) for c in k])

    cid, t = events(0.85, 12)
    m = len(cid)
    bureau = pd.DataFrame({
        "SK_ID_CURR": cid, "SK_ID_BUREAU": np.arange(5000000, 5000000 + m),
        "CREDIT_ACTIVE": rng.choice(["Active", "Closed", "Sold"], m),
        "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan"], m),
        "DAYS_CREDIT": -rng.integers(1, 3000, m), "CREDIT_DAY_OVERDUE": rng.integers(0, 3, m) * 10,
        "DAYS_CREDIT_ENDDATE": rng.integers(-2000, 2000, m), "DAYS_ENDDATE_FACT": -rng.integers(1, 2000, m),
        "AMT_CREDIT_MAX_OVERDUE": rng.exponential(1000, m), "CNT_CREDIT_PROLONG": rng.integers(0, 2, m),
        "AMT_CREDIT_SUM": rng.exponential(2e5, m), "AMT_CREDIT_SUM_DEBT": rng.exponential(1e5, m),
        "AMT_CREDIT_SUM_LIMIT": rng.exponential(1e4, m), "AMT_CREDIT_SUM_OVERDUE": rng.exponential(10, m),
        "DAYS_CREDIT_UPDATE": -rng.integers(1, 500, m), "AMT_ANNUITY": rng.exponential(1e4, m),
    })
    bureau.to_csv(path / "bureau.csv", index=False)
    bb_id = np.repeat(bureau.SK_ID_BUREAU.to_numpy(), 3)
    pd.DataFrame({"SK_ID_BUREAU": bb_id, "MONTHS_BALANCE": -np.tile([0, 1, 2], m),
                  "STATUS": rng.choice(["0", "1", "C", "X", "5"], len(bb_id))}).to_csv(
        path / "bureau_balance.csv", index=False)

    cid, t = events(0.3, 20)
    m = len(cid)
    pd.DataFrame({
        "SK_ID_CURR": cid, "MONTHS_BALANCE": -t, "AMT_BALANCE": rng.exponential(5e4, m),
        "AMT_CREDIT_LIMIT_ACTUAL": rng.choice([45000, 90000, 0], m), "AMT_DRAWINGS_CURRENT": rng.exponential(1e4, m),
        "AMT_DRAWINGS_ATM_CURRENT": rng.exponential(5e3, m), "AMT_PAYMENT_TOTAL_CURRENT": rng.exponential(5e3, m),
        "AMT_INST_MIN_REGULARITY": rng.exponential(2e3, m), "CNT_DRAWINGS_CURRENT": rng.integers(0, 10, m),
        "SK_DPD": rng.integers(0, 2, m) * 30, "SK_DPD_DEF": rng.integers(0, 2, m),
    }).to_csv(path / "credit_card_balance.csv", index=False)

    cid, t = events(0.9, 40)
    m = len(cid)
    inst_day = -t * 30
    pd.DataFrame({
        "SK_ID_CURR": cid, "NUM_INSTALMENT_VERSION": rng.integers(0, 3, m), "NUM_INSTALMENT_NUMBER": t + 1,
        "DAYS_INSTALMENT": inst_day, "DAYS_ENTRY_PAYMENT": inst_day + rng.integers(-10, 15, m),
        "AMT_INSTALMENT": rng.exponential(1e4, m),
        "AMT_PAYMENT": np.where(rng.random(m) < 0.02, np.nan, rng.exponential(1e4, m)),
    }).to_csv(path / "installments_payments.csv", index=False)

    cid, t = events(0.9, 8)
    m = len(cid)
    pd.DataFrame({
        "SK_ID_CURR": cid, "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled"], m),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Consumer loans"], m),
        "NAME_YIELD_GROUP": rng.choice(["middle", "high", "XNA"], m),
        "AMT_ANNUITY": rng.exponential(1e4, m), "AMT_APPLICATION": rng.exponential(1e5, m),
        "AMT_CREDIT": rng.exponential(1e5, m), "AMT_DOWN_PAYMENT": rng.exponential(1e3, m),
        "AMT_GOODS_PRICE": rng.exponential(1e5, m), "DAYS_DECISION": -rng.integers(1, 2500, m),
        "CNT_PAYMENT": rng.integers(6, 36, m), "RATE_DOWN_PAYMENT": rng.random(m),
        "DAYS_FIRST_DUE": np.where(rng.random(m) < 0.3, 365243, -rng.integers(1, 2000, m)),
        "DAYS_LAST_DUE": -rng.integers(1, 2000, m), "DAYS_TERMINATION": -rng.integers(1, 2000, m),
        "HOUR_APPR_PROCESS_START": rng.integers(0, 24, m),
    }).to_csv(path / "previous_application.csv", index=False)

    cid, t = events(0.6, 15)
    m = len(cid)
    pd.DataFrame({
        "SK_ID_CURR": cid, "MONTHS_BALANCE": -t, "CNT_INSTALMENT": rng.integers(6, 24, m),
        "CNT_INSTALMENT_FUTURE": rng.integers(0, 12, m), "SK_DPD": rng.integers(0, 2, m),
        "SK_DPD_DEF": rng.integers(0, 2, m),
        "NAME_CONTRACT_STATUS": rng.choice(["Active", "Completed", "Signed"], m),
    }).to_csv(path / "POS_CASH_balance.csv", index=False)
    return path
