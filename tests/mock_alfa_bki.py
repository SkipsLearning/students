"""Мини-копия схемы данных соревнования Альфа-Банка (те же файлы и колонки,
случайные коды) для smoke-теста адаптера без скачивания реальных данных."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ORDINAL_MAX = {
    "pre_since_opened": 19, "pre_since_confirmed": 17, "pre_pterm": 17, "pre_fterm": 16,
    "pre_till_pclose": 16, "pre_till_fclose": 15, "pre_loans_credit_limit": 19,
    "pre_loans_next_pay_summ": 6, "pre_loans_outstanding": 5, "pre_loans_total_overdue": 1,
    "pre_loans_max_overdue_sum": 3, "pre_loans_credit_cost_rate": 13, "pre_loans5": 16,
    "pre_loans530": 18, "pre_loans3060": 9, "pre_loans6090": 4, "pre_loans90": 19,
    "is_zero_loans5": 1, "is_zero_loans530": 1, "is_zero_loans3060": 1, "is_zero_loans6090": 1,
    "is_zero_loans90": 1, "pre_util": 19, "pre_over2limit": 19, "pre_maxover2limit": 19,
    "is_zero_util": 1, "is_zero_over2limit": 1, "is_zero_maxover2limit": 1,
    **{f"enc_paym_{i}": 3 for i in range(25)},
    "enc_loans_account_holder_type": 6, "enc_loans_credit_status": 6, "enc_loans_credit_type": 5,
    "enc_loans_account_cur": 3, "pclose_flag": 1, "fclose_flag": 1,
}


def make_mock_alfa_bki(path: Path, n: int = 3000, n_chunks: int = 2, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    (path / "train_data").mkdir(parents=True, exist_ok=True)
    n_credits = rng.integers(1, 15, n)
    ids = np.repeat(np.arange(n), n_credits)
    rn = np.concatenate([np.arange(1, k + 1) for k in n_credits])
    hist = pd.DataFrame({"id": ids, "rn": rn})
    for col, mx in ORDINAL_MAX.items():
        hist[col] = rng.integers(0, mx + 1, len(hist))
    # риск зависит от просрочек в истории, чтобы модели было что выучить
    overdue = hist.groupby("id")["pre_loans90"].mean().to_numpy()
    y = (rng.random(n) < 1 / (1 + np.exp(-(-4.0 + 0.2 * overdue)))).astype(int)
    pd.DataFrame({"id": np.arange(n), "flag": y}).to_csv(path / "train_target.csv", index=False)
    for i, part in enumerate(np.array_split(np.arange(n), n_chunks)):
        hist[hist["id"].isin(part)].to_parquet(path / "train_data" / f"train_data_{i}.pq", index=False)
    return path
