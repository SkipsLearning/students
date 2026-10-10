"""Скоринг клиентов обученной моделью.

    python scripts/predict.py --run-dir runs/<run> --input requests.jsonl --output scores.csv

Каждая строка входного JSONL — клиент с любым непустым набором источников
(формат описан в unified_scoring/inference.py и README).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_scoring.inference import Scorer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True, help="JSONL с запросами")
    parser.add_argument("--output", default=None, help="CSV с PD (по умолчанию — вывод в консоль)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    scorer = Scorer.from_run_dir(args.run_dir)
    pd_values = scorer.predict(records)
    out = pd.DataFrame({
        "client_id": [r.get("client_id", i) for i, r in enumerate(records)],
        "sources": [",".join(list(r.get("embeddings") or {}) + (["tabular"] if r.get("tabular") else []))
                    for r in records],
        "pd": pd_values,
    })
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Сохранено {len(out)} скоров в {args.output}")
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
