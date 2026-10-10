"""Запуск эксперимента: обучение, оценка, базлайны, отчёт.

    python scripts/run.py --config configs/synthetic.yaml
    python scripts/run.py --config configs/synthetic.yaml --set train.epochs=5 --set sources.graph.enabled=false
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_scoring.config import load_config  # noqa: E402
from unified_scoring.pipeline import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML-конфиг")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="переопределение параметра конфига (можно несколько)")
    parser.add_argument("--warm-start", default=None,
                        help="папка прошлого запуска: перенести веса совпадающих слоёв (towers старых источников)")
    args = parser.parse_args()
    run_experiment(load_config(args.config, args.set), warm_start=args.warm_start)


if __name__ == "__main__":
    main()
