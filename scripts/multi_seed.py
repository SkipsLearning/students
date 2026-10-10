"""Повтор эксперимента на нескольких seed и сводка прироста Джини (среднее ± std).

    python scripts/multi_seed.py --config configs/synthetic.yaml --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_scoring.config import load_config  # noqa: E402
from unified_scoring.pipeline import run_experiment  # noqa: E402
from unified_scoring.report import BASELINE, MAIN_MODEL  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", default="multi_seed_summary.md")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        run = run_experiment(load_config(args.config, [*args.set, f"experiment.seed={seed}"]))
        summary = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        for target in summary["targets"]:
            for split in ("val", "test"):
                for model, by_split in summary["metrics"].items():
                    rows.append({"seed": seed, "target": target, "split": split, "model": model,
                                 "gini": by_split[split][target]["gini"]})
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["target", "split", "seed"], columns="model", values="gini")
    piv["delta"] = piv[MAIN_MODEL] - piv[BASELINE]
    agg = piv.groupby(level=["target", "split"]).agg(["mean", "std"])

    lines = [f"# Сводка по seed: {args.seeds}", "", "| Таргет | Выборка | " +
             " | ".join(f"{c}, Джини %" for c in piv.columns[:-1]) + " | Δ к базлайну, п.п. |",
             "|---|---|" + "---:|" * len(piv.columns)]
    for (target, split), r in agg.iterrows():
        cells = [f"{r[(c, 'mean')]:.2f} ± {r[(c, 'std')]:.2f}" for c in piv.columns]
        lines.append(f"| {target} | {split} | " + " | ".join(cells) + " |")
    lines += ["", "Δ = Единая NN − LR на векторе скоров. Значения: среднее ± std по seed.", "",
              f"Δ > 0 во всех запусках: {bool((piv['delta'] > 0).all())} "
              f"(min Δ = {piv['delta'].min():.2f} п.п.)"]
    text = "\n".join(lines)
    Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
