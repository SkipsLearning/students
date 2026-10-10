"""Читаемый отчёт о качестве (Markdown + консоль)."""
from __future__ import annotations

from typing import Any


def _int(n: int) -> str:
    """Целое с пробелом-разделителем разрядов: 50000 -> '50 000'."""
    return f"{n:,}".replace(",", " ")

MAIN_MODEL = "Единая NN (эмбеддинги)"
BASELINE = "LR на векторе скоров"
TARGET_TITLES = {"main": "основной таргет", "short": "короткий таргет"}


def _metrics_table(metrics: dict[str, Any], split: str, target: str) -> list[str]:
    rows = ["| Модель | Джини, % | ROC AUC | KS, % | PR-AUC | Brier | ср. PD | DR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, by_split in metrics.items():
        m = by_split.get(split, {}).get(target)
        if m is None:
            continue
        rows.append(f"| {model} | {m['gini']:.2f} | {m['auc']:.4f} | {m['ks']:.2f} | {m['pr_auc']:.4f} "
                    f"| {m['brier']:.4f} | {m['mean_pd']:.4f} | {m['default_rate']:.4f} |")
    return rows


def format_console(summary: dict[str, Any]) -> str:
    lines = []
    for target in summary["targets"]:
        for split in ("val", "test"):
            lines.append(f"\n== {split.upper()} · {TARGET_TITLES.get(target, target)} ==")
            lines.append(f"{'Модель':32s} {'Джини%':>8s} {'AUC':>7s} {'KS%':>7s} {'PR-AUC':>7s} {'Brier':>7s}")
            for model, by_split in summary["metrics"].items():
                m = by_split[split][target]
                lines.append(f"{model:32s} {m['gini']:8.2f} {m['auc']:7.4f} {m['ks']:7.2f} "
                             f"{m['pr_auc']:7.4f} {m['brier']:7.4f}")
    return "\n".join(lines)


def build_markdown(summary: dict[str, Any]) -> str:
    md: list[str] = [f"# Отчёт: {summary['experiment']}", ""]
    d = summary["data"]
    md += [
        "## Данные", "",
        f"* Источник данных: **{d['kind']}**, клиентов: {_int(d['n'])}",
        f"* Определение основного таргета: {d['target_definition']}",
        f"* Разбиение: **{d['split_scheme']}**",
        "", "| Часть | Клиентов | Уровень дефолта |", "|---|---:|---:|",
    ]
    for name, s in d["splits"].items():
        md.append(f"| {name} | {_int(s['n'])} | {100 * s['default_rate']:.2f}% |")
    md += ["", "Доступность источников (вся выборка):", "", "| Источник | Покрытие |", "|---|---:|"]
    md += [f"| {k} | {100 * v:.1f}% |" for k, v in d["availability"].items()]

    t = summary["training"]
    md += ["", "## Обучение единой модели", "",
           f"Лучшая эпоха: {t['best_epoch']} из {t['epochs_run']}, Джини на val: {t['best_val_gini']:.2f}%. "
           f"Обучаемых параметров: {_int(t['n_params'])}."]

    for target in summary["targets"]:
        for split in ("val", "test"):
            md += ["", f"## Метрики: {split} · {TARGET_TITLES.get(target, target)}", ""]
            md += _metrics_table(summary["metrics"], split, target)

    md += ["", "## Прирост к базлайну (Джини, п.п.)", "",
           "| Таргет | Выборка | Единая NN | LR на векторе скоров | Δ |", "|---|---|---:|---:|---:|"]
    for target in summary["targets"]:
        for split in ("val", "test"):
            a = summary["metrics"][MAIN_MODEL][split][target]["gini"]
            b = summary["metrics"][BASELINE][split][target]["gini"]
            md.append(f"| {TARGET_TITLES.get(target, target)} | {split} | {a:.2f} | {b:.2f} | **{a - b:+.2f}** |")

    if "significance" in summary:
        md += ["", "## Значимость прироста (test, основной таргет, парный бутстрэп)", "",
               "Клиенты test пересэмплируются с возвращением, обе модели оцениваются на одной подвыборке. "
               "Если весь 95%-интервал выше нуля, прирост статистически значим.", "",
               "| Единая NN против | Δ Джини, п.п. | 95% интервал | p-value |", "|---|---:|---:|---:|"]
        for model, r in summary["significance"].items():
            md.append(f"| {model} | {r['delta']:+.2f} | [{r['ci_low']:+.2f}; {r['ci_high']:+.2f}] "
                      f"| {r['p_value']:.3f} |")

    if summary.get("embeddings_vs_scores"):
        md += ["", "## Ключевые сравнения подходов (test, бутстрэп)", "",
               "| Сравнение | Δ Джини, п.п. | 95% интервал | p-value |", "|---|---:|---:|---:|"]
        for name, r in summary["embeddings_vs_scores"].items():
            md.append(f"| {name} | {r['delta']:+.2f} | [{r['ci_low']:+.2f}; {r['ci_high']:+.2f}] "
                      f"| {r['p_value']:.3f} |")

    md += ["", "## Стабильность (PSI распределения PD)", "",
           "| Модель | train→val | train→test |", "|---|---:|---:|"]
    for model, p in summary["psi"].items():
        md.append(f"| {model} | {p['train_val']:.4f} | {p['train_test']:.4f} |")
    md.append("\nPSI < 0.1 — стабильно; 0.1–0.25 — умеренный сдвиг; > 0.25 — значимый сдвиг.")

    if "business" in summary:
        b = summary["business"]
        md += ["", "## Бизнес-срез (test): уровень дефолтов среди одобренных", "",
               "Одобряем долю клиентов с наименьшим PD. Снижение bad rate при том же объёме одобрений "
               "— прямой эффект роста Джини на потери (EL = PD × LGD × EAD).", "",
               "| Уровень одобрения | " + " | ".join(b) + " | Снижение bad rate к базлайну |",
               "|---|" + "---:|" * (len(b) + 1)]
        for ar in next(iter(b.values())):
            cells = " | ".join(f"{100 * b[m][ar]:.2f}%" for m in b)
            rel = 1 - b[MAIN_MODEL][ar] / b[BASELINE][ar] if b[BASELINE][ar] > 0 else float("nan")
            md.append(f"| {100 * float(ar):.0f}% | {cells} | {100 * rel:.1f}% |")

    md += ["", "## Джини на test по числу доступных источников", "",
           "| Число источников | Клиентов | " + " | ".join(summary["segments"].keys()) + " |",
           "|---|---:|" + "---:|" * len(summary["segments"])]
    groups = sorted({g for seg in summary["segments"].values() for g in seg}, key=int)
    for g in groups:
        first = next(iter(summary["segments"].values()))
        n = first.get(g, {}).get("n", 0)
        vals = [seg.get(g, {}).get("gini") for seg in summary["segments"].values()]
        cells = " | ".join("—" if v is None else f"{v:.2f}" for v in vals)
        md.append(f"| {g} | {n} | {cells} |")

    md += ["", "## Вклад источников (единая NN, test): Джини при «выключении» источника", "",
           "| Источник выключен | Джини, % | Δ к полной модели |", "|---|---:|---:|"]
    for src, a in summary["ablation"].items():
        md.append(f"| {src} | {a['gini']:.2f} | {a['delta']:+.2f} |")
    md.append("")
    return "\n".join(md)
