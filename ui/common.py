"""Общее для всех страниц: где лежат результаты, понятные названия моделей и данных, графики."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from unified_scoring.report import BASELINE, MAIN_MODEL

ROOT = Path(__file__).resolve().parent.parent
STANDARD_MARK = "стандартный подход"  # так в задаче банка подписан его стандартный подход
DEFAULT_APPROVAL = "0.7"

# Названия моделей простыми словами (техническое название остаётся в подсказках и таблицах)
MODEL_PLAIN = {
    MAIN_MODEL: "Единая модель проекта",
    BASELINE: "Обычный подход: смешать готовые оценки",
    "LR на эмбеддингах": "Простая модель на «сжатых» данных",
    "Бустинг на векторе скоров": "Бустинг на готовых оценках",
    "Бустинг на эмбеддингах": "Бустинг на «сжатых» данных",
    "Бустинг на агрегатах (стандартный подход)": "Стандарт банков: бустинг на сводных показателях",
    "GRU на сырой кредитной истории": "Нейросеть читает историю (1 сеть)",
    "GRU по фолдам (ансамбль)": "Нейросеть читает историю (5 сетей)",
}

SOURCE_TITLES = {
    "card": "Карты", "account": "Расчётный счёт", "bki": "Кредитная история",
    "graph": "Связи с другими клиентами", "tabular": "Анкета", "insurance": "Страховые полисы",
    "prev_app": "Прошлые заявки", "pos": "Покупки в рассрочку",
    "bki_seq": "История кредитов целиком (нейросеть)", "bki_agg": "Сводные показатели истории",
    "bki_gru_cv": "Прогноз 5 нейросетей",
}


@dataclass(frozen=True)
class Dataset:
    folder: str
    title: str
    about: str
    note: str = ""


REFERENCE = [
    Dataset(
        "alfa_bki_1m", "Задача банка · 1 млн клиентов",
        "Реальные обезличенные кредитные истории **1 000 000** клиентов Альфа-Банка. "
        "Модели учились на более ранних заявках, а проверялись на **самых поздних** — как в жизни, "
        "когда модель оценивает новых клиентов.",
        "Единая модель здесь не лучшая. Сводные показатели считаются из той же самой истории, поэтому "
        "нового они не добавляют. Объединение источников полезно, когда данные действительно разные — "
        "это видно на учебных данных.",
    ),
    Dataset(
        "alfa_bki_reference", "Задача банка · 500 тыс. клиентов",
        "Тот же эксперимент на **500 000** самых поздних клиентов — проверка, что вывод не зависит от объёма.",
        "Выводы те же, что на 1 млн клиентов.",
    ),
    Dataset(
        "synthetic_reference", "Учебные данные · 5 источников",
        "**50 000** искусственных клиентов, у каждого до пяти видов данных: карты, расчётный счёт, "
        "кредитная история, связи с другими клиентами и анкета. Здесь видно, зачем объединять разные источники.",
    ),
    Dataset(
        "home_credit_full", "Открытые данные Home Credit",
        "**307 511** реальных заявок из открытого соревнования Kaggle: анкета, кредитная история, "
        "прошлые заявки, карты и рассрочки.",
        "«Сжатые» данные (эмбеддинги) лучше готовых оценок, но лучший результат даёт бустинг на них, "
        "а не единая нейросеть.",
    ),
]


def plain(model: str) -> str:
    return MODEL_PLAIN.get(model, model)


# ---------------------------------------------------------------------- чтение запусков
def list_runs() -> list[Path]:
    """Все папки с обученной моделью: эталонные (results/) и свои (runs/), новые сверху."""
    dirs = [p.parent for base in ("results", "runs") for p in (ROOT / base).glob("*/metrics.json")]
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def run_label(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


@st.cache_data
def _read_json(path: str, mtime: float) -> dict:  # mtime в ключе кэша: перечитываем изменённый файл
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_summary(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "metrics.json"
    return _read_json(str(path), path.stat().st_mtime)


def available_reference() -> list[Dataset]:
    return [d for d in REFERENCE if (ROOT / "results" / d.folder / "metrics.json").exists()]


# ---------------------------------------------------------------------- выводы из metrics.json
def test_metric(summary: dict, metric: str) -> pd.Series:
    return pd.Series({m: v["test"]["main"][metric] for m, v in summary["metrics"].items()})


def baseline_of(summary: dict) -> str:
    """С чем сравниваемся: в задаче банка — его стандартный подход, иначе — смешивание готовых оценок."""
    return next((m for m in summary["metrics"] if STANDARD_MARK in m), BASELINE)


def comparisons(summary: dict) -> dict[tuple[str, str], dict]:
    """Все пары моделей, для которых посчитан бутстрэп-интервал разницы Джини: (лучше?, чем) -> статистика."""
    pairs = {(MAIN_MODEL, other): v for other, v in summary.get("significance", {}).items()}
    for key, v in summary.get("embeddings_vs_scores", {}).items():
        match = re.fullmatch(r"(.*) против «(.*)»", key)
        if match:
            pairs[(match.group(1), match.group(2))] = v
    return pairs


def headline(summary: dict) -> tuple[str, str, dict | None]:
    """Главное сравнение для страницы: самый сильный подход против базового и его интервал."""
    base = baseline_of(summary)
    against_base = {a: v for (a, b), v in comparisons(summary).items() if b == base}
    if not against_base:
        return MAIN_MODEL, base, None
    best = max(against_base, key=lambda a: against_base[a]["delta"])
    return best, base, against_base[best]


def spaced(n: int) -> str:
    """1000000 -> «1 000 000» (неразрывные пробелы между разрядами)."""
    return f"{n:,}".replace(",", "\u00a0")


def out_of_100(auc: float) -> str:
    return f"{100 * auc:.0f} из 100"


# ---------------------------------------------------------------------- графики
ROLE_COLORS = {"лучший": "#2e9e5b", "стандарт": "#e8913a", "единая модель": "#4c8bf5", "другие": "#9aa0a6"}


def model_dot_chart(auc: pd.Series, best: str, base: str) -> None:
    """Точность моделей точками: шкала не от нуля, чтобы разница в 1–2 пункта была видна честно и без искажения длины."""
    def role(m: str) -> str:
        return ("лучший" if m == best else "стандарт" if m == base
                else "единая модель" if m == MAIN_MODEL else "другие")

    df = pd.DataFrame({"model": [plain(m) for m in auc.index], "tech": auc.index,
                       "score": 100 * auc.values, "role": [role(m) for m in auc.index]})
    df = df.sort_values("score", ascending=False)
    base_chart = alt.Chart(df).encode(
        y=alt.Y("model:N", sort=None, title=None, axis=alt.Axis(labelLimit=400)),
        x=alt.X("score:Q", title="Сколько раз из 100 модель верно отличает рискованного клиента",
                scale=alt.Scale(zero=False, padding=20), axis=alt.Axis(tickCount=6)),
        tooltip=[alt.Tooltip("model:N", title="Подход"), alt.Tooltip("tech:N", title="Техническое название"),
                 alt.Tooltip("score:Q", title="Из 100", format=".1f")],
    )
    color = alt.Color("role:N", title=None, legend=alt.Legend(orient="bottom"),
                      scale=alt.Scale(domain=list(ROLE_COLORS), range=list(ROLE_COLORS.values())))
    dots = base_chart.mark_circle(size=260, opacity=1).encode(color=color)
    text = base_chart.mark_text(align="left", dx=12, color="#888888").encode(text=alt.Text("score:Q", format=".1f"))
    st.altair_chart((dots + text).properties(height=46 * len(df) + 60), width="stretch")


def hbar(values: pd.Series, title: str, fmt: str = ".2f", height: int = 240) -> None:
    """Горизонтальная столбчатая диаграмма с полными подписями и значениями на столбцах."""
    df = values.rename("value").rename_axis("label").reset_index()
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=400)),
        x=alt.X("value:Q", title=title),
    )
    bars = base.mark_bar(color="#4c8bf5")
    text = base.mark_text(align="left", dx=4, color="#888888").encode(text=alt.Text("value:Q", format=fmt))
    st.altair_chart((bars + text).properties(height=height), width="stretch")
