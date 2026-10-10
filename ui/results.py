"""Страница «Результаты»: что получилось, простыми словами, с переходом к подробным цифрам."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from unified_scoring.report import MAIN_MODEL

from .common import (DEFAULT_APPROVAL, ROOT, SOURCE_TITLES, Dataset, available_reference, headline,
                     hbar, model_dot_chart, out_of_100, plain, read_summary, spaced, test_metric)
from .expert import show_run

PER_LOANS = 10_000  # невозвраты считаем на 10 000 выданных кредитов — так проще представить


def render() -> None:
    st.title("Результаты")
    datasets = available_reference()
    if not datasets:
        st.info("Готовых результатов нет. Их можно получить в разделе «Для специалистов».")
        return
    by_title = {d.title: d for d in datasets}
    title = st.segmented_control("На каких данных", list(by_title), default=datasets[0].title)
    data = by_title[title or datasets[0].title]
    st.markdown(data.about)
    s = read_summary(ROOT / "results" / data.folder)

    best, base, stats = headline(s)
    _verdict(s, best, base, stats)
    st.divider()
    _accuracy(s, best, base)
    st.divider()
    _business(s, best, base)
    st.divider()
    _sources(s)
    if data.note:
        st.info(f"**Честная оговорка.** {data.note}", icon=":material/lightbulb:")
    with st.expander("Подробные цифры для специалистов: Джини, KS, интервалы, кривая обучения"):
        show_run(ROOT / "results" / data.folder)


def _verdict(s: dict, best: str, base: str, stats: dict | None) -> None:
    auc = test_metric(s, "auc")
    n_test = s["metrics"][MAIN_MODEL]["test"]["main"]["n"]
    st.subheader("Главный вывод")
    if stats and stats["ci_low"] > 0:
        st.success(
            f"**{plain(best)}** точнее, чем **{plain(base)}**, и это не случайность. "
            f"Проверено на {spaced(n_test)} клиентах, которых модели не видели при обучении.",
            icon=":material/check_circle:",
        )
    else:
        st.warning(f"Разница между **{plain(best)}** и **{plain(base)}** может быть случайной.",
                   icon=":material/help:")
    c1, c2, c3 = st.columns(3)
    c1.metric(plain(best), out_of_100(auc[best]), border=True,
              help=f"ROC AUC = {auc[best]:.4f}. Из пары «вернул / не вернул кредит» модель верно "
                   "указывает, кто рискованнее, примерно столько раз из 100.")
    c2.metric(plain(base), out_of_100(auc[base]), border=True,
              help=f"ROC AUC = {auc[base]:.4f}")
    if stats:
        c3.metric("Надёжность разницы", f"от {stats['ci_low']:+.1f} до {stats['ci_high']:+.1f}", border=True,
                  help=f"Сравнение повторили {stats['n_boot']} раз на случайных выборках клиентов. "
                       "В 95% повторов разница в пунктах Джини попала в этот интервал. "
                       "Если обе границы больше нуля — преимущество надёжное.")


def _accuracy(s: dict, best: str, base: str) -> None:
    st.subheader("Насколько точны разные подходы")
    st.caption("Берём пару клиентов: один потом вернул кредит, другой — нет. Сколько раз из 100 модель "
               "правильно скажет, кто из них рискованнее? 50 — угадывание наугад, 100 — идеально.")
    model_dot_chart(test_metric(s, "auc"), best, base)


def _business(s: dict, best: str, base: str) -> None:
    st.subheader("Что это даёт банку")
    rates = list(s["business"][best])
    rate = st.segmented_control(
        "Какую долю заявок банк одобряет", rates, default=DEFAULT_APPROVAL if DEFAULT_APPROVAL in rates else rates[0],
        format_func=lambda r: f"{float(r):.0%}",
    ) or rates[0]
    bad_base = PER_LOANS * s["business"][base][rate]
    bad_best = PER_LOANS * s["business"][best][rate]
    saved = bad_base - bad_best
    st.markdown(
        f"Банк одобряет **{float(rate):.0%}** заявок — тех, кого модель считает самыми надёжными. "
        f"Сколько из каждых **{spaced(PER_LOANS)}** выданных кредитов потом не вернут:"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric(plain(base), f"{bad_base:.0f}", border=True)
    c2.metric(plain(best), f"{bad_best:.0f}", border=True)
    c3.metric("Разница", f"на {abs(saved):.0f} {'меньше' if saved >= 0 else 'больше'}",
              f"{-saved / bad_base:+.1%}" if bad_base else None, delta_color="inverse", border=True,
              help="Зелёный — невозвратов стало меньше")


def _sources(s: dict) -> None:
    loss = pd.Series({SOURCE_TITLES.get(k, k): -v["delta"] for k, v in s["ablation"].items()})
    loss = loss.sort_values(ascending=False)
    st.subheader("Какие данные важнее всего для единой модели")
    st.caption("Убираем один вид данных и смотрим, насколько упадёт качество (в пунктах Джини). "
               "Чем длиннее столбец, тем сильнее модель опирается на эти данные. "
               "Значение около нуля или меньше — без этих данных модель почти ничего не теряет.")
    hbar(loss, "Потеря качества без этих данных, пунктов Джини", fmt=".1f", height=44 * len(loss) + 50)
