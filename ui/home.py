"""Главная: о чём проект, как он работает и что получилось — без формул."""
from __future__ import annotations

import streamlit as st

from .common import DEFAULT_APPROVAL, ROOT, headline, out_of_100, plain, read_summary, test_metric

BANK_RUN = ROOT / "results/alfa_bki_1m"
PER_LOANS = 10_000

SOURCES = ["Карты", "Расчётный счёт", "Кредитная история", "Связи с клиентами", "Анкета"]
FLOW_CSS = """
<style>
.flow {display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin:8px 0 4px;}
.flow .col {display:flex; flex-direction:column; gap:6px;}
.flow .box {border:1px solid rgba(128,128,128,.35); border-radius:10px; padding:7px 14px; text-align:center;}
.flow .src {background:rgba(76,139,245,.14);}
.flow .mix {background:rgba(46,158,91,.18); font-weight:600; padding:16px 18px;}
.flow .out {background:rgba(232,145,58,.18);}
.flow .arrow {font-size:1.6rem; opacity:.55;}
</style>
"""


def render() -> None:
    st.title("Как банку точнее понять, вернёт ли клиент кредит")
    st.markdown(
        "Когда человек просит кредит, банк оценивает **вероятность, что деньги не вернут**. "
        "От этой оценки зависит, одобрят ли заявку. Ошибка дорогая в обе стороны: "
        "выдать ненадёжному — потерять деньги, отказать надёжному — потерять клиента."
    )
    _steps()
    st.divider()
    _bank_findings()
    st.divider()
    _idea()
    st.divider()
    _where_next()


def _steps() -> None:
    cols = st.columns(3)
    steps = [
        (":material/database:", "1. Данные", "Что банк знает о клиенте: кредитная история, карты, счёт, анкета."),
        (":material/neurology:", "2. Модель", "Учится на прошлых клиентах, чьё поведение уже известно, "
                                              "и оценивает риск новых."),
        (":material/gavel:", "3. Решение", "Чем ниже риск, тем выше шанс одобрения. "
                                           "Точнее модель — меньше потерь у банка."),
    ]
    for col, (icon, title, text) in zip(cols, steps):
        with col.container(border=True):
            st.markdown(f"#### {icon} {title}")
            st.markdown(text)


def _bank_findings() -> None:
    st.header("Задача банка и ответ на неё")
    st.markdown(
        "Альфа-Банк дал задачу: оценить клиентов **только по кредитной истории** и сравнить два подхода.\n"
        "- **Стандартный** — посчитать сводные показатели (сколько кредитов, сколько просрочек…) "
        "и обучить на них бустинг.\n"
        "- **Альтернативный** — дать нейросети прочитать историю кредитов **целиком, по порядку**."
    )
    if not (BANK_RUN / "metrics.json").exists():
        st.info("Результатов по задаче банка пока нет.")
        return
    s = read_summary(BANK_RUN)
    best, base, stats = headline(s)
    auc = test_metric(s, "auc")
    bad_base = PER_LOANS * s["business"][base][DEFAULT_APPROVAL]
    bad_best = PER_LOANS * s["business"][best][DEFAULT_APPROVAL]
    c1, c2, c3 = st.columns(3)
    c1.metric("Стандартный подход", out_of_100(auc[base]), border=True,
              help="Сколько раз из 100 модель верно отличает рискованного клиента от надёжного (ROC AUC).")
    c2.metric("Нейросеть, читающая историю", out_of_100(auc[best]),
              f"{100 * (auc[best] - auc[base]):+.1f}", border=True,
              help=f"{plain(best)}. ROC AUC = {auc[best]:.4f} против {auc[base]:.4f}.")
    c3.metric("Невозвратов на 10 000 кредитов", f"{bad_best:.0f} вместо {bad_base:.0f}",
              f"{(bad_best - bad_base) / bad_base:+.0%}", delta_color="inverse", border=True,
              help="Если банк одобряет 70% заявок — самых надёжных по оценке модели.")
    if stats and stats["ci_low"] > 0:
        st.success("**Ответ банку:** нейросеть на полной истории точнее стандартного подхода. Проверено на "
                   "самых поздних заявках, которых модель не видела, и разница не случайна.",
                   icon=":material/check_circle:")


def _idea() -> None:
    st.header("Идея проекта: одна модель на все данные")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.html(_flow_html())
    with right:
        st.markdown(
            "Обычно у банка для каждого вида данных своя модель, и каждая выдаёт **одно число**. "
            "Потом эти числа смешивают. При сжатии в одно число часть информации теряется.\n\n"
            "В проекте **одна нейросеть** получает от каждого источника подробное описание клиента "
            "и сама решает, как их сочетать.\n\n"
            "**Удобно:** если каких-то данных о клиенте нет, модель всё равно работает. "
            "Новый вид данных добавляется без переделки модели."
        )


def _where_next() -> None:
    st.header("Что посмотреть дальше")
    from .nav import PAGES  # импорт здесь: модуль навигации сам импортирует эту страницу

    cols = st.columns(4)
    links = [
        ("results", "Все результаты простыми словами"),
        ("client", "Попробовать оценить клиента"),
        ("glossary", "Что значат термины"),
        ("expert", "Подробности и обучение"),
    ]
    for col, (key, text) in zip(cols, links):
        with col.container(border=True):
            st.page_link(PAGES[key], label=text, width="stretch")


def _flow_html() -> str:
    """Схема модели обычным HTML: работает без интернета и в светлой, и в тёмной теме."""
    sources = "".join(f'<div class="box src">{name}</div>' for name in SOURCES)
    return (FLOW_CSS + '<div class="flow">'
            f'<div class="col">{sources}</div><div class="arrow">→</div>'
            '<div class="box mix">Единая нейросеть<br>объединяет всё,<br>что есть о клиенте</div>'
            '<div class="arrow">→</div><div class="box out">Вероятность<br>невозврата</div>'
            '<div class="arrow">→</div><div class="box out">Одобрить<br>или отказать</div></div>')
