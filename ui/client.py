"""Страница «Оценить клиента»: демо на учебных данных — светофор риска и что повлияло на оценку."""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from unified_scoring.config import load_config
from unified_scoring.data.container import make_splits
from unified_scoring.data.synthetic import generate_synthetic
from unified_scoring.data.torch_dataset import build_dataset
from unified_scoring.inference import Scorer
from unified_scoring.train import predict_logits

from .common import ROOT, SOURCE_TITLES, list_runs, read_summary, run_label

APPROVAL_RATE = 0.7       # банк одобряет 70% самых надёжных заявок
LOW_RISK_SHARE = 0.5      # самая надёжная половина — «зелёная» зона
EXAMPLE_POOL = 0.05       # примеры надёжных / рискованных берём из 5% краёв
REFERENCE_RUN = ROOT / "results/synthetic_reference"
GOOD_COLOR, BAD_COLOR = "#2e9e5b", "#d9534f"

SOURCE_HINTS = {
    "card": "траты по картам", "account": "движение денег по счёту", "bki": "прошлые кредиты и просрочки",
    "graph": "с кем клиент связан переводами", "tabular": "доход, возраст, сумма кредита…",
    "insurance": "страховые полисы",
}
FIELD_TITLES = {
    "region": "Регион", "occupation": "Профессия", "education": "Образование", "channel": "Где подана заявка",
    "product": "Что берёт", "income": "Доход в месяц", "age": "Возраст", "credit_amount": "Сумма кредита",
    "debt_to_income": "Долг к доходу", "months_at_job": "Стаж на работе, мес.", "n_dependents": "Иждивенцев",
}
PRODUCTS = {"auto": "Автокредит", "card": "Кредитная карта", "cash": "Кредит наличными", "pos": "Рассрочка в магазине"}
CODE_PREFIX = {"REG": "Регион", "OCC": "Группа профессий", "EDU": "Уровень", "CH": "Канал"}


def option_title(value: str) -> str:
    """Учебные данные обезличены (REG_03): показываем по-человечески — «Регион 4»."""
    if value in PRODUCTS:
        return PRODUCTS[value]
    prefix, _, num = str(value).partition("_")
    return f"{CODE_PREFIX[prefix]} {int(num) + 1}" if prefix in CODE_PREFIX and num.isdigit() else str(value)


@st.cache_resource
def load_scorer(run_dir: str) -> Scorer:
    return Scorer.from_run_dir(run_dir)


@st.cache_resource
def load_test_clients(run_dir: str):
    """Восстанавливает test-выборку синтетики по сохранённому конфигу запуска и считает PD всем клиентам."""
    cfg = load_config(Path(run_dir) / "config.yaml")
    data = generate_synthetic(cfg)
    test = data.subset(make_splits(data, cfg)["test"])
    bundle = load_scorer(run_dir).bundle
    logits = predict_logits(bundle.model, build_dataset(test, bundle.preprocessor, "main"), torch.device("cpu"))
    return test, bundle.logits_to_pd(logits)


def _pick_model() -> Path | None:
    runs = [r for r in list_runs() if (r / "model.pt").exists() and read_summary(r)["data"]["kind"] == "synthetic"]
    if not runs:
        return None
    default = REFERENCE_RUN if REFERENCE_RUN in runs else runs[0]
    if len(runs) == 1:
        return default
    with st.expander("Выбрать другую обученную модель"):
        return st.selectbox("Модель", runs, runs.index(default), format_func=run_label, key="score_run")


def render() -> None:
    st.title("Оценить клиента")
    st.caption("Демонстрация на учебных данных: 50 000 искусственных клиентов, у которых бывает до пяти видов "
               "данных. Выберите клиента, выключайте данные и меняйте анкету — оценка пересчитается сразу.")
    run_dir = _pick_model()
    if run_dir is None:
        st.info("Нет обученной модели на учебных данных. Обучите её в разделе «Для специалистов».")
        return
    scorer = load_scorer(str(run_dir))
    with st.spinner("Загружаю клиентов…"):
        test, all_pd = load_test_clients(str(run_dir))

    i = _client_picker(all_pd)
    left, right = st.columns([5, 6], gap="large")
    with left:
        use, tab_values = _data_panel(test, i, scorer)
    if not any(use.values()):
        right.error("Включите хотя бы один вид данных — совсем без данных модель не может оценить клиента.")
        return
    pd_value = float(scorer.predict([_client_record(test, i, use, tab_values)])[0])
    with right:
        _result_card(pd_value, all_pd, bool(test.targets["main"][i]))
    st.divider()
    _influence(test, i, scorer, use, tab_values, pd_value)


def _client_picker(all_pd: np.ndarray) -> int:
    n = len(all_pd)
    st.session_state.setdefault("client", 0)
    order = np.argsort(all_pd)
    pool = max(1, int(EXAMPLE_POOL * n))
    rng = np.random.default_rng()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    if c1.button("Случайный клиент", icon=":material/shuffle:", width="stretch"):
        st.session_state["client"] = int(rng.integers(n))
    if c2.button("Надёжный", icon=":material/thumb_up:", width="stretch"):
        st.session_state["client"] = int(rng.choice(order[:pool]))
    if c3.button("Рискованный", icon=":material/warning:", width="stretch"):
        st.session_state["client"] = int(rng.choice(order[-pool:]))
    return int(c4.number_input(f"Номер клиента (0–{n - 1})", 0, n - 1, key="client"))


def _data_panel(test, i: int, scorer: Scorer) -> tuple[dict[str, bool], dict | None]:
    st.subheader("Что банк знает о клиенте")
    st.caption("Выключите переключатель — и модель «забудет» эти данные.")
    use = {}
    for name in scorer.source_names:
        has = bool(test.masks[name][i])
        label = SOURCE_TITLES.get(name, name)
        hint = SOURCE_HINTS.get(name, "") if has else "у этого клиента таких данных нет"
        use[name] = st.toggle(f"{label} — {hint}" if hint else label, has, disabled=not has, key=f"use_{name}_{i}")
    tab_values = None
    if "tabular" in scorer.source_names and test.masks["tabular"][i] and use.get("tabular"):
        with st.expander("Анкета клиента — можно поменять", icon=":material/edit:"):
            tab_values = _tabular_editor(test, i, scorer)
    return use, tab_values


def _tabular_editor(test, i: int, scorer: Scorer) -> dict:
    pre = scorer.bundle.preprocessor
    row = test.tabular.iloc[i]
    values: dict = {}
    cols = st.columns(2)
    for j, c in enumerate(pre.categorical):
        options = sorted(pre.vocab[c])
        current = row[c] if row[c] in pre.vocab[c] else options[0]
        values[c] = cols[j % 2].selectbox(FIELD_TITLES.get(c, c), options, options.index(current),
                                          format_func=option_title, key=f"cat_{c}_{i}")
    for j, c in enumerate(pre.numerical, start=len(pre.categorical)):
        v = row[c]
        values[c] = cols[j % 2].number_input(FIELD_TITLES.get(c, c), value=None if pd.isna(v) else float(v),
                                             key=f"num_{c}_{i}", placeholder="не указано")
    return values


def _client_record(test, i: int, use: dict[str, bool], tab_values: dict | None) -> dict:
    rec: dict = {"client_id": str(test.client_id[i]), "embeddings": {}}
    for name, emb in test.embeddings.items():
        if use.get(name):
            rec["embeddings"][name] = emb[i].tolist()
    if tab_values is not None and use.get("tabular"):
        rec["tabular"] = tab_values
    return rec


def _result_card(pd_value: float, all_pd: np.ndarray, actual_default: bool) -> None:
    approve_cut = float(np.quantile(all_pd, APPROVAL_RATE))
    low_cut = float(np.quantile(all_pd, LOW_RISK_SHARE))
    st.subheader("Оценка модели")
    with st.container(border=True):
        st.metric("Вероятность, что клиент не вернёт кредит", f"{pd_value:.1%}",
                  help="Это PD (probability of default) — дефолт: просрочка 90+ дней в течение года.")
        per100 = round(100 * pd_value)
        safer_than = float((all_pd > pd_value).mean())
        st.markdown(f"Из **100** похожих клиентов не вернут кредит примерно "
                    f"**{per100 if per100 >= 1 else 'меньше 1'}**. "
                    f"У типичного клиента — **{np.median(all_pd):.1%}**.  \n"
                    f"Этот клиент надёжнее, чем **{safer_than:.0%}** заявителей.")
        if pd_value <= low_cut:
            st.success("**Низкий риск — одобрить**", icon=":material/check_circle:")
        elif pd_value <= approve_cut:
            st.warning("**Риск выше обычного — одобрить, но осторожно** (например, меньшую сумму)",
                       icon=":material/error:")
        else:
            st.error("**Высокий риск — отказать**", icon=":material/cancel:")
        st.caption(f"Правило: банк одобряет {APPROVAL_RATE:.0%} самых надёжных заявок "
                   f"(порог — {approve_cut:.1%}). Зелёная зона — самая надёжная половина.")
    with st.expander("Что было с этим клиентом на самом деле"):
        st.markdown("Клиент **не вернул** кредит." if actual_default else "Клиент **вернул** кредит.")
        st.caption("Модель ошибается на отдельных клиентах — её точность видна только на тысячах заявок. "
                   "Смотрите страницу «Результаты».")


def _influence(test, i: int, scorer: Scorer, use: dict[str, bool], tab_values: dict | None, pd_value: float) -> None:
    st.subheader("Что повлияло на оценку")
    active = [n for n, on in use.items() if on]
    if len(active) < 2:
        st.caption("Включён один вид данных — сравнивать не с чем.")
        return
    effects = {}
    for name in active:
        rec = _client_record(test, i, {**use, name: False}, tab_values)
        effects[SOURCE_TITLES.get(name, name)] = 100 * (pd_value - float(scorer.predict([rec])[0]))
    st.caption("Насколько эти данные изменили риск клиента (в процентных пунктах). "
               "Вправо — данные **повышают** риск, влево — **снижают**, то есть говорят в пользу клиента.")
    _influence_chart(pd.Series(effects).sort_values())


def _influence_chart(effects: pd.Series) -> None:
    """Столбцы от нуля: зелёные — данные в пользу клиента, красные — повышают риск."""
    df = pd.DataFrame({"source": effects.index, "pp": effects.values,
                       "effect": np.where(effects.values > 0, "повышает риск", "снижает риск")})
    base = alt.Chart(df).encode(
        y=alt.Y("source:N", sort=None, title=None, axis=alt.Axis(labelLimit=400)),
        x=alt.X("pp:Q", title="Влияние на вероятность невозврата, п.п."),
    )
    bars = base.mark_bar().encode(color=alt.Color(
        "effect:N", title=None, legend=alt.Legend(orient="bottom"),
        scale=alt.Scale(domain=["снижает риск", "повышает риск"], range=[GOOD_COLOR, BAD_COLOR])))
    right = base.transform_filter("datum.pp > 0").mark_text(align="left", dx=4, color="#888888")
    left = base.transform_filter("datum.pp <= 0").mark_text(align="right", dx=-4, color="#888888")
    label = alt.Text("pp:Q", format="+.1f")
    chart = bars + right.encode(text=label) + left.encode(text=label)
    st.altair_chart(chart.properties(height=44 * len(df) + 70), width="stretch")
