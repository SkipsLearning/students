"""Раздел «Для специалистов»: подробные метрики, обучение модели с параметрами, отчёты всех запусков."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from unified_scoring.config import load_config
from unified_scoring.pipeline import run_experiment
from unified_scoring.report import MAIN_MODEL

from .common import ROOT, SOURCE_TITLES, baseline_of, hbar, list_runs, read_summary, run_label

CONFIGS = {
    "Задача банка: кредитные истории Альфа-Банка": "configs/alfa_bki.yaml",
    "Базовый эксперимент (5 источников)": "configs/synthetic.yaml",
    "Добавление нового источника insurance": "configs/synthetic_new_source.yaml",
    "Реальные данные Home Credit (нужно скачать)": "configs/home_credit.yaml",
}
HOME_CREDIT_DIR = ROOT / "data/home-credit-default-risk"
ALFA_DIR = ROOT / "data/data_for_competition"
LOG_LINES = 14


def render() -> None:
    st.title("Для специалистов")
    st.caption("Подробные метрики, обучение модели со своими параметрами и отчёты всех запусков.")
    train, reports = st.tabs(["Обучить модель", "Отчёты всех запусков"])
    with train:
        tab_train()
    with reports:
        tab_reports()


# ---------------------------------------------------------------------- подробный отчёт запуска
def test_table(summary: dict) -> pd.DataFrame:
    rows = []
    for model, by_split in summary["metrics"].items():
        m = by_split["test"]["main"]
        rows.append({"Модель": model, "ROC AUC": m["auc"], "Джини, %": m["gini"], "KS, %": m["ks"],
                     "Brier": m["brier"], "Средний PD": m["mean_pd"], "Факт. доля дефолтов": m["default_rate"]})
    return pd.DataFrame(rows).set_index("Модель")


def _interval_table(items: dict, first_col: str) -> pd.DataFrame:
    def verdict(v: dict) -> str:
        return "лучше (значимо)" if v["ci_low"] > 0 else "хуже (значимо)" if v["ci_high"] < 0 else "незначимо"

    return pd.DataFrame([
        {first_col: k, "Δ Джини, п.п.": v["delta"], "Интервал от": v["ci_low"], "Интервал до": v["ci_high"],
         "Вывод": verdict(v)}
        for k, v in items.items()
    ]).set_index(first_col)


def show_run(run_dir: Path) -> None:
    """Результаты одного запуска: метрики, значимость, кривая обучения, вклад источников."""
    s = read_summary(run_dir)
    table = test_table(s)
    base_name = baseline_of(s)
    nn, base = table.loc[MAIN_MODEL, "Джини, %"], table.loc[base_name, "Джини, %"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Джини единой модели (test)", f"{nn:.2f}%", f"{nn - base:+.2f} п.п. к базлайну")
    c2.metric(f"Джини базлайна: {base_name}", f"{base:.2f}%")
    c3.metric("Стабильность PSI train→test", f"{s['psi'][MAIN_MODEL]['train_test']:.4f}",
              help="PSI < 0.1 — распределение скора стабильно")

    st.subheader("Сравнение моделей на test")
    hbar(table["Джини, %"], "Джини на test, %")
    st.dataframe(table.style.format("{:.4f}").format("{:.2f}", subset=["Джини, %", "KS, %"]))

    fmt = {c: "{:+.2f}" for c in ("Δ Джини, п.п.", "Интервал от", "Интервал до")}
    if s.get("significance"):
        st.subheader("Значим ли прирост единой модели? (бутстрэп, 95%-интервал)")
        st.dataframe(_interval_table(s["significance"], "Единая NN против").style.format(fmt))
    if s.get("embeddings_vs_scores"):
        st.subheader("Ключевые сравнения подходов (бутстрэп, 95%-интервал)")
        st.dataframe(_interval_table(s["embeddings_vs_scores"], "Сравнение").style.format(fmt))

    left, right = st.columns(2)
    hist_path = run_dir / "history.csv"
    if hist_path.exists():
        left.subheader("Обучение по эпохам (Джини на val)")
        left.line_chart(pd.read_csv(hist_path).set_index("epoch")[["val_gini"]], height=250)
    right.subheader("Насколько упадёт Джини без источника")
    abl = pd.Series({SOURCE_TITLES.get(k, k): v["delta"] for k, v in s["ablation"].items()})
    with right:
        hbar(abl, "Изменение Джини, п.п.")

    st.subheader("Доля дефолтов среди одобренных")
    bus = pd.DataFrame(s["business"]).rename(index=lambda r: f"одобряем {float(r):.0%}")
    st.dataframe((100 * bus).style.format("{:.2f}%"))

    report = run_dir / "REPORT.md"
    if report.exists():
        with st.expander("Полный текстовый отчёт (REPORT.md)"):
            st.markdown(report.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- обучение
def _data_missing(cfg_path: Path) -> str | None:
    if "home_credit" in cfg_path.name and not (HOME_CREDIT_DIR / "application_train.csv").exists():
        return (f"Данные Home Credit не найдены в `{HOME_CREDIT_DIR}`. "
                "Как их скачать — в инструкции по подготовке (шаг про Home Credit).")
    if "alfa_bki" in cfg_path.name and not (ALFA_DIR / "train_target.csv").exists():
        return (f"Данные соревнования не найдены в `{ALFA_DIR}`. Скачайте data_for_competition "
                "с https://ods.ai/competitions/dl-fintech-bki и распакуйте туда.")
    return None


def _data_controls(cfg_path: Path, base_cfg) -> list[str]:
    st.markdown("**Данные**")
    if "alfa_bki" in cfg_path.name:
        n = st.select_slider("Сколько клиентов взять (из 3 млн)",
                             [50_000, 100_000, 250_000, 500_000, 1_000_000], 500_000,
                             help="1 000 000 — около 45 минут; 500 000 — около 15 минут; 50 000 — около 1 минуты")
        overrides = [f"data.alfa_bki.n_clients={n}", f"data.alfa_bki.path={ALFA_DIR}"]
    elif "home_credit" in cfg_path.name:
        nrows = st.select_slider("Сколько заявок взять", [20_000, 50_000, 100_000, 307_511], 50_000)
        overrides = [f"data.home_credit.nrows={nrows}", f"data.home_credit.path={HOME_CREDIT_DIR}"]
    else:
        n = st.select_slider("Число клиентов", [10_000, 20_000, 50_000, 100_000], 50_000)
        inter = st.slider("Сила связей между источниками", 0.0, 1.5,
                          float(base_cfg.data.synthetic.interaction_strength), 0.1,
                          help="0 — источники влияют на риск независимо друг от друга")
        overrides = [f"data.synthetic.n_clients={n}", f"data.synthetic.interaction_strength={inter}"]
    seed = st.number_input("Seed (случайность)", 0, 10_000, base_cfg.experiment.seed)
    return overrides + [f"experiment.seed={seed}"]


def _model_controls(base_cfg) -> list[str]:
    st.markdown("**Модель и обучение**")
    sd = st.slider("Source dropout (прятать источники при обучении)", 0.0, 0.5,
                   float(base_cfg.model.source_dropout), 0.05)
    flags = st.checkbox("Флаги наличия источников", base_cfg.model.use_presence_flags)
    epochs = st.slider("Максимум эпох", 2, 60, base_cfg.train.epochs)
    boosting = st.checkbox("Сравнить с бустингом (+20 с)", True)
    return [f"model.source_dropout={sd}", f"model.use_presence_flags={str(flags).lower()}",
            f"train.epochs={epochs}", f"baselines.boosting={str(boosting).lower()}"]


def _warm_start_picker(cfg_path: Path) -> str | None:
    if "new_source" not in cfg_path.name:
        return None
    candidates = [r for r in list_runs() if (r / "model.pt").exists() and "insurance" not in r.name]
    if not candidates:
        return None
    pick = st.selectbox("Перенести веса старых источников из запуска (warm start)",
                        ["не переносить"] + [run_label(r) for r in candidates])
    return None if pick == "не переносить" else str(ROOT / pick)


def tab_train() -> None:
    st.caption("Меняйте параметры и смотрите, как меняется качество. Один запуск на синтетике — около 30 секунд.")
    cfg_path = ROOT / CONFIGS[st.selectbox("Эксперимент", list(CONFIGS))]
    missing = _data_missing(cfg_path)
    if missing:
        st.warning(missing)
        return
    base_cfg = load_config(cfg_path)
    overrides = ["experiment.output_dir=" + str(ROOT / "runs")]

    c1, c2, c3 = st.columns(3)
    with c1:
        overrides += _data_controls(cfg_path, base_cfg)
    with c2:
        st.markdown("**Источники**")
        all_src = [s.name for s in base_cfg.sources]
        chosen = st.multiselect("Какие источники использовать", all_src,
                                [s.name for s in base_cfg.sources if s.enabled],
                                format_func=lambda x: SOURCE_TITLES.get(x, x))
        overrides += [f"sources.{name}.enabled={str(name in chosen).lower()}" for name in all_src]
    with c3:
        overrides += _model_controls(base_cfg)
    warm = _warm_start_picker(cfg_path)

    if not chosen:
        st.error("Выберите хотя бы один источник")
        return
    if st.button("Запустить обучение", type="primary"):
        _run_with_live_log(overrides, cfg_path, warm)
    last = st.session_state.get("last_run")
    if last:
        st.success(f"Готово: {run_label(Path(last))}")
        show_run(Path(last))


class _StreamlitLog(logging.Handler):
    """Показывает последние строки лога обучения прямо на странице."""

    def __init__(self, box) -> None:
        super().__init__()
        self.box = box
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if msg.startswith("Гиперпараметры"):
            return
        self.lines.append(msg.split("\n")[0])
        self.box.code("\n".join(self.lines[-LOG_LINES:]), language=None)


def _run_with_live_log(overrides: list[str], cfg_path: Path, warm: str | None) -> None:
    cfg = load_config(cfg_path, overrides)
    handler = _StreamlitLog(st.empty())
    logger = logging.getLogger("unified_scoring")
    logger.addHandler(handler)
    try:
        with st.spinner("Идёт обучение…"):
            st.session_state["last_run"] = str(run_experiment(cfg, warm_start=warm))
    except Exception as exc:  # показываем пользователю понятную ошибку вместо трассировки
        st.error(f"Обучение не удалось: {exc}")
        logger.exception("Ошибка обучения")
    finally:
        logger.removeHandler(handler)


def tab_reports() -> None:
    runs = list_runs()
    if not runs:
        st.info("Запусков пока нет — обучите модель на вкладке «Обучить модель».")
        return
    show_run(st.selectbox("Запуск", runs, format_func=run_label))
