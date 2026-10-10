"""Тесты: метрики vs sklearn, таргет, модель на любом наборе источников,
модульность конфига, сквозные smoke-тесты (синтетика и мок Home Credit)."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from unified_scoring import metrics  # noqa: E402
from unified_scoring.config import load_config  # noqa: E402
from unified_scoring.data.target import default_flag  # noqa: E402
from unified_scoring.inference import Scorer  # noqa: E402
from unified_scoring.model import SourceSpec, UnifiedScoringModel  # noqa: E402
from unified_scoring.pipeline import run_experiment  # noqa: E402


@pytest.fixture
def scores() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    s = np.round(y * 0.8 + rng.standard_normal(5000), 1)  # округление -> много совпадений
    return y, s


def test_auc_gini_match_sklearn(scores):
    y, s = scores
    assert metrics.roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-10)
    assert metrics.gini(y, s) == pytest.approx(100 * (2 * roc_auc_score(y, s) - 1), abs=1e-8)


def test_ks_and_pr_auc_match_sklearn(scores):
    y, s = scores
    fpr, tpr, _ = roc_curve(y, s)
    assert metrics.ks_statistic(y, s) == pytest.approx(100 * np.max(tpr - fpr), abs=1e-8)
    assert metrics.pr_auc(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-10)


def test_psi_brier():
    rng = np.random.default_rng(1)
    a = rng.random(10000)
    assert metrics.psi(a, rng.random(10000)) < 0.01
    assert metrics.psi(a, rng.random(10000) ** 3) > 0.25
    assert metrics.brier_score([0, 1], [0.0, 1.0]) == 0.0


def test_bootstrap_gini_diff():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < 0.2).astype(int)
    strong = y + 0.5 * rng.standard_normal(3000)
    weak = y + 3.0 * rng.standard_normal(3000)
    better = metrics.bootstrap_gini_diff(y, strong, weak, n_boot=200)
    assert better["ci_low"] > 0 and better["p_value"] < 0.01
    same = metrics.bootstrap_gini_diff(y, weak, weak, n_boot=200)
    assert same["delta"] == 0 and same["ci_low"] == same["ci_high"] == 0


def test_default_flag():
    paths = np.array([[0, 30, 60, 90, 0], [0, 30, 0, 30, 60], [0, 0, 0, 0, 120]])
    assert default_flag(paths, 4, 90).tolist() == [1, 0, 0]
    assert default_flag(paths, 4, 30).tolist() == [1, 1, 0]
    assert default_flag(paths, 5, 90).tolist() == [1, 0, 1]


def test_model_any_subset_of_sources():
    specs = [SourceSpec("a", "embedding", 8, [16, 8], 0.1), SourceSpec("b", "embedding", 4, [8, 4], 0.1),
             SourceSpec("t", "tabular", 6, [8, 4], 0.1, cardinalities=[5, 7], cat_emb_dims=[3, 4], n_numeric=2)]
    model = UnifiedScoringModel(specs, [16, 8], 0.2).eval()
    n = 4
    base = {"emb": {"a": torch.randn(n, 8), "b": torch.randn(n, 4)},
            "cat": torch.randint(0, 5, (n, 2)), "num": torch.randn(n, 2)}
    for combo in itertools.product([0, 1], repeat=3):
        if sum(combo) == 0:
            continue
        batch = dict(base, mask={k: torch.full((n,), bool(v)) for k, v in zip("abt", combo)})
        pd_ = model.predict_pd(batch)
        assert pd_.shape == (n,) and torch.all((pd_ >= 0) & (pd_ <= 1))
    # нет источника -> результат не зависит от содержимого эмбеддинга
    mask = {"a": torch.zeros(n, dtype=torch.bool), "b": torch.ones(n, dtype=torch.bool),
            "t": torch.ones(n, dtype=torch.bool)}
    p1 = model.predict_pd(dict(base, mask=mask))
    p2 = model.predict_pd(dict(base, mask=mask, emb={"a": torch.randn(n, 8) * 100, "b": base["emb"]["b"]}))
    assert torch.allclose(p1, p2)


def test_new_source_is_config_only():
    cfg = load_config(ROOT / "configs/synthetic_new_source.yaml")
    names = [s.name for s in cfg.enabled_sources]
    assert names[-1] == "insurance" and len(names) == 6
    base = load_config(ROOT / "configs/synthetic.yaml", ["sources.graph.enabled=false"])
    assert "graph" not in [s.name for s in base.enabled_sources]


def _small(cfg_path: Path, tmp: Path, extra: list[str]) -> Path:
    cfg = load_config(cfg_path, [f"experiment.output_dir={tmp}", "train.epochs=2", *extra])
    return run_experiment(cfg)


def test_synthetic_end_to_end(tmp_path):
    run = _small(ROOT / "configs/synthetic.yaml", tmp_path,
                 ["data.synthetic.n_clients=6000", "train.pos_weight=auto", "train.calibration=platt"])
    summary = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert set(summary["metrics"]) == {"Единая NN (эмбеддинги)", "LR на эмбеддингах", "LR на векторе скоров",
                                       "Бустинг на векторе скоров", "Бустинг на эмбеддингах"}
    sig = summary["significance"]["LR на векторе скоров"]
    assert sig["ci_low"] <= sig["delta"] <= sig["ci_high"]
    reqs = [json.loads(line) for line in (run / "sample_requests.jsonl").read_text(encoding="utf-8").splitlines()]
    pd_ = Scorer.from_run_dir(run).predict(reqs)
    assert np.all((pd_ >= 0) & (pd_ <= 1))
    with pytest.raises(ValueError):
        Scorer.from_run_dir(run).predict([{"client_id": "empty", "embeddings": {}}])


def test_home_credit_adapter_on_mock(tmp_path):
    from mock_home_credit import make_mock_home_credit

    data_dir = make_mock_home_credit(tmp_path / "hc")
    run = _small(ROOT / "configs/home_credit.yaml", tmp_path,
                 [f"data.home_credit.path={data_dir}", "data.home_credit.encoder_epochs=1",
                  "sources.pos.enabled=true"])
    summary = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert summary["targets"] == ["main"]
    assert (run / "domain_encoders.pt").exists()


def test_alfa_bki_adapter_on_mock(tmp_path):
    from mock_alfa_bki import make_mock_alfa_bki

    data_dir = make_mock_alfa_bki(tmp_path / "alfa")
    run = _small(ROOT / "configs/alfa_bki.yaml", tmp_path,
                 [f"data.alfa_bki.path={data_dir}", "data.alfa_bki.n_clients=2500",
                  "data.alfa_bki.encoder_epochs=1", "data.alfa_bki.cv_folds=2"])
    summary = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    models = set(summary["metrics"])
    assert {"GRU на сырой кредитной истории", "GRU по фолдам (ансамбль)",
            "Бустинг на агрегатах (стандартный подход)"} <= models
    assert summary["data"]["n"] == 2500
    assert summary["data"]["split_scheme"] == "time"


def test_id_periods_follow_id_order():
    from unified_scoring.data.alfa_bki import id_periods

    periods = id_periods(np.array([50, 10, 40, 20, 30, 60]), n_periods=3)
    assert periods.tolist() == [2, 0, 1, 0, 1, 2]


def _render_page(root: str, module: str) -> None:
    """Скрипт для AppTest: исполняется отдельно, поэтому импорты внутри."""
    import importlib
    import sys

    sys.path.insert(0, root)
    importlib.import_module(f"ui.{module}").render()


@pytest.mark.parametrize("page", ["home", "results", "client", "glossary", "expert"])
def test_ui_pages_render_without_errors(page):
    from streamlit.testing.v1 import AppTest

    if not (ROOT / "results/synthetic_reference/metrics.json").exists():
        pytest.skip("нет эталонных результатов")
    app = AppTest.from_function(_render_page, args=(str(ROOT), page), default_timeout=180)
    app.run()
    assert not app.exception, [e.value for e in app.exception]


def test_ui_headline_picks_strongest_significant_comparison():
    from ui.common import headline, spaced

    summary = {"metrics": {"Бустинг на агрегатах (стандартный подход)": {}},
               "significance": {"Бустинг на агрегатах (стандартный подход)": {"delta": 2.5}},
               "embeddings_vs_scores": {
                   "GRU по фолдам (ансамбль) против «Бустинг на агрегатах (стандартный подход)»": {"delta": 3.6},
                   "LR на эмбеддингах против «LR на векторе скоров»": {"delta": 9.0}}}
    best, base, stats = headline(summary)
    assert (best, base, stats["delta"]) == ("GRU по фолдам (ансамбль)", "Бустинг на агрегатах (стандартный подход)", 3.6)
    assert spaced(1_000_000) == "1 000 000"
