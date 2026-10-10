"""Полный цикл эксперимента: данные -> обучение -> оценка -> базлайны -> отчёт -> артефакты."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .baseline import BoostingBaseline, EmbeddingLRBaseline, ScoreVectorBaseline
from .config import Config, SourceConfig
from .data import ScoringData, prepare_data
from .data.tabular import TabularPreprocessor
from .data.torch_dataset import ScoringTorchDataset, build_dataset
from .inference import ModelBundle, Scorer
from .metrics import bootstrap_gini_diff, classification_report, gini, psi
from .model import SourceSpec, UnifiedScoringModel, load_matching_weights
from .report import BASELINE, MAIN_MODEL, build_markdown, format_console
from .train import PlattCalibrator, predict_logits, train_model
from .utils import get_logger, resolve_device, save_json, set_seed

TARGET_DEFINITIONS = {
    "home_credit": "TARGET Home Credit (проблемы с оплатой)",
    "alfa_bki": "flag — дефолт клиента по кредиту (данные Альфа-Банка)",
}
EMB_LR = "LR на эмбеддингах"
GBM_SCORES = "Бустинг на векторе скоров"
GBM_EMB = "Бустинг на эмбеддингах"


def build_specs(sources: list[SourceConfig], pre: TabularPreprocessor | None) -> list[SourceSpec]:
    specs = []
    for s in sources:
        spec = SourceSpec(name=s.name, kind=s.type, dim=s.dim,
                          tower_hidden=list(s.tower.hidden), tower_dropout=s.tower.dropout)
        if s.type == "tabular":
            assert pre is not None
            spec.cardinalities = pre.cardinalities
            spec.cat_emb_dims = pre.embedding_dims
            spec.n_numeric = len(pre.numerical)
        specs.append(spec)
    return specs


def build_model(cfg: Config, pre: TabularPreprocessor | None) -> UnifiedScoringModel:
    return UnifiedScoringModel(
        specs=build_specs(cfg.enabled_sources, pre),
        head_hidden=list(cfg.model.head_hidden),
        head_dropout=cfg.model.head_dropout,
        use_presence_flags=cfg.model.use_presence_flags,
        source_dropout=cfg.model.source_dropout,
        missing_embedding=cfg.model.missing_embedding,
    )


def _with_masks(ds: ScoringTorchDataset, masks: dict[str, torch.Tensor]) -> ScoringTorchDataset:
    out = copy.copy(ds)
    out.mask = masks
    return out


def _safe_report(y: np.ndarray, p: np.ndarray) -> dict[str, float] | None:
    if len(y) == 0 or y.min() == y.max():
        return None
    return classification_report(y, p)


def _py(v: Any) -> Any:
    """numpy-скаляры -> встроенные типы Python (для JSON)."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def _sample_requests(test: ScoringData, sources: list[SourceConfig], k: int = 3) -> list[dict[str, Any]]:
    """Демонстрационные запросы: все источники / один источник / только анкета."""
    emb_names = [s.name for s in sources if s.type == "embedding"]
    tab = next((s.name for s in sources if s.type == "tabular"), None)
    reqs = []
    for i in range(min(k, len(test))):
        rec: dict[str, Any] = {"client_id": str(test.client_id[i]), "embeddings": {}}
        for name in emb_names:
            if test.masks[name][i]:
                rec["embeddings"][name] = test.embeddings[name][i].round(5).tolist()
        if tab and test.masks[tab][i]:
            row = test.tabular.iloc[i].to_dict()
            rec["tabular"] = {c: _py(v) for c, v in row.items()}
        reqs.append(rec)
    # клиент только с одним источником-эмбеддингом
    if emb_names:
        name = emb_names[0]
        j = int(np.flatnonzero(test.masks[name])[0])
        reqs.append({"client_id": f"{test.client_id[j]}_only_{name}",
                     "embeddings": {name: test.embeddings[name][j].round(5).tolist()}})
    if tab:
        j = int(np.flatnonzero(test.masks[tab])[0])
        row = test.tabular.iloc[j].to_dict()
        reqs.append({"client_id": f"{test.client_id[j]}_only_{tab}", "embeddings": {},
                     "tabular": {c: _py(v) for c, v in row.items()}})
    return reqs


def run_experiment(cfg: Config, warm_start: str | None = None) -> Path:
    set_seed(cfg.experiment.seed)
    if cfg.train.num_threads:
        torch.set_num_threads(cfg.train.num_threads)
    run_dir = Path(cfg.experiment.output_dir) / f"{cfg.experiment.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_file=run_dir / "train.log")
    cfg.save(run_dir / "config.yaml")
    logger.info("Запуск %s, результаты: %s", cfg.experiment.name, run_dir)
    logger.info("Гиперпараметры:\n%s", json.dumps(cfg.to_dict(), ensure_ascii=False, indent=1, default=str))
    device = resolve_device(cfg.train.device)

    # ------------------------------------------------------------------ данные
    data, splits, encoders = prepare_data(cfg, logger)
    parts = {k: data.subset(v) for k, v in splits.items()}
    target_key = cfg.target.train_on
    targets = list(data.targets)
    for name, part in parts.items():
        logger.info("%-10s n=%7d  DR(main)=%.4f", name, len(part),
                    part.targets["main"].mean() if len(part) else float("nan"))
    logger.info("Покрытие источников:\n%s", data.availability().round(3).to_string())

    sources = cfg.enabled_sources
    tab_src = next((s for s in sources if s.type == "tabular"), None)
    pre = TabularPreprocessor(copy.deepcopy(tab_src.tabular)).fit(parts["train"].tabular) if tab_src else None

    # ------------------------------------------------------------------ модель
    set_seed(cfg.experiment.seed)
    model = build_model(cfg, pre)
    train = parts["train"]
    model.init_default_embeddings({
        name: emb[train.masks[name]].mean(axis=0)
        for name, emb in train.embeddings.items() if train.masks[name].any()
    })
    if warm_start:
        state = torch.load(Path(warm_start) / "model.pt", map_location="cpu", weights_only=False)["state_dict"]
        moved = load_matching_weights(model, state)
        logger.info("Warm start из %s: перенесено %d тензоров", warm_start, len(moved))

    ds = {k: build_dataset(parts[k], pre, target_key) for k in ("train", "val", "test")}
    result = train_model(model, ds["train"], ds["val"], cfg.train, device, cfg.experiment.seed, logger)
    model = result.model.cpu()
    pd.DataFrame(result.history).to_csv(run_dir / "history.csv", index=False)

    logits = {k: predict_logits(model, ds[k], torch.device("cpu")) for k in ds}
    calibrator = None
    if cfg.train.calibration == "platt":
        calibrator = PlattCalibrator.fit(logits["val"], parts["val"].targets[target_key])
        logger.info("Platt-калибровка на val: a=%.3f b=%.3f", calibrator.a, calibrator.b)
    bundle = ModelBundle(model, pre, tab_src.name if tab_src else None, calibrator)
    bundle.save(run_dir / "model.pt")
    if encoders:
        torch.save(encoders, run_dir / "domain_encoders.pt")

    preds: dict[str, dict[str, np.ndarray]] = {MAIN_MODEL: {k: bundle.logits_to_pd(v) for k, v in logits.items()}}

    # ------------------------------------------------------------------ базлайны
    logger.info("Базлайн: доменные скоры + LR на векторе скоров")
    sv = ScoreVectorBaseline(sources, target_key, logger).fit(parts["domain_fit"], train)
    logger.info("Базлайн: LR на конкатенации эмбеддингов")
    el = EmbeddingLRBaseline(sources, target_key).fit(train)
    baselines: list[tuple[str, Any]] = [(BASELINE, sv), (EMB_LR, el)]
    boosted = ((GBM_SCORES, sv.design), (GBM_EMB, el.design)) if cfg.baselines.boosting else ()
    for name, design in boosted:
        logger.info("Базлайн: %s", name)
        gbm = BoostingBaseline(name, design, target_key, cfg.experiment.seed).fit(train, parts["val"])
        logger.info("  деревьев (подобрано по val): %d", gbm.best_trees)
        baselines.append((name, gbm))
    single_fit = train
    if cfg.baselines.single_source_data == "domain_fit+train":
        single_fit = data.subset(np.sort(np.concatenate([splits["domain_fit"], splits["train"]])))
    for src_name, title in (cfg.baselines.single_source if cfg.baselines.boosting else {}).items():
        src = next((s for s in sources if s.name == src_name), None)
        if src is None:
            raise ValueError(f"baselines.single_source: нет включённого источника {src_name}")
        logger.info("Базлайн: %s (обучение: %s, %d клиентов)", title, cfg.baselines.single_source_data,
                    len(single_fit))
        design = EmbeddingLRBaseline([src], target_key).fit(single_fit).design
        gbm = BoostingBaseline(title, design, target_key, cfg.experiment.seed).fit(single_fit, parts["val"])
        logger.info("  деревьев (подобрано по val): %d", gbm.best_trees)
        baselines.append((title, gbm))
    for name, mdl in baselines:
        preds[name] = {k: mdl.predict_pd(parts[k]) for k in ("train", "val", "test")}
    # готовые прогнозы внешних моделей (например, доменной GRU на сырой истории)
    for name in data.extra_scores:
        preds[name] = {k: 1.0 / (1.0 + np.exp(-parts[k].extra_scores[name])) for k in ("train", "val", "test")}

    # ------------------------------------------------------------------ метрики
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, by_split in preds.items():
        metrics[model_name] = {
            split: {t: _safe_report(parts[split].targets[t], by_split[split]) for t in targets}
            for split in ("val", "test")
        }
    psi_res = {m: {"train_val": psi(p["train"], p["val"]), "train_test": psi(p["train"], p["test"])}
               for m, p in preds.items()}

    test = parts["test"]
    n_avail = np.sum(np.stack([test.masks[s.name] for s in sources], axis=1), axis=1)
    segments: dict[str, dict[str, dict[str, float]]] = {}
    y_test = test.targets[target_key]
    # на очень больших test 300 повторов бутстрэпа достаточно для интервала и в разы быстрее
    n_boot = 1000 if len(y_test) <= 100_000 else 300
    # значимость прироста: парный бутстрэп разницы Джини единой модели и каждого базлайна
    significance = {
        name: bootstrap_gini_diff(y_test, preds[MAIN_MODEL]["test"], p["test"], n_boot=n_boot,
                                  seed=cfg.experiment.seed)
        for name, p in preds.items() if name != MAIN_MODEL
    }
    # главный тезис: тот же смеситель на эмбеддингах против него же на скорах
    embeddings_vs_scores = {
        f"{emb} против «{sc}»": bootstrap_gini_diff(y_test, preds[emb]["test"], preds[sc]["test"],
                                                   n_boot=n_boot, seed=cfg.experiment.seed)
        for emb, sc in ((EMB_LR, BASELINE), (GBM_EMB, GBM_SCORES)) if emb in preds and sc in preds
    }
    # подходы из постановки банка: готовая модель на сырой истории против бустинга на одном источнике
    for ext in data.extra_scores:
        for title in (cfg.baselines.single_source if cfg.baselines.boosting else {}).values():
            embeddings_vs_scores[f"{ext} против «{title}»"] = bootstrap_gini_diff(
                y_test, preds[ext]["test"], preds[title]["test"], n_boot=n_boot, seed=cfg.experiment.seed)
    for model_name, by_split in preds.items():
        seg = {}
        for g in np.unique(n_avail):
            sel = n_avail == g
            if sel.sum() >= 200 and 0 < y_test[sel].sum() < sel.sum():
                seg[str(int(g))] = {"n": int(sel.sum()), "gini": gini(y_test[sel], by_split["test"][sel])}
        segments[model_name] = seg

    # бизнес-срез: уровень дефолтов среди одобренных при фиксированном уровне одобрения
    business = {}
    for model_name, by_split in preds.items():
        order = np.argsort(by_split["test"], kind="mergesort")
        business[model_name] = {
            str(ar): float(y_test[order[: int(round(ar * len(order)))]].mean())
            for ar in (0.5, 0.7, 0.8, 0.9)
        }

    full_gini = gini(y_test, preds[MAIN_MODEL]["test"])
    ablation = {}
    for s in sources:
        masks = dict(ds["test"].mask)
        masks[s.name] = torch.zeros_like(masks[s.name])
        # клиенты, у которых это единственный источник, исключаются из абляции
        keep = torch.stack(list(masks.values()), 1).any(1).numpy()
        lg = predict_logits(model, _with_masks(ds["test"], masks), torch.device("cpu"))
        g = gini(y_test[keep], lg[keep])
        ablation[s.name] = {"gini": g, "delta": g - gini(y_test[keep], preds[MAIN_MODEL]["test"][keep])}

    summary = {
        "experiment": cfg.experiment.name,
        "targets": targets,
        "data": {
            "kind": cfg.data.kind,
            "n": len(data),
            "split_scheme": cfg.split.scheme,
            "target_definition": (
                f"просрочка {cfg.target.dpd_threshold}+ дней в пределах {cfg.target.horizon_months} мес. "
                f"(короткое: {cfg.target.short_dpd_threshold}+)"
                if cfg.data.kind == "synthetic" else TARGET_DEFINITIONS[cfg.data.kind]
            ),
            "splits": {k: {"n": len(v), "default_rate": float(v.targets["main"].mean()) if len(v) else 0.0}
                       for k, v in parts.items()},
            "availability": data.availability().to_dict(),
        },
        "training": {"best_epoch": result.best_epoch, "best_val_gini": result.best_val_gini,
                     "epochs_run": len(result.history),
                     "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "metrics": metrics,
        "psi": psi_res,
        "significance": significance,
        "embeddings_vs_scores": embeddings_vs_scores,
        "segments": segments,
        "ablation": ablation,
        "business": business,
        "full_test_gini": full_gini,
    }
    save_json(summary, run_dir / "metrics.json")
    (run_dir / "REPORT.md").write_text(build_markdown(summary), encoding="utf-8")
    logger.info("Итоговые метрики:%s", format_console(summary))

    # ------------------------------------------------------------------ инференс-демо
    reqs = _sample_requests(test, sources)
    with open(run_dir / "sample_requests.jsonl", "w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    scorer = Scorer.from_run_dir(run_dir)
    pd_demo = scorer.predict(reqs)
    for r, p in zip(reqs, pd_demo):
        logger.info("Демо-инференс %-22s источники=%s PD=%.4f", r["client_id"],
                    list(r["embeddings"]) + (["tabular"] if "tabular" in r else []), p)
    n_full = min(3, len(test))
    if not np.allclose(pd_demo[:n_full], preds[MAIN_MODEL]["test"][:n_full], atol=1e-4):
        logger.warning("Инференс расходится с пайплайном — проверьте предобработку")
    logger.info("Готово. Отчёт: %s", run_dir / "REPORT.md")
    return run_dir
