"""Конфигурация эксперимента: dataclass-схема + загрузка/сохранение YAML.

Вся параметризация (источники, размерности, архитектура, обучение, таргет,
разбиение) живёт в конфиге. Новый источник добавляется новым элементом
списка ``sources`` — ядро модели при этом не меняется.
"""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str = "unified_scoring"
    seed: int = 42
    output_dir: str = "runs"


@dataclass
class SyntheticConfig:
    """Параметры синтетического генератора (см. data/synthetic.py)."""

    n_clients: int = 50_000
    n_months: int = 24            # число месяцев выдачи (для out-of-time разбиения)
    latent_dim: int = 10          # число скрытых факторов риска (>= 10)
    default_rate: float = 0.07    # целевой уровень дефолтов по основному определению
    cure_prob: float = 0.35       # вероятность выхода из просрочки за месяц
    interaction_strength: float = 0.8  # вклад межисточниковых взаимодействий в риск
    risk_scale: float = 1.8       # сила связи латентного риска с вероятностью просрочки
    drift: float = 0.3            # дрейф популяции во времени (для PSI)


@dataclass
class HomeCreditConfig:
    """Параметры адаптера Kaggle Home Credit Default Risk."""

    path: str = "data/home-credit-default-risk"
    nrows: int | None = None      # ограничение строк application_train (для отладки)
    encoder_hidden: int = 64      # скрытый размер GRU доменных моделей
    encoder_epochs: int = 6
    encoder_batch_size: int = 512
    encoder_lr: float = 1e-3


@dataclass
class AlfaBkiConfig:
    """Параметры адаптера соревнования Альфа-Банка по данным БКИ (ods.ai/competitions/dl-fintech-bki)."""

    path: str = "data/data_for_competition"
    n_clients: int | None = 500_000  # None — все 3 млн (не помещается в 16 ГБ RAM)
    take: str = "latest"          # latest | earliest — какие клиенты по id (id растёт со временем)
    max_len: int = 32             # последних кредитов в истории (покрывает 99.9% клиентов)
    encoder_hidden: int = 64      # скрытый размер доменной GRU
    encoder_epochs: int = 4
    encoder_batch_size: int = 1024
    encoder_lr: float = 2e-3
    cv_folds: int = 5             # GRU по фолдам на domain_fit+train (как у 7-го места); 0 — не обучать
    cv_score_source: str = "bki_gru_cv"  # источник, в который подаётся out-of-fold прогноз GRU (стекинг)


@dataclass
class DataConfig:
    kind: str = "synthetic"       # synthetic | home_credit | alfa_bki
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    home_credit: HomeCreditConfig = field(default_factory=HomeCreditConfig)
    alfa_bki: AlfaBkiConfig = field(default_factory=AlfaBkiConfig)


@dataclass
class TargetConfig:
    """Определение дефолта: просрочка K+ дней в пределах N месяцев."""

    horizon_months: int = 12      # N
    dpd_threshold: int = 90       # K (основное определение)
    short_dpd_threshold: int = 30  # K для «короткого» определения (проверка устойчивости)
    train_on: str = "main"        # main | short — на каком таргете обучать


@dataclass
class SplitConfig:
    """Разбиение. ``time`` — out-of-time по периодам, ``random`` — стратифицированное.

    Часть ``domain_fit`` резервируется под обучение доменных моделей
    (энкодеров источников / скоров базлайна), чтобы эмбеддинги и скоры
    на train/val/test были out-of-sample.
    """

    scheme: str = "time"
    domain_fit_periods: list[int] = field(default_factory=lambda: [0, 5])
    train_periods: list[int] = field(default_factory=lambda: [6, 15])
    val_periods: list[int] = field(default_factory=lambda: [16, 19])
    test_periods: list[int] = field(default_factory=lambda: [20, 23])
    fractions: dict[str, float] = field(
        default_factory=lambda: {"domain_fit": 0.3, "train": 0.4, "val": 0.15, "test": 0.15}
    )


@dataclass
class TowerConfig:
    """MLP источника: Linear -> ReLU -> Dropout -> Linear -> ReLU."""

    hidden: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2


@dataclass
class TabularConfig:
    """Табличный источник: Entity Embeddings для категорий + нормализация чисел."""

    categorical: list[str] | None = None   # None -> автоопределение по dtype
    numerical: list[str] | None = None     # None -> все остальные числовые колонки
    exclude: list[str] = field(default_factory=list)
    log_patterns: list[str] = field(default_factory=list)  # подстроки имён -> signed log1p
    min_category_freq: int = 10
    max_emb_dim: int = 16
    scaler: str = "standard"               # standard | minmax


@dataclass
class SourceConfig:
    """Описание одного источника данных.

    type:
      * ``embedding`` — на вход приходит готовый эмбеддинг размерности ``dim``;
      * ``tabular``  — сырые табличные признаки, кодируемые Entity Embeddings
        в вектор размерности ``dim``.
    ``synthetic`` / ``home_credit`` — параметры, которые читает соответствующий
    загрузчик данных; ядро модели их не использует.
    """

    name: str
    dim: int
    type: str = "embedding"
    enabled: bool = True
    tower: TowerConfig = field(default_factory=TowerConfig)
    tabular: TabularConfig = field(default_factory=TabularConfig)
    synthetic: dict[str, Any] = field(default_factory=dict)
    home_credit: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    head_hidden: list[int] = field(default_factory=lambda: [128, 64])
    head_dropout: float = 0.3
    use_presence_flags: bool = True
    missing_embedding: str = "learned_mean"  # learned_mean | fixed_mean
    source_dropout: float = 0.1              # случайное «выключение» источников при обучении


@dataclass
class TrainConfig:
    batch_size: int = 512
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-5
    pos_weight: float | str | None = None    # None | "auto" | число
    patience: int = 6                        # early stopping по Джини на val
    scheduler_patience: int = 2
    calibration: str = "none"                # none | platt
    device: str = "auto"
    num_threads: int | None = None


@dataclass
class BaselinesConfig:
    boosting: bool = True    # бустинг на векторе скоров и на эмбеддингах (≈ +20 с на синтетике)
    # бустинг только на одном источнике: {имя источника: название модели в отчёте}
    single_source: dict[str, str] = field(default_factory=dict)
    # на каких данных его обучать: train | domain_fit+train (столько же, сколько у GRU по фолдам)
    single_source_data: str = "train"


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    sources: list[SourceConfig] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)

    @property
    def enabled_sources(self) -> list[SourceConfig]:
        return [s for s in self.sources if s.enabled]

    def validate(self) -> None:
        names = [s.name for s in self.sources]
        if len(names) != len(set(names)):
            raise ValueError(f"Имена источников должны быть уникальны: {names}")
        if not self.enabled_sources:
            raise ValueError("Нужен хотя бы один включённый источник")
        if sum(s.type == "tabular" for s in self.enabled_sources) > 1:
            raise ValueError("Поддерживается не более одного табличного источника")
        for s in self.sources:
            if s.type not in {"embedding", "tabular"}:
                raise ValueError(f"{s.name}: неизвестный type={s.type}")
            if s.dim <= 0 or not s.tower.hidden:
                raise ValueError(f"{s.name}: некорректные dim/tower.hidden")
        if self.model.missing_embedding not in {"learned_mean", "fixed_mean"}:
            raise ValueError("model.missing_embedding: learned_mean | fixed_mean")
        if self.target.train_on not in {"main", "short"}:
            raise ValueError("target.train_on: main | short")
        if self.split.scheme not in {"time", "random"}:
            raise ValueError("split.scheme: time | random")
        if self.baselines.single_source_data not in {"train", "domain_fit+train"}:
            raise ValueError("baselines.single_source_data: train | domain_fit+train")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)


def _from_dict(cls: type, data: dict[str, Any] | None) -> Any:
    """Рекурсивно собирает dataclass из словаря; неизвестные ключи -> ошибка."""
    data = dict(data or {})
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: неизвестные параметры {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        tp = hints[name]
        if dataclasses.is_dataclass(tp) and isinstance(value, dict):
            kwargs[name] = _from_dict(tp, value)
        elif name == "sources" and cls is Config:
            kwargs[name] = [_from_dict(SourceConfig, s) for s in value]
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> Config:
    cfg = _from_dict(Config, data)
    cfg.validate()
    return cfg


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Слияние конфигов. Список ``sources`` сливается по имени источника:
    совпадающие имена переопределяются, новые — добавляются в конец."""
    out = dict(base)
    for k, v in over.items():
        if k == "sources" and isinstance(v, list):
            merged = {s["name"]: dict(s) for s in base.get("sources", [])}
            for s in v:
                merged[s["name"]] = _deep_merge(merged.get(s["name"], {}), s)
            out[k] = list(merged.values())
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    if base:
        return _deep_merge(_read_yaml((path.parent / base).resolve()), raw)
    return raw


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Переопределения вида ``train.lr=0.0005`` или ``sources.graph.enabled=false``."""
    for item in overrides:
        key, _, value = item.partition("=")
        parts = key.split(".")
        node: Any = raw
        for p in parts[:-1]:
            if isinstance(node, list):  # sources.<name>
                node = next(s for s in node if s["name"] == p)
            else:
                node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    raw = _read_yaml(Path(path))
    return config_from_dict(apply_overrides(raw, overrides or []))
