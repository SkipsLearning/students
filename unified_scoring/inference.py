"""Инференс: PD для клиента с ЛЮБЫМ непустым набором источников.

Формат запроса (JSON-объект на клиента)::

    {
      "client_id": "42",
      "embeddings": {"card": [... 32 числа ...], "bki": [... 256 чисел ...]},
      "tabular": {"region": "REG_07", "income": 55000, ...}   # необязательно
    }

Отсутствующие источники просто не передаются — модель подставит для них
константный эмбеддинг и флаг отсутствия.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import TabularConfig
from .data.tabular import TabularPreprocessor
from .data.torch_dataset import ScoringTorchDataset
from .model import UnifiedScoringModel
from .train import PlattCalibrator, predict_logits


@dataclass
class ModelBundle:
    model: UnifiedScoringModel
    preprocessor: TabularPreprocessor | None
    tabular_source: str | None
    calibrator: PlattCalibrator | None

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model_kwargs": self.model.build_kwargs(),
                "state_dict": self.model.state_dict(),
                "tabular_source": self.tabular_source,
                "tabular_cfg": None if self.preprocessor is None else self.preprocessor.cfg.__dict__,
                "tabular_state": None if self.preprocessor is None else self.preprocessor.state_dict(),
                "calibration": None if self.calibrator is None else self.calibrator.__dict__,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        raw = torch.load(path, map_location="cpu", weights_only=False)
        model = UnifiedScoringModel.from_kwargs(raw["model_kwargs"])
        model.load_state_dict(raw["state_dict"])
        model.eval()
        pre = None
        if raw["tabular_state"] is not None:
            pre = TabularPreprocessor.from_state_dict(raw["tabular_state"], TabularConfig(**raw["tabular_cfg"]))
        cal = None if raw["calibration"] is None else PlattCalibrator(**raw["calibration"])
        return cls(model=model, preprocessor=pre, tabular_source=raw["tabular_source"], calibrator=cal)

    def logits_to_pd(self, logits: np.ndarray) -> np.ndarray:
        if self.calibrator is not None:
            return self.calibrator(logits)
        return 1.0 / (1.0 + np.exp(-logits))


class Scorer:
    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self.specs = {s.name: s for s in bundle.model.specs}

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "Scorer":
        return cls(ModelBundle.load(Path(run_dir) / "model.pt"))

    @property
    def source_names(self) -> list[str]:
        return list(self.specs)

    def _to_dataset(self, records: list[dict[str, Any]]) -> ScoringTorchDataset:
        n = len(records)
        emb_specs = [s for s in self.specs.values() if s.kind == "embedding"]
        embeddings = {s.name: np.zeros((n, s.dim), dtype=np.float32) for s in emb_specs}
        masks = {name: np.zeros(n, dtype=bool) for name in self.specs}
        tab_rows: list[dict[str, Any]] = []
        for i, rec in enumerate(records):
            for name, vec in (rec.get("embeddings") or {}).items():
                spec = self.specs.get(name)
                if spec is None or spec.kind != "embedding":
                    raise ValueError(f"Неизвестный источник-эмбеддинг '{name}'. Доступны: {self.source_names}")
                if vec is None:
                    continue
                arr = np.asarray(vec, dtype=np.float32)
                if arr.shape != (spec.dim,):
                    raise ValueError(f"{name}: ожидается вектор длины {spec.dim}, получено {arr.shape}")
                embeddings[name][i] = arr
                masks[name][i] = True
            tab = rec.get("tabular")
            if tab is not None and self.bundle.tabular_source is not None:
                masks[self.bundle.tabular_source][i] = True
            tab_rows.append(tab or {})
            if not any(m[i] for m in masks.values()):
                raise ValueError(f"Запись {rec.get('client_id', i)}: нет ни одного источника")
        cat = num = None
        if self.bundle.preprocessor is not None:
            cat, num = self.bundle.preprocessor.transform(pd.DataFrame(tab_rows, index=range(n)))
        return ScoringTorchDataset(embeddings, masks, cat, num, None)

    def predict(self, records: list[dict[str, Any]]) -> np.ndarray:
        """PD ∈ [0, 1] для каждой записи."""
        ds = self._to_dataset(records)
        logits = predict_logits(self.bundle.model, ds, torch.device("cpu"))
        return self.bundle.logits_to_pd(logits)
