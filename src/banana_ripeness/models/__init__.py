"""Mô hình. Mọi mô hình kế thừa ``BaseRipenessModel`` và được tạo qua ``build_model(name, **params)``.

========== ============================== ===============================
name       lớp                            người phụ trách
========== ============================== ===============================
svm_color  ColorHistogramSVM              Luân
knn_color  ColorHistogramKNN (mốc)        Luân
hog_svm    HogSVM                         Nguyên
resnet18   ResNet18Classifier (+Grad-CAM) Quỳnh
========== ============================== ===============================

Import lười (lazy) để người chỉ chạy SVM không cần torch.
"""
from __future__ import annotations

import importlib
from pathlib import Path

from .base import BaseRipenessModel, ImageFeatureModel

REGISTRY: dict[str, tuple[str, str]] = {
    "svm_color": (".svm_color", "ColorHistogramSVM"),
    "knn_color": (".svm_color", "ColorHistogramKNN"),
    "hog_svm": (".hog_svm", "HogSVM"),
    "resnet18": (".resnet18", "ResNet18Classifier"),
}


def get_model_class(name: str) -> type[BaseRipenessModel]:
    if name not in REGISTRY:
        raise KeyError(f"Không có mô hình '{name}'. Có: {sorted(REGISTRY)}")
    mod, cls = REGISTRY[name]
    return getattr(importlib.import_module(mod, __name__), cls)


def build_model(name: str, **params) -> BaseRipenessModel:
    """``build_model("svm_color", kernel="rbf", space="hsv", scheme="external")``."""
    return get_model_class(name)(**params)


def load_model(path: str | Path) -> BaseRipenessModel:
    """Nạp mô hình từ file (``model.joblib`` / ``model.pt``) hoặc thư mục lần chạy (có ``result.json``)."""
    import json
    path = Path(path)
    if path.is_dir():
        rec = json.loads((path / "result.json").read_text(encoding="utf-8"))
        cls = get_model_class(rec["model"])
        files = sorted(path.glob("model.*"))
        if not files:
            raise FileNotFoundError(f"{path} không có file model.* (lần chạy với save_model=False?)")
        return cls.load(files[0])
    if path.suffix == ".pt":
        return get_model_class("resnet18").load(path)
    return ImageFeatureModel.load(path)


__all__ = ["BaseRipenessModel", "ImageFeatureModel", "REGISTRY", "build_model", "get_model_class", "load_model"]
