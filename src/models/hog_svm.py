"""HOG + SVM (tuyến tính / RBF).

Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.
Đặc trưng: ``features/hog.py``. Trích đặc trưng + cache, StandardScaler, GridSearchCV theo fold nhóm, đánh giá và
lưu kết quả do ``ImageFeatureModel`` (``base.py``) lo.
"""
from __future__ import annotations

from typing import Any

from ..features.hog import HogExtractor
from .base import ImageFeatureModel


class HogSVM(ImageFeatureModel):
    name = "hog_svm"

    def __init__(self, kernel: str = "linear", orientations: int = 9, pixels_per_cell: int = 16,
                 cells_per_block: int = 2, resize: int = 128, color_mode: str = "rgb", use_mask: bool = True,
                 class_weight: str | None = "balanced", **kw):
        # kw: scheme, seed, param_grid, n_jobs, cache_dir, verbose (xem ImageFeatureModel)
        super().__init__(**kw)
        self.kernel = kernel
        self.orientations, self.pixels_per_cell, self.cells_per_block = orientations, pixels_per_cell, cells_per_block
        self.resize, self.color_mode, self.use_mask = resize, color_mode, use_mask
        self.class_weight = class_weight

    def make_extractor(self):
        return HogExtractor(orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                            cells_per_block=self.cells_per_block, resize=self.resize,
                            color_mode=self.color_mode, use_mask=self.use_mask)

    def make_estimator(self):
        from sklearn.svm import SVC
        return SVC(kernel=self.kernel, class_weight=self.class_weight, random_state=self.seed, cache_size=500)

    def default_param_grid(self) -> dict[str, list]:
        if self.kernel == "linear":
            return {"C": [0.01, 0.1, 1, 10]}
        return {"C": [1, 10, 100], "gamma": ["scale", 1e-4, 1e-3]}

    def feature_params(self) -> dict[str, Any]:
        """Tham số ghi vào bảng kết quả."""
        return {"kernel": self.kernel, "orientations": self.orientations, "pixels_per_cell": self.pixels_per_cell,
                "cells_per_block": self.cells_per_block, "resize": self.resize, "color_mode": self.color_mode,
                "use_mask": self.use_mask, "class_weight": self.class_weight}

    @property
    def tag(self) -> str:
        """Tên ngắn của cấu hình, dùng đặt tên thư mục kết quả (phải khác nhau giữa các cấu hình)."""
        return (f"{self.kernel}_{self.color_mode}_{'mask' if self.use_mask else 'nomask'}"
                f"_o{self.orientations}_p{self.pixels_per_cell}")


__all__ = ["HogSVM"]
