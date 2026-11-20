"""KHUNG — HOG + SVM.

>>> NGƯỜI PHỤ TRÁCH: Nguyên. File này chỉ là khung, Nguyên tự cài đặt phần TODO. <<<

Kế thừa ``ImageFeatureModel`` nên ĐÃ CÓ SẴN (không cần viết lại):
  trích đặc trưng + cache, StandardScaler, GridSearchCV theo fold nhóm (``cv_fold_<scheme>``),
  predict / evaluate / save / load, và lưu kết quả cùng định dạng với SVM màu của Luân
  (``run_experiment`` / ``run_grid`` -> ``results/<run_name>/result.json``) để vào bảng so sánh chung.

Nguyên chỉ cần:
  1. Cài đặt ``HogExtractor`` trong ``features/hog.py``.
  2. Chọn bộ phân loại trong ``make_estimator`` và lưới tham số trong ``default_param_grid``.
  3. Khai báo thí nghiệm trong ``configs/experiments.yaml`` (mục ``nguyen``).

Xem ``models/svm_color.py`` làm ví dụ hoàn chỉnh.
"""
from __future__ import annotations

from typing import Any

from ..features.hog import HogExtractor
from .base import ImageFeatureModel


class HogSVM(ImageFeatureModel):
    name = "hog_svm"

    def __init__(self, kernel: str = "linear", orientations: int = 9, pixels_per_cell: int = 16,
                 cells_per_block: int = 2, resize: int = 128, color_mode: str = "gray", use_mask: bool = True,
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
        # TODO(Nguyên): trả về bộ phân loại sklearn, vd. LinearSVC / SVC(kernel=self.kernel, ...)
        raise NotImplementedError("HogSVM.make_estimator chưa cài đặt (phần của Nguyên)")

    def default_param_grid(self) -> dict[str, list]:
        # TODO(Nguyên): lưới tham số cho bộ phân loại (khoá KHÔNG có tiền tố "clf__"), vd. {"C": [...]}
        raise NotImplementedError("HogSVM.default_param_grid chưa cài đặt (phần của Nguyên)")

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
