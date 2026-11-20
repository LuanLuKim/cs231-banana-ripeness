"""SVM tuyến tính / RBF và KNN trên histogram màu (phần của Luân).

Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.
Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Trans. Inf. Theory.

Ví dụ::

    m = ColorHistogramSVM(kernel="rbf", space="hsv", use_mask=True, scheme="external")
    m.fit(train_df)
    m.evaluate(test_df)["f1_macro"]
"""
from __future__ import annotations

from typing import Any

from ..features.color_histogram import ColorHistogramExtractor
from .base import ImageFeatureModel

LINEAR_GRID = {"C": [0.01, 0.1, 1, 10, 100]}
RBF_GRID = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1, 1]}
KNN_GRID = {"n_neighbors": [1, 3, 5, 7, 9, 15]}


class _ColorFeatureMixin:
    space: str
    bins: int
    stats: bool
    use_mask: bool

    def make_extractor(self):
        return ColorHistogramExtractor(space=self.space, bins=self.bins, stats=self.stats, use_mask=self.use_mask)

    def feature_params(self) -> dict[str, Any]:
        return {"space": self.space, "bins": self.bins, "stats": self.stats, "use_mask": self.use_mask}

    @property
    def _feat_tag(self) -> str:
        t = f"{self.space}_{'mask' if self.use_mask else 'nomask'}_b{self.bins}"
        return t if self.stats else t + "_nostats"


class ColorHistogramSVM(_ColorFeatureMixin, ImageFeatureModel):
    name = "svm_color"

    def __init__(self, kernel: str = "rbf", space: str = "rgb", use_mask: bool = True, bins: int = 16,
                 stats: bool = True, class_weight: str | None = "balanced", **kw):
        super().__init__(**kw)
        if kernel not in ("linear", "rbf"):
            raise ValueError("kernel phải là 'linear' hoặc 'rbf'")
        self.kernel, self.space, self.use_mask, self.bins, self.stats = kernel, space, use_mask, bins, stats
        self.class_weight = class_weight

    def make_estimator(self):
        from sklearn.svm import SVC
        return SVC(kernel=self.kernel, class_weight=self.class_weight, random_state=self.seed,
                   cache_size=500)

    def default_param_grid(self) -> dict[str, list]:
        return dict(LINEAR_GRID if self.kernel == "linear" else RBF_GRID)

    def feature_params(self) -> dict[str, Any]:
        return {"kernel": self.kernel, **super().feature_params(), "class_weight": self.class_weight}

    @property
    def tag(self) -> str:
        return f"{self.kernel}_{self._feat_tag}"


class ColorHistogramKNN(_ColorFeatureMixin, ImageFeatureModel):
    """KNN làm mốc (baseline) trên cùng đặc trưng màu."""
    name = "knn_color"

    def __init__(self, space: str = "rgb", use_mask: bool = True, bins: int = 16, stats: bool = True,
                 weights: str = "uniform", **kw):
        super().__init__(**kw)
        self.space, self.use_mask, self.bins, self.stats, self.weights = space, use_mask, bins, stats, weights

    def make_estimator(self):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(weights=self.weights)

    def default_param_grid(self) -> dict[str, list]:
        return dict(KNN_GRID)

    def feature_params(self) -> dict[str, Any]:
        return {**super().feature_params(), "weights": self.weights}

    @property
    def tag(self) -> str:
        return self._feat_tag


__all__ = ["ColorHistogramSVM", "ColorHistogramKNN", "LINEAR_GRID", "RBF_GRID", "KNN_GRID"]
