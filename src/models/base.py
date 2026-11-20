"""Wrapper chung cho MỌI mô hình: ``fit / predict / evaluate / save / load``.

Dữ liệu vào luôn là DataFrame của manifest (``data.load_manifest`` + ``data.get_split``),
nên SVM, HOG và ResNet18 dùng chung một cách chia và chung hàm đánh giá.

Hai lớp cơ sở:

- ``BaseRipenessModel``  : giao diện chung (ResNet18 kế thừa trực tiếp).
- ``ImageFeatureModel``  : đặc trưng thủ công + ``Pipeline[StandardScaler, clf]`` của sklearn.
  Tham số chọn bằng ``GridSearchCV(cv=PredefinedSplit(cv_fold_<scheme>), scoring="f1_macro")``
  -> chọn **trong tập train, theo nhóm ảnh gần trùng** (không nhìn val/test).
  Độ lệch chuẩn CV lấy từ 5 fold ở bộ tham số tốt nhất.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import CLASS_NAMES
from ..data import fold_col, labels_of
from ..evaluation import MAIN_METRICS, compute_metrics


class BaseRipenessModel(ABC):
    """Giao diện chung.

    Lớp con cần định nghĩa:
      - ``name``        : tên họ mô hình (khoá trong registry, vd. ``"svm_color"``)
      - ``fit``, ``predict`` (hoặc ``predict_proba``), ``save``, ``load``
      - ``get_params``  : dict tham số cấu hình (ghi vào bảng kết quả)
      - ``tag``         : chuỗi ngắn mô tả cấu hình, dùng đặt tên lần chạy
    """
    name: str = "base"

    def __init__(self, scheme: str = "mixed", seed: int = 42):
        fold_col(scheme)            # kiểm tra scheme hợp lệ
        self.scheme = scheme
        self.seed = seed
        self.class_names = list(CLASS_NAMES)
        self.fit_info: dict[str, Any] = {}

    # ---- bắt buộc --------------------------------------------------------
    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "BaseRipenessModel":
        ...

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseRipenessModel":
        ...

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        ...

    # ---- mặc định --------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray | None:
        """Xác suất (n, 4) nếu mô hình hỗ trợ, ngược lại ``None``."""
        return None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(df)
        if p is None:
            raise NotImplementedError(f"{type(self).__name__} cần cài đặt predict hoặc predict_proba")
        return p.argmax(1)

    def evaluate(self, df: pd.DataFrame) -> dict:
        return compute_metrics(labels_of(df), self.predict(df), self.class_names)

    def set_data_root(self, root: str | Path | None) -> None:
        """Cho mô hình biết thư mục output preprocessing (vd. để đọc ``stats.json``). Mặc định: bỏ qua."""

    @property
    def tag(self) -> str:
        return "default"

    @property
    def file_suffix(self) -> str:
        return ".joblib"

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({ps})"


# ---------------------------------------------------------------------------
class ImageFeatureModel(BaseRipenessModel):
    """Đặc trưng thủ công + StandardScaler + bộ phân loại sklearn, chọn tham số bằng CV theo nhóm.

    Lớp con định nghĩa ``make_extractor()``, ``make_estimator()`` và ``default_param_grid()``
    (khoá KHÔNG có tiền tố ``clf__``; lớp này tự thêm).
    """

    def __init__(self, scheme: str = "mixed", seed: int = 42, param_grid: dict | None = None,
                 n_jobs: int = -1, cache_dir: str | Path | None = None, verbose: bool = True):
        super().__init__(scheme, seed)
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.pipeline_ = None
        self.search_ = None

    # ---- lớp con cài đặt ------------------------------------------------
    @abstractmethod
    def make_extractor(self):
        ...

    @abstractmethod
    def make_estimator(self):
        ...

    @abstractmethod
    def default_param_grid(self) -> dict[str, list]:
        ...

    def feature_params(self) -> dict[str, Any]:
        """Tham số của bộ trích đặc trưng (để ghi bảng kết quả)."""
        return {}

    # ---- dùng chung -----------------------------------------------------
    @property
    def extractor(self):
        return self.make_extractor()

    def set_data_root(self, root: str | Path | None) -> None:
        self.data_root_ = root

    def features(self, df: pd.DataFrame) -> np.ndarray:
        from ..features.cache import data_fingerprint, default_cache_dir, extract_features
        cache = self.cache_dir
        if cache is not False:
            base = Path(cache) if cache else default_cache_dir()
            cache = base / data_fingerprint(getattr(self, "data_root_", None))
        return extract_features(self.extractor, df, cache_dir=cache, n_jobs=self.n_jobs, verbose=self.verbose)

    def feature_names(self) -> list[str]:
        return self.extractor.names()

    def get_params(self) -> dict[str, Any]:
        return {**self.feature_params(), "scheme": self.scheme, "seed": self.seed,
                "param_grid": self.param_grid or self.default_param_grid()}

    def _cv_split(self, train_df: pd.DataFrame):
        from sklearn.model_selection import PredefinedSplit, StratifiedKFold
        col = fold_col(self.scheme)
        if col in train_df and (train_df[col] >= 0).all() and train_df[col].nunique() >= 2:
            return PredefinedSplit(train_df[col].to_numpy())
        # dữ liệu không có cột fold (vd. tự tạo DataFrame) -> CV phân tầng thường, ghi rõ
        self.fit_info["cv_warning"] = f"Không có {col} hợp lệ, dùng StratifiedKFold(5) (KHÔNG theo nhóm)"
        return StratifiedKFold(5, shuffle=True, random_state=self.seed)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "ImageFeatureModel":
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        t0 = time.time()
        X, y = self.features(train_df), labels_of(train_df)
        t_feat = time.time() - t0
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", self.make_estimator())])
        grid = {f"clf__{k}": v for k, v in (self.param_grid or self.default_param_grid()).items()}
        from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
        scoring = {"accuracy": "accuracy",
                   "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
                   "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
                   "f1_macro": make_scorer(f1_score, average="macro", zero_division=0)}
        self.search_ = GridSearchCV(pipe, grid, scoring=scoring, refit="f1_macro", cv=self._cv_split(train_df),
                                    n_jobs=self.n_jobs, error_score="raise")
        self.search_.fit(X, y)
        self.pipeline_ = self.search_.best_estimator_
        cv = self.search_.cv_results_
        b = self.search_.best_index_
        self.fit_info.update({
            "best_params": {k.removeprefix("clf__"): _jsonable(v) for k, v in self.search_.best_params_.items()},
            "n_features": int(X.shape[1]),
            "n_train": int(len(y)),
            "n_candidates": int(len(cv["params"])),
            "feature_time_s": round(t_feat, 2),
            "fit_time_s": round(time.time() - t0, 2),
        })
        for m in MAIN_METRICS:
            self.fit_info[f"cv_{m}_mean"] = float(cv[f"mean_test_{m}"][b])
            self.fit_info[f"cv_{m}_std"] = float(cv[f"std_test_{m}"][b])
        if self.verbose:
            print(f"[{self.name}] tham số tốt nhất: {self.fit_info['best_params']} | "
                  f"CV macro-F1 = {self.fit_info['cv_f1_macro_mean']:.3f} ± {self.fit_info['cv_f1_macro_std']:.3f}")
        return self

    def cv_results(self) -> pd.DataFrame:
        """Bảng toàn bộ lưới tham số (mean/std từng fold) để vẽ / đưa vào phụ lục."""
        if self.search_ is None:
            raise RuntimeError("Chưa fit")
        cv = pd.DataFrame(self.search_.cv_results_)
        keep = [c for c in cv if c.startswith("param_") or c.startswith("mean_test") or c.startswith("std_test")
                or c == "rank_test_f1_macro"]
        return cv[keep].sort_values("rank_test_f1_macro").reset_index(drop=True)

    def _check(self):
        if self.pipeline_ is None:
            raise RuntimeError("Mô hình chưa được fit / load")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._check()
        return self.pipeline_.predict(self.features(df))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray | None:
        self._check()
        if hasattr(self.pipeline_, "predict_proba"):
            try:
                return self.pipeline_.predict_proba(self.features(df))
            except AttributeError:      # SVC(probability=False)
                return None
        return None

    def save(self, path: str | Path) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        search, self.search_ = self.search_, None      # cv_results_ đã nằm trong fit_info, bỏ cho nhẹ
        try:
            joblib.dump(self, path)
        finally:
            self.search_ = search
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ImageFeatureModel":
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, ImageFeatureModel):
            raise TypeError(f"{path} không phải ImageFeatureModel")
        return obj


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, tuple):
        return list(v)
    return v


__all__ = ["BaseRipenessModel", "ImageFeatureModel"]
