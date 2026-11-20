"""KHUNG — ResNet18 (transfer learning) cho 4 mức độ chín.

>>> NGƯỜI PHỤ TRÁCH: Quỳnh. File này chỉ là khung, Quỳnh tự cài đặt phần TODO. <<<

Tài liệu: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.

Kế thừa ``BaseRipenessModel`` nên chạy qua ``run_experiment`` / ``run_grid`` như SVM của Luân, và kết quả
(``results/<run_name>/result.json``, ``pred_<split>.csv``, ``cm_<split>.png``) tự vào bảng so sánh chung.

Hợp đồng cần cài đặt:

- ``fit(train_df, val_df)``  : huấn luyện, trả về ``self``. ``val_df`` dùng cho early stopping
  (theo val macro-F1, gợi ý dùng ``evaluation.compute_metrics``). Ghi vào ``self.fit_info`` (dict, lưu
  vào result.json), nên có: ``history`` (list dict mỗi epoch), ``best_epoch``, ``best_val_f1_macro``.
- ``predict_proba(df)``      : np.ndarray (len(df), 4), thứ tự lớp = ``self.class_names``.
  (``predict`` = argmax đã có sẵn ở lớp cha.)
- ``save(path)`` / ``load(path)`` : lưu / nạp (``path`` có đuôi ``.pt``).

Đã có sẵn trong khung:

- ``set_data_root`` đọc ``stats.json`` -> ``self.train_mean_std()`` (mean/std RGB của train) và
  ``self.train_class_weight()`` (trọng số lớp), đúng theo ``scheme`` đang chạy.
- ``load_image(image_path, mask_path, use_mask)`` đọc ảnh RGB uint8 (tô đen nền nếu dùng mask).
- Cột DataFrame: ``image_path``, ``mask_path``, ``label_idx`` (0..3), xem ``data.py``.
- Độ lệch chuẩn: ``run_grid(..., seeds=[42, 43, 44])`` chạy nhiều seed, bảng chung tự tính std giữa các seed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data import labels_of, load_stats, read_mask, read_rgb
from .base import BaseRipenessModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(img_path: str, mask_path: str | None, use_mask: bool) -> np.ndarray:
    """Ảnh RGB uint8 (224, 224, 3); nếu ``use_mask`` thì tô đen phần nền (ngoài mask Otsu)."""
    rgb = read_rgb(img_path)
    if use_mask and mask_path:
        m = read_mask(mask_path)
        if m.any():
            rgb = rgb.copy()
            rgb[~m] = 0
    return rgb


class ResNet18Classifier(BaseRipenessModel):
    name = "resnet18"

    def __init__(self, pretrained: bool = True, epochs: int = 20, batch_size: int = 32, lr: float = 3e-4,
                 weight_decay: float = 1e-4, patience: int = 5, use_mask: bool = False, img_size: int = 224,
                 num_workers: int = 2, device: str | None = None, verbose: bool = True,
                 scheme: str = "mixed", seed: int = 42):
        # TODO(Quỳnh): thêm/bớt tham số (augmentation, freeze backbone, ...). Nhớ cập nhật get_params + tag.
        super().__init__(scheme, seed)
        self.pretrained, self.epochs, self.batch_size, self.lr = pretrained, epochs, batch_size, lr
        self.weight_decay, self.patience, self.use_mask, self.img_size = weight_decay, patience, use_mask, img_size
        self.num_workers, self.device, self.verbose = num_workers, device, verbose
        self.stats: dict | None = None
        self.net = None

    # ---- có sẵn ---------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        """Tham số ghi vào bảng kết quả."""
        return {k: getattr(self, k) for k in ("pretrained", "epochs", "batch_size", "lr", "weight_decay",
                                               "patience", "use_mask", "img_size", "scheme", "seed")}

    @property
    def tag(self) -> str:
        """Tên ngắn của cấu hình, dùng đặt tên thư mục kết quả (phải khác nhau giữa các cấu hình)."""
        return f"{'pre' if self.pretrained else 'scratch'}_{'mask' if self.use_mask else 'nomask'}_e{self.epochs}"

    @property
    def file_suffix(self) -> str:
        return ".pt"

    def set_data_root(self, root: str | Path | None) -> None:
        """``run_experiment`` tự gọi: đọc ``stats.json`` của output preprocessing (nếu có)."""
        if root is None:
            return
        try:
            self.stats = load_stats(root)
        except FileNotFoundError:
            self.stats = None

    def _scheme_stats(self) -> dict:
        return ((self.stats or {}).get("schemes") or {}).get(self.scheme, {})

    def train_mean_std(self) -> tuple[list[float], list[float]]:
        """mean/std RGB (thang [0, 1]) của tập train theo scheme; không có stats.json -> ImageNet."""
        st = self._scheme_stats()
        if st.get("mean_rgb"):
            return list(st["mean_rgb"]), list(st["std_rgb"])
        return list(IMAGENET_MEAN), list(IMAGENET_STD)

    def train_class_weight(self, train_df: pd.DataFrame | None = None) -> list[float]:
        """Trọng số lớp theo thứ tự ``class_names`` (từ stats.json, hoặc tính từ ``train_df``)."""
        cw = self._scheme_stats().get("class_weight_train")
        if cw:
            return [float(cw[c]) for c in self.class_names]
        if train_df is None:
            return [1.0] * len(self.class_names)
        cnt = np.bincount(labels_of(train_df), minlength=len(self.class_names)).astype(float)
        return (cnt.sum() / (len(cnt) * np.maximum(cnt, 1))).tolist()

    # ---- TODO(Quỳnh) ----------------------------------------------------
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "ResNet18Classifier":
        # TODO(Quỳnh): torchvision.models.resnet18, thay lớp fc -> 4 lớp, train + early stopping theo val macro-F1,
        #              ghi self.fit_info["history"], ["best_epoch"], ["best_val_f1_macro"]
        raise NotImplementedError("ResNet18Classifier.fit chưa cài đặt (phần của Quỳnh)")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # TODO(Quỳnh): softmax (len(df), 4)
        raise NotImplementedError("ResNet18Classifier.predict_proba chưa cài đặt (phần của Quỳnh)")

    def save(self, path: str | Path) -> Path:
        # TODO(Quỳnh): torch.save({"params": self.get_params(), "state_dict": ..., "fit_info": self.fit_info}, path)
        raise NotImplementedError("ResNet18Classifier.save chưa cài đặt (phần của Quỳnh)")

    @classmethod
    def load(cls, path: str | Path) -> "ResNet18Classifier":
        # TODO(Quỳnh): đọc file .pt, dựng lại mô hình bằng cls(**params) rồi nạp state_dict
        raise NotImplementedError("ResNet18Classifier.load chưa cài đặt (phần của Quỳnh)")


__all__ = ["ResNet18Classifier", "load_image", "IMAGENET_MEAN", "IMAGENET_STD"]
