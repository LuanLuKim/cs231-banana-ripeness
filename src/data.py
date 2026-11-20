"""Đọc output của bước tiền xử lý. MỌI mô hình chỉ đọc dữ liệu qua module này.

Thư mục output (``cfg.output_dir``, hoặc Kaggle Dataset đã lưu) có dạng::

    <root>/
      manifest.csv        # 1 dòng / ảnh: nhãn, đường dẫn, nhóm, split của từng cách chia, fold CV
      stats.json          # mean/std RGB và class weight tính trên tập train của từng cách chia
      images/<dataset>/<label>/<image_id>.png    # ảnh RGB 224x224 (pad replicate + resize)
      masks/<dataset>/<label>/<image_id>.png     # mask tách nền 0/255, cùng kích thước
      reports/            # bảng, hình kiểm tra chất lượng

Hai cách chia (cột trong manifest):

- ``split_mixed``     : gộp 3 dataset rồi chia train/val/test (stratify theo nhãn x dataset)
- ``split_external``  : BananaImageBD -> ``external``; 2 dataset còn lại chia train/val/test

Mỗi cách chia có cột fold CV tương ứng (``cv_fold_mixed`` / ``cv_fold_external``):
số 0..K-1 cho ảnh train, -1 cho ảnh không thuộc train.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import CLASS_NAMES

SCHEMES = ("mixed", "external")
SPLITS = ("train", "val", "test", "external")


def split_col(scheme: str) -> str:
    if scheme not in SCHEMES:
        raise ValueError(f"scheme phải là một trong {SCHEMES}, nhận '{scheme}'")
    return f"split_{scheme}"


def fold_col(scheme: str) -> str:
    split_col(scheme)
    return f"cv_fold_{scheme}"


def load_manifest(root: str | Path) -> pd.DataFrame:
    """Đọc manifest và đổi đường dẫn tương đối thành tuyệt đối theo ``root``."""
    root = Path(root)
    df = pd.read_csv(root / "manifest.csv")
    for col in ("image_path", "mask_path"):
        df[col] = [str(root / p) for p in df[col]]
    return df


def find_data_root(search: str | Path | list | None = None, max_depth: int = 4) -> Path:
    """Tìm thư mục chứa ``manifest.csv`` (output preprocessing).

    Biến môi trường ``BANANA_DATA_ROOT`` (nếu đặt) được ưu tiên.
    Mặc định tìm trong ``/kaggle/input`` (Kaggle Dataset đã gắn), ``/kaggle/working``, ``/content``
    (Colab), ``/content/drive/MyDrive`` và ``outputs/``. Không đi vào ``images/``, ``masks/``.
    """
    import os
    if search is None and os.environ.get("BANANA_DATA_ROOT"):
        search = os.environ["BANANA_DATA_ROOT"]
    roots = [Path(p) for p in ([search] if isinstance(search, (str, Path)) else search)] if search else [
        Path("/kaggle/input"), Path("/kaggle/working"), Path("/content"), Path("/content/drive/MyDrive"),
        Path("outputs")]
    skip = {"images", "masks", "reports", "cache", "results", ".git"}
    for base in roots:
        if not base.exists():
            continue
        base_depth = len(base.parts)
        for dirpath, dirnames, filenames in os.walk(base):
            if "manifest.csv" in filenames and "stats.json" in filenames:
                return Path(dirpath)
            depth = len(Path(dirpath).parts) - base_depth
            dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if d not in skip)
    raise FileNotFoundError(
        "Không tìm thấy manifest.csv. Trên Kaggle: Add Input -> Kaggle Dataset chứa output preprocessing, "
        "hoặc chạy notebooks/preprocessing.ipynb trước. Hoặc truyền đường dẫn: find_data_root('/duong/dan').")


def load_stats(root: str | Path) -> dict:
    return json.loads((Path(root) / "stats.json").read_text(encoding="utf-8"))


def get_split(df: pd.DataFrame, scheme: str, split: str | list[str]) -> pd.DataFrame:
    """Lấy các dòng thuộc ``split`` (hoặc danh sách split) của cách chia ``scheme``."""
    splits = [split] if isinstance(split, str) else list(split)
    out = df[df[split_col(scheme)].isin(splits)].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Không có ảnh nào ở split={splits} của scheme='{scheme}'")
    return out


def labels_of(df: pd.DataFrame) -> np.ndarray:
    """Nhãn dạng số nguyên theo thứ tự ``CLASS_NAMES``."""
    return df["label_idx"].to_numpy()


def read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    """Mask bool (True = chuối)."""
    m = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Không đọc được mask: {path}")
    return m > 127


def apply_mask(rgb: np.ndarray, mask: np.ndarray, fill: int = 0) -> np.ndarray:
    out = rgb.copy()
    out[~mask] = fill
    return out


__all__ = ["CLASS_NAMES", "SCHEMES", "SPLITS", "split_col", "fold_col", "load_manifest", "find_data_root",
           "load_stats", "get_split", "labels_of", "read_rgb", "read_mask", "apply_mask"]
