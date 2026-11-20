"""Bước 5: chia train / val / test theo NHÓM ảnh gần giống, stratify theo (nhãn, dataset).

Hai cách chia, lưu thành 2 cột trong manifest:

- ``mixed``    : gộp 3 dataset, test ~1/7, val ~1/7, train ~5/7.
- ``external`` : toàn bộ ``cfg.external_dataset`` (BananaImageBD) -> ``external``.
                 Nhóm nào có ảnh của BananaImageBD cũng đưa cả nhóm sang ``external`` để không rò rỉ.
                 Phần còn lại chia train/val/test như trên.

Lưu ý khi báo cáo: "chia theo nhóm" ở đây là nhóm ảnh gần trùng theo dHash, KHÔNG phải
theo ID từng quả chuối (các dataset không cung cấp ID quả).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from ..config import Config


def _take_fold(pool: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # lớp ít mẫu hơn n_splits
        tr, te = next(sgkf.split(np.zeros(len(pool)), y[pool], groups[pool]))
    return pool[tr], pool[te]


def _strata(df: pd.DataFrame) -> np.ndarray:
    return (df.label + "|" + df.dataset).to_numpy()


def split_pool(df: pd.DataFrame, pool: np.ndarray, cfg: Config) -> pd.Series:
    """Chia các chỉ số ``pool`` thành train/val/test. Trả về Series cùng index df ("" = ngoài pool)."""
    y, groups = _strata(df), df["group"].to_numpy()
    rest, test = _take_fold(pool, y, groups, cfg.n_test_folds, cfg.seed)
    train, val = _take_fold(rest, y, groups, cfg.n_val_folds, cfg.seed)
    out = pd.Series("", index=df.index, dtype=object)
    out.iloc[train], out.iloc[val], out.iloc[test] = "train", "val", "test"
    return out


def split_mixed(df: pd.DataFrame, cfg: Config) -> pd.Series:
    return split_pool(df, np.arange(len(df)), cfg)


def split_external(df: pd.DataFrame, cfg: Config) -> pd.Series:
    ext = cfg.external_dataset
    if ext not in set(df.dataset):
        raise ValueError(f"external_dataset='{ext}' không có trong dữ liệu")
    ext_groups = set(df.loc[df.dataset == ext, "group"])
    is_ext = df["group"].isin(ext_groups).to_numpy()
    out = split_pool(df, np.flatnonzero(~is_ext), cfg)
    out.iloc[np.flatnonzero(is_ext)] = "external"
    return out


def assign_cv_folds(df: pd.DataFrame, split: pd.Series, cfg: Config) -> np.ndarray:
    """Fold 0..K-1 cho ảnh train (theo nhóm, stratify nhãn x dataset), -1 cho ảnh khác."""
    folds = np.full(len(df), -1, dtype=int)
    train = np.flatnonzero((split == "train").to_numpy())
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_cv_folds, shuffle=True, random_state=cfg.seed)
    y, groups = _strata(df), df["group"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for k, (_, te) in enumerate(sgkf.split(np.zeros(len(train)), y[train], groups[train])):
            folds[train[te]] = k
    return folds


def leakage_report(df: pd.DataFrame, split: pd.Series, H: np.ndarray, cfg: Config,
                   scheme: str) -> pd.DataFrame:
    """Kiểm chứng không rò rỉ: mỗi nhóm/MD5 chỉ ở 1 split, và khoảng cách dHash nhỏ nhất
    từ ảnh val/test/external tới train phải > ``group_thr``."""
    tmp = pd.DataFrame({"group": df.group, "md5": df.md5, "split": split})
    assert (tmp.groupby("group")["split"].nunique() == 1).all(), f"[{scheme}] nhóm nằm ở 2 split!"
    assert (tmp.groupby("md5")["split"].nunique() == 1).all(), f"[{scheme}] MD5 nằm ở 2 split!"
    train = np.flatnonzero((split == "train").to_numpy())
    nn = NearestNeighbors(n_neighbors=1, metric="hamming", algorithm="brute").fit(H[train])
    rows = []
    for name in ("val", "test", "external"):
        idx = np.flatnonzero((split == name).to_numpy())
        if not len(idx):
            continue
        d = nn.kneighbors(H[idx])[0][:, 0] * H.shape[1]
        rows.append({"scheme": scheme, "split": name, "n": len(idx), "min_bit_toi_train": int(d.min()),
                     "p5": float(np.percentile(d, 5)), "median": float(np.median(d))})
        assert d.min() > cfg.group_thr, f"[{scheme}] {name} còn ảnh gần giống train"
    return pd.DataFrame(rows)
