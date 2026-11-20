"""Trích đặc trưng cho cả DataFrame manifest, có cache theo ``image_id``.

Cache: ``<cache_dir>/<extractor.key>.npz`` chứa ``ids`` và ``X``. Lần sau chỉ tính ảnh chưa có.
Các thí nghiệm dùng chung một bộ đặc trưng (vd. cùng RGB + mask, khác kernel SVM) sẽ không tính lại.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import read_mask, read_rgb

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_):
        return x


def default_cache_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/cache/features")
    return Path("outputs/cache/features")


def data_fingerprint(root: str | Path | None) -> str:
    """Mã ngắn của 1 bộ output preprocessing (theo ``config_used.yaml``), để cache của 2 lần
    preprocessing khác tham số (vd. đổi cách tách nền) không lẫn vào nhau."""
    import hashlib
    if root is None:
        return "unknown"
    p = Path(root) / "config_used.yaml"
    if not p.exists():
        return "unknown"
    return hashlib.md5(p.read_bytes()).hexdigest()[:8]


def _extract_chunk(extractor, items):
    out = []
    for img_path, mask_path in items:
        rgb = read_rgb(img_path)
        mask = read_mask(mask_path) if (extractor.needs_mask and mask_path) else None
        out.append(extractor(rgb, mask))
    return out


def _compute(extractor, df: pd.DataFrame, n_jobs: int, verbose: bool) -> np.ndarray:
    mask_paths = df["mask_path"] if "mask_path" in df else [None] * len(df)
    items = list(zip(df["image_path"], mask_paths))
    if not items:
        return np.zeros((0, len(extractor.names())), np.float32)
    n_jobs = (os.cpu_count() or 1) if n_jobs in (None, -1) else n_jobs
    size = 64
    chunks = [items[i:i + size] for i in range(0, len(items), size)]
    if n_jobs > 1 and len(chunks) > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(delayed(_extract_chunk)(extractor, c)
                                      for c in tqdm(chunks, desc=f"Đặc trưng {extractor.key}", disable=not verbose))
    else:
        res = [_extract_chunk(extractor, c) for c in tqdm(chunks, desc=f"Đặc trưng {extractor.key}",
                                                          disable=not verbose)]
    return np.stack([f for chunk in res for f in chunk]).astype(np.float32)


def extract_features(extractor, df: pd.DataFrame, cache_dir: str | Path | None = None,
                     n_jobs: int | None = -1, verbose: bool = True) -> np.ndarray:
    """Ma trận đặc trưng (len(df), d) theo đúng thứ tự dòng của ``df``.

    ``cache_dir=False`` để tắt cache.
    """
    if cache_dir is False:
        return _compute(extractor, df, n_jobs, verbose)
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{extractor.key}.npz"

    known: dict[str, np.ndarray] = {}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        known = dict(zip(z["ids"].tolist(), z["X"]))
    ids = df["image_id"].astype(str).tolist()
    missing = [i for i, x in enumerate(ids) if x not in known]
    if missing:
        X_new = _compute(extractor, df.iloc[missing], n_jobs, verbose)
        for i, f in zip(missing, X_new):
            known[ids[i]] = f
        all_ids = np.array(list(known.keys()))
        np.savez(path, ids=all_ids, X=np.stack([known[k] for k in all_ids]))
    return np.stack([known[x] for x in ids]).astype(np.float32)


__all__ = ["extract_features", "default_cache_dir", "data_fingerprint"]
