"""Bước 1-2: thu thập ảnh, chuẩn hoá nhãn, đọc mỗi ảnh 1 lần để lấy MD5 + dHash + thumbnail."""
from __future__ import annotations

import concurrent.futures
import hashlib
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import Config

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_):
        return x


def collect_images(cfg: Config) -> tuple[pd.DataFrame, Counter]:
    """Quét các thư mục dataset, nhãn = tên thư mục cha (sau khi qua ``label_mapping``).

    Bỏ qua thư mục augmentation/synthetic/varying illumination và thư mục có tên không có
    trong ``label_mapping`` (ví dụ thư mục giống chuối: cavendish, ambon, ...).
    """
    excluded = [x.lower() for x in cfg.excluded_dir_names]
    exts = {e.lower() for e in cfg.image_extensions}
    rows, skipped = [], Counter()
    for ds, root in cfg.dataset_dirs.items():
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"{ds}: không tìm thấy {root}")
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            rel_parts = [x.lower() for x in p.relative_to(root).parts[:-1]]
            if any(ex in part for part in rel_parts for ex in excluded):
                skipped[(ds, "thư mục bị loại")] += 1
                continue
            raw = p.parent.name.lower().strip()
            if raw not in cfg.label_mapping:
                skipped[(ds, f"nhãn lạ: {raw}")] += 1
                continue
            rows.append({
                "dataset": ds,
                "src_path": str(p),
                "raw_label": raw,
                "label": cfg.label_mapping[raw],
                "src_folder": str(p.parent.parent.relative_to(root)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Không thu thập được ảnh nào, kiểm tra lại dataset_dirs")
    return df, skipped


def compare_with_eda(df: pd.DataFrame, eda_counts: dict[str, int]) -> pd.DataFrame:
    out = []
    for ds, n_eda in eda_counts.items():
        n = int((df.dataset == ds).sum())
        out.append({"dataset": ds, "preprocessing": n, "eda": n_eda, "lech": n - n_eda})
    return pd.DataFrame(out)


def _dhash(gray: np.ndarray, size: int) -> np.ndarray:
    g = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def _read_one(args: tuple[str, int]):
    path, hash_size = args
    try:
        data = Path(path).read_bytes()
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        thumb = cv2.resize(rgb, (16, 16), interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        return (hashlib.md5(data).hexdigest(), _dhash(gray, hash_size), _dhash(gray, 8), thumb, w, h)
    except Exception:
        return None


def read_fingerprints(df: pd.DataFrame, cfg: Config):
    """Trả về (df đã lọc ảnh lỗi, H [n, n_bits], H64 [n, 64], T [n,16,16,3], n_lỗi)."""
    workers = cfg.num_workers or os.cpu_count() or 1
    args = [(p, cfg.hash_size) for p in df["src_path"]]
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(tqdm(ex.map(_read_one, args, chunksize=32), total=len(args), desc="Đọc ảnh"))
    else:
        res = [_read_one(a) for a in tqdm(args, desc="Đọc ảnh")]
    ok = np.array([r is not None for r in res])
    df = df[ok].reset_index(drop=True)
    res = [r for r in res if r is not None]
    df["md5"] = [r[0] for r in res]
    df["w"] = [r[4] for r in res]
    df["h"] = [r[5] for r in res]
    H = np.stack([r[1] for r in res])
    H64 = np.stack([r[2] for r in res])
    T = np.stack([r[3] for r in res])
    return df, H, H64, T, int((~ok).sum())
