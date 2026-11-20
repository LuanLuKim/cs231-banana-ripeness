"""Hình kiểm tra chất lượng (QA) của bước tiền xử lý. Lưu vào ``<output>/reports``."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..data import read_mask, read_rgb  # noqa: E402


def _imshow(ax, path, title):
    try:
        ax.imshow(read_rgb(path))
    except Exception:
        ax.text(0.5, 0.5, "không đọc được", ha="center")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def pair_grid(pairs: pd.DataFrame, title_fn, out: Path, n: int = 8) -> Path | None:
    """Lưới 2 cột: mỗi hàng là 1 cặp ảnh (path_a, path_b)."""
    top = pairs.head(n)
    if top.empty:
        return None
    fig, axes = plt.subplots(len(top), 2, figsize=(6, 2.8 * len(top)))
    axes = np.atleast_2d(axes)
    for r, (_, row) in enumerate(top.iterrows()):
        ta, tb = title_fn(row)
        _imshow(axes[r, 0], row.path_a, ta)
        _imshow(axes[r, 1], row.path_b, tb)
    plt.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def mask_qa_grid(manifest: pd.DataFrame, root: Path, out: Path, class_names: list[str],
                 per_cell: int = 3, seed: int = 0) -> Path:
    """Mỗi hàng = 1 dataset x 1 nhãn; mỗi ô = ảnh gốc có viền mask (đỏ = ảnh fallback)."""
    import cv2

    rng = np.random.default_rng(seed)
    cells = [(ds, c) for ds in sorted(manifest.dataset.unique()) for c in class_names]
    fig, axes = plt.subplots(len(cells), per_cell, figsize=(2.2 * per_cell, 2.2 * len(cells)))
    axes = np.atleast_2d(axes)
    for r, (ds, c) in enumerate(cells):
        sub = manifest[(manifest.dataset == ds) & (manifest.label == c)]
        pick = sub.iloc[rng.permutation(len(sub))[:per_cell]] if len(sub) else sub
        for k in range(per_cell):
            ax = axes[r, k]
            ax.axis("off")
            if k >= len(pick):
                continue
            row = pick.iloc[k]
            rgb = read_rgb(root / row.image_path)
            m = read_mask(root / row.mask_path).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            color = (255, 0, 0) if row.mask_fallback else (0, 255, 0)
            vis = cv2.drawContours(rgb.copy(), cnts, -1, color, 2)
            ax.imshow(vis)
            ax.set_title(f"{ds[:12]} | {c}\nS={row.mask_area:.2f}", fontsize=7)
    plt.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out
