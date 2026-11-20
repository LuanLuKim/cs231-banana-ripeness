"""Độ đo, độ lệch chuẩn bootstrap, ma trận nhầm lẫn và bảng kết quả.

Mọi mô hình (SVM / KNN / HOG / ResNet18) đều đánh giá bằng CÙNG các hàm ở đây, nên kết quả
của 3 người so sánh trực tiếp được.

Độ đo chính (macro = trung bình không trọng số trên 4 lớp, không thiên về lớp đông):
accuracy, precision_macro, recall_macro, f1_macro.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import CLASS_NAMES

MAIN_METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


# ---------------------------------------------------------------------------
def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Ma trận nhầm lẫn: hàng = nhãn thật, cột = nhãn dự đoán."""
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    return np.bincount(y_true * n_classes + y_pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    """accuracy + precision/recall/F1 macro. Lớp không có mẫu thật nào bị bỏ khỏi trung bình macro."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    support, predicted = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(predicted > 0, tp / predicted, 0.0)
        rec = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = support > 0
    n = cm.sum()
    return {
        "accuracy": float(tp.sum() / n) if n else float("nan"),
        "precision_macro": float(prec[present].mean()) if present.any() else float("nan"),
        "recall_macro": float(rec[present].mean()) if present.any() else float("nan"),
        "f1_macro": float(f1[present].mean()) if present.any() else float("nan"),
        "f1_weighted": float((f1 * support).sum() / n) if n else float("nan"),
    }


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                    class_names: Sequence[str] = CLASS_NAMES) -> dict:
    """Độ đo tổng + báo cáo theo lớp + ma trận nhầm lẫn (dạng list để lưu JSON)."""
    K = len(class_names)
    cm = confusion(np.asarray(y_true), np.asarray(y_pred), K)
    out = {"n": int(cm.sum()), **metrics_from_confusion(cm)}
    tp = np.diag(cm).astype(float)
    per = {}
    for k, c in enumerate(class_names):
        p = tp[k] / cm[:, k].sum() if cm[:, k].sum() else 0.0
        r = tp[k] / cm[k].sum() if cm[k].sum() else 0.0
        per[c] = {"precision": float(p), "recall": float(r),
                  "f1": float(2 * p * r / (p + r)) if p + r else 0.0, "support": int(cm[k].sum())}
    out["per_class"] = per
    out["confusion_matrix"] = cm.tolist()
    return out


def bootstrap_std(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = len(CLASS_NAMES),
                  n_boot: int = 1000, seed: int = 42,
                  metrics: Iterable[str] = MAIN_METRICS) -> dict[str, float]:
    """Độ lệch chuẩn + khoảng tin cậy 95% của các độ đo, lấy mẫu lại có hoàn lại tập đánh giá.

    Trả về ``{"<metric>_std": ..., "<metric>_ci_low": ..., "<metric>_ci_high": ...}``.
    """
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    metrics = list(metrics)
    n = len(y_true)
    if n == 0 or n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    vals = np.empty((n_boot, len(metrics)))
    codes = y_true * n_classes + y_pred
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        cm = np.bincount(codes[idx], minlength=n_classes ** 2).reshape(n_classes, n_classes)
        m = metrics_from_confusion(cm)
        vals[b] = [m[k] for k in metrics]
    out = {}
    for j, k in enumerate(metrics):
        out[f"{k}_std"] = float(np.nanstd(vals[:, j], ddof=1))
        out[f"{k}_ci_low"] = float(np.nanpercentile(vals[:, j], 2.5))
        out[f"{k}_ci_high"] = float(np.nanpercentile(vals[:, j], 97.5))
    return out


# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, class_names: Sequence[str] = CLASS_NAMES, normalize: bool = True,
                          title: str | None = None, ax=None, save_path: str | Path | None = None):
    """Vẽ ma trận nhầm lẫn. ``normalize=True``: chia theo hàng (= recall từng lớp), ô ghi cả số ảnh."""
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    show = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.6, 4))
    else:
        fig = ax.figure
    im = ax.imshow(show, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    K = len(class_names)
    ax.set_xticks(range(K), class_names, rotation=30, ha="right")
    ax.set_yticks(range(K), class_names)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    thr = (show.max() if show.size else 1) / 2
    for i in range(K):
        for j in range(K):
            txt = f"{show[i, j]:.2f}\n({cm[i, j]})" if normalize else f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if show[i, j] > thr else "black")
    if title:
        ax.set_title(title, fontsize=10)
    if own:
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def top_confusions(cm, class_names: Sequence[str] = CLASS_NAMES, k: int = 5) -> pd.DataFrame:
    """Các cặp (thật -> đoán) nhầm nhiều nhất, vd. Half_Ripe -> Ripe."""
    cm = np.asarray(cm)
    rows = [{"that": class_names[i], "doan": class_names[j], "so_anh": int(cm[i, j]),
             "ti_le_trong_lop": cm[i, j] / max(cm[i].sum(), 1)}
            for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    return pd.DataFrame(rows).sort_values("so_anh", ascending=False).head(k).reset_index(drop=True) \
        if rows else pd.DataFrame(columns=["that", "doan", "so_anh", "ti_le_trong_lop"])


# ---------------------------------------------------------------------------
def results_table(results: pd.DataFrame, split: str = "test", metrics: Sequence[str] = MAIN_METRICS,
                  std: str = "auto", digits: int = 3, extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Bảng "mean ± std" cho slide từ DataFrame của ``experiment.load_results``.

    ``std``:
      - ``"boot"`` : bootstrap trên tập đánh giá (luôn có).
      - ``"seed"`` : độ lệch giữa các lần chạy khác seed (cột ``*_seed_std``, cần ``summarize_runs``).
      - ``"cv"``   : độ lệch 5 fold CV trên train (chỉ mô hình đặc trưng + GridSearch).
      - ``"auto"`` : seed nếu có >1 seed, ngược lại bootstrap.
    """
    df = results[results["split"] == split].copy()
    if df.empty:
        raise ValueError(f"Không có kết quả nào ở split='{split}'")
    out = df[["run_name", "model", "scheme", *extra_cols]].copy()
    for m in metrics:
        if std == "auto":
            s = df.get(f"{m}_seed_std")
            s = s.where(df.get("n_seeds", pd.Series(1, index=df.index)) > 1, df.get(f"{m}_std")) \
                if s is not None else df.get(f"{m}_std")
        elif std == "seed":
            s = df.get(f"{m}_seed_std")
        elif std == "cv":
            s = df.get(f"cv_{m}_std")
        else:
            s = df.get(f"{m}_std")
        mean = df[m]
        if s is None:
            out[m] = mean.map(lambda v: f"{v:.{digits}f}")
        else:
            out[m] = [f"{a:.{digits}f} ± {b:.{digits}f}" if pd.notna(b) else f"{a:.{digits}f}"
                      for a, b in zip(mean, s)]
    order = df[metrics[-1]].to_numpy().argsort()[::-1]
    return out.iloc[order].reset_index(drop=True)


__all__ = ["MAIN_METRICS", "confusion", "metrics_from_confusion", "compute_metrics", "bootstrap_std",
           "plot_confusion_matrix", "top_confusions", "results_table"]
