"""Đặc trưng màu: histogram 1D từng kênh + thống kê mô-men của histogram.

Với mỗi kênh của không gian màu (RGB hoặc HSV), chỉ tính trên pixel thuộc mask (nếu có):

1. Histogram ``bins`` bin, chuẩn hoá tổng = 1                      -> 3 x bins chiều
2. Thống kê tính từ histogram mịn (256 mức; kênh H của OpenCV: 180 mức), mức xám
   chuẩn hoá z ∈ [0, 1] (Gonzalez & Woods, *Digital Image Processing*, mục
   "Statistical approaches" trong phần mô tả kết cấu):

   ======================  =======================================
   mean        m           Σ z·p(z)
   std         σ           sqrt(Σ (z - m)²·p(z))
   skewness                μ3 / σ³   (μ3 = Σ (z - m)³·p(z))
   energy (uniformity) U   Σ p(z)²
   smoothness  R           1 - 1 / (1 + σ²)
   entropy     e           -Σ p(z)·log2 p(z)
   ======================  =======================================
                                                                  -> 3 x 6 = 18 chiều

Mặc định ``bins=16`` -> 48 + 18 = **66 chiều**.

HSV của OpenCV: H ∈ [0, 180) (độ / 2), S, V ∈ [0, 256). Lưu ý H là đại lượng vòng
(0 và 179 đều là đỏ); với chuối (xanh -> vàng -> nâu) gần như không có pixel ở vùng
đỏ nên coi H tuyến tính là chấp nhận được (ghi chú trên slide nếu bị hỏi).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

STAT_NAMES = ("mean", "std", "skewness", "energy", "smoothness", "entropy")
CHANNELS = {"rgb": ("R", "G", "B"), "hsv": ("H", "S", "V")}
LEVELS = {"rgb": (256, 256, 256), "hsv": (180, 256, 256)}


def _convert(rgb: np.ndarray, space: str) -> np.ndarray:
    if space == "rgb":
        return rgb
    if space == "hsv":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    raise ValueError(f"space phải là 'rgb' hoặc 'hsv', nhận '{space}'")


def histogram_stats(p: np.ndarray) -> np.ndarray:
    """6 thống kê của histogram chuẩn hoá ``p`` (độ dài L, tổng = 1), z = i / (L - 1)."""
    L = len(p)
    z = np.arange(L, dtype=np.float64) / (L - 1)
    m = float((z * p).sum())
    d = z - m
    var = float((d ** 2 * p).sum())
    sd = np.sqrt(var)
    mu3 = float((d ** 3 * p).sum())
    skew = mu3 / sd ** 3 if sd > 1e-8 else 0.0
    energy = float((p ** 2).sum())
    smooth = 1.0 - 1.0 / (1.0 + var)
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    return np.array([m, sd, skew, energy, smooth, entropy], dtype=np.float64)


def color_histogram(rgb: np.ndarray, mask: np.ndarray | None = None, space: str = "rgb",
                    bins: int = 16, stats: bool = True) -> np.ndarray:
    """Vector đặc trưng màu của 1 ảnh RGB uint8 (H, W, 3).

    ``mask``: bool (H, W), True = chuối. ``None`` hoặc mask rỗng -> dùng cả ảnh.
    Trả về float32 độ dài ``3*bins (+18 nếu stats)``, thứ tự khớp ``feature_names``.
    """
    img = _convert(rgb, space)
    if mask is not None and mask.any():
        px = img[mask]
    else:
        px = img.reshape(-1, 3)
    hists, moments = [], []
    for c, L in enumerate(LEVELS[space]):
        fine = np.bincount(px[:, c].astype(np.int64), minlength=L)[:L].astype(np.float64)
        fine /= max(fine.sum(), 1.0)
        # gộp L mức thành `bins` bin đều nhau (cùng cách chia với np.histogram trên [0, L))
        edges = np.linspace(0, L, bins + 1)
        idx = np.minimum(np.searchsorted(edges, np.arange(L), side="right") - 1, bins - 1)
        hists.append(np.bincount(idx, weights=fine, minlength=bins))
        if stats:
            moments.append(histogram_stats(fine))
    parts = hists + moments
    return np.concatenate(parts).astype(np.float32)


def feature_names(space: str = "rgb", bins: int = 16, stats: bool = True) -> list[str]:
    ch = CHANNELS[space]
    names = [f"{c}_hist{b:02d}" for c in ch for b in range(bins)]
    if stats:
        names += [f"{c}_{s}" for c in ch for s in STAT_NAMES]
    return names


def feature_summary(spaces=("rgb", "hsv"), bins: int = 16, stats: bool = True):
    """Bảng số chiều đặc trưng cho slide (nguồn, số bin, số chiều)."""
    import pandas as pd
    rows = []
    for sp in spaces:
        rows.append({"khong_gian": sp.upper(), "kenh": ", ".join(CHANNELS[sp]), "so_bin_moi_kenh": bins,
                     "chieu_histogram": 3 * bins, "chieu_thong_ke": 3 * len(STAT_NAMES) if stats else 0,
                     "tong_so_chieu": 3 * bins + (3 * len(STAT_NAMES) if stats else 0),
                     "thong_ke": ", ".join(STAT_NAMES) if stats else ""})
    return pd.DataFrame(rows)


def plot_color_features(image_path: str, mask_path: str | None = None, spaces=("rgb", "hsv"), bins: int = 16,
                        title: str | None = None, save_path=None):
    """Minh hoạ 1 ảnh: ảnh gốc | ảnh sau tách nền | histogram từng không gian màu (có và không mask)."""
    import matplotlib.pyplot as plt

    from ..data import apply_mask, read_mask, read_rgb
    rgb = read_rgb(image_path)
    mask = read_mask(mask_path) if mask_path else None
    fig, axes = plt.subplots(1, 2 + len(spaces), figsize=(3.2 * (2 + len(spaces)), 3))
    axes[0].imshow(rgb)
    axes[0].set_title(title or "ảnh", fontsize=9)
    axes[1].imshow(apply_mask(rgb, mask) if mask is not None else rgb)
    axes[1].set_title("sau tách nền (Otsu kênh S)" if mask is not None else "không có mask", fontsize=9)
    colors = {"R": "tab:red", "G": "tab:green", "B": "tab:blue", "H": "tab:orange", "S": "tab:purple",
              "V": "tab:gray"}
    x = np.arange(bins)
    for ax, sp in zip(axes[2:], spaces):
        f_mask = color_histogram(rgb, mask, sp, bins, stats=False)
        f_full = color_histogram(rgb, None, sp, bins, stats=False)
        for c, name in enumerate(CHANNELS[sp]):
            ax.plot(x, f_mask[c * bins:(c + 1) * bins], color=colors[name], label=f"{name} (mask)")
            ax.plot(x, f_full[c * bins:(c + 1) * bins], color=colors[name], ls=":", alpha=0.7)
        ax.set_title(f"histogram {sp.upper()} ({bins} bin/kênh)\nliền: trong mask, chấm: cả ảnh", fontsize=8)
        ax.set_xlabel("bin")
        ax.legend(fontsize=6)
    for ax in axes[:2]:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


@dataclass(frozen=True)
class ColorHistogramExtractor:
    """Bộ trích đặc trưng có ``key`` riêng để cache (đổi tham số -> cache khác)."""
    space: str = "rgb"
    bins: int = 16
    stats: bool = True
    use_mask: bool = True

    @property
    def key(self) -> str:
        return f"color_{self.space}_b{self.bins}_{'stats' if self.stats else 'nostats'}_{'mask' if self.use_mask else 'full'}"

    @property
    def needs_mask(self) -> bool:
        return self.use_mask

    def __call__(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return color_histogram(rgb, mask if self.use_mask else None, self.space, self.bins, self.stats)

    def names(self) -> list[str]:
        return feature_names(self.space, self.bins, self.stats)


__all__ = ["color_histogram", "histogram_stats", "feature_names", "feature_summary", "plot_color_features",
           "ColorHistogramExtractor", "STAT_NAMES"]
