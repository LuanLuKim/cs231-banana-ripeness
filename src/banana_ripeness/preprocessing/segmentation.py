"""Tách nền bằng ngưỡng Otsu.

Otsu, N. (1979). A threshold selection method from gray-level histograms.
IEEE Trans. Systems, Man, and Cybernetics, 9(1), 62-66.

Các bước (dễ trình bày trên slide):

1. Lấy kênh Saturation của HSV (nền trắng/xám có S thấp, vỏ chuối có S cao).
2. Làm mờ Gaussian để giảm nhiễu.
3. Otsu chọn ngưỡng tự động -> mask nhị phân.
4. Mở rồi đóng hình thái học để bỏ đốm nhỏ, lấp khe hở.
5. Giữ các thành phần liên thông đủ lớn, lấp lỗ bên trong quả.
6. Nếu diện tích mask quá nhỏ hoặc quá lớn -> coi như thất bại, dùng cả ảnh (đánh dấu ``fallback``).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import MaskConfig


@dataclass
class MaskResult:
    mask: np.ndarray        # uint8 0/255
    threshold: float
    area: float             # tỉ lệ pixel thuộc chuối
    fallback: bool


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = mask.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    # loang từ các góc là nền
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flood[seed[1], seed[0]] == 0:
            cv2.floodFill(flood, ff, seed, 255)
    return mask | cv2.bitwise_not(flood)


def otsu_mask(rgb: np.ndarray, cfg: MaskConfig | None = None) -> MaskResult:
    cfg = cfg or MaskConfig()
    if cfg.channel == "s":
        ch = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    elif cfg.channel == "gray":
        ch = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"channel không hợp lệ: {cfg.channel}")

    if cfg.blur_ksize > 1:
        ch = cv2.GaussianBlur(ch, (cfg.blur_ksize, cfg.blur_ksize), 0)
    thr, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cfg.channel == "gray":       # chuối tối hơn nền trắng
        m = cv2.bitwise_not(m)

    if cfg.morph_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_ksize, cfg.morph_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.flatnonzero(areas >= cfg.min_component_ratio * areas.max()) + 1
        m = np.isin(lab, keep).astype(np.uint8) * 255
    m = _fill_holes(m)

    area = float((m > 0).mean())
    fallback = not (cfg.min_area <= area <= cfg.max_area)
    if fallback:
        m = np.full_like(m, 255)
    return MaskResult(mask=m, threshold=float(thr), area=area, fallback=fallback)
