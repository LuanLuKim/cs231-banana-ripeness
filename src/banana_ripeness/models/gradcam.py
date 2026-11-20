"""KHUNG — Grad-CAM: vùng ảnh mà ResNet18 dựa vào để quyết định.

>>> NGƯỜI PHỤ TRÁCH: Quỳnh. File này chỉ là khung, Quỳnh tự cài đặt phần TODO. <<<

Tài liệu: Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via
gradient-based localization. ICCV.

Ý tưởng: A = feature map lớp conv cuối (``layer4`` của ResNet18), y_c = logit lớp c,
α_k = trung bình không gian của ∂y_c/∂A_k, CAM = ReLU(Σ_k α_k A_k), phóng to về kích thước ảnh.

Hàm có sẵn: ``overlay(rgb, cam)`` chồng heatmap lên ảnh để vẽ.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class GradCAM:
    def __init__(self, net, target_layer=None):
        # TODO(Quỳnh): lưu net, chọn target_layer (mặc định net.layer4[-1]), đăng ký hook lấy activation + gradient
        raise NotImplementedError("GradCAM chưa cài đặt (phần của Quỳnh)")

    def __call__(self, x, class_idx: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
        """``x``: tensor (1, 3, H, W). Trả về (cam [0,1] kích thước HxW, lớp được giải thích, xác suất 4 lớp)."""
        raise NotImplementedError

    def remove(self):
        """Gỡ hook."""
        raise NotImplementedError


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Chồng heatmap (JET) lên ảnh RGB uint8. ``cam``: float [0, 1]."""
    cam = cv2.resize(cam.astype(np.float32), (rgb.shape[1], rgb.shape[0]))
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * rgb + alpha * heat)


def gradcam_grid(model, df: pd.DataFrame, class_idx: int | None = None, save_path: str | Path | None = None):
    """Lưới ảnh gốc + Grad-CAM cho các dòng của ``df`` (ghi nhãn thật / đoán). ``model``: ResNet18Classifier."""
    # TODO(Quỳnh)
    raise NotImplementedError("gradcam_grid chưa cài đặt (phần của Quỳnh)")


__all__ = ["GradCAM", "overlay", "gradcam_grid"]
