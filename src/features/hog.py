"""Đặc trưng HOG (Histogram of Oriented Gradients).

Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. CVPR.

Ảnh RGB (224, 224, 3) + mask -> nền thành trắng (nếu ``use_mask``) -> resize ``resize`` x ``resize`` -> HOG
(``skimage.feature.hog``, chuẩn hoá block L2-Hys). ``color_mode``:

- ``gray`` : HOG của ảnh xám.
- ``rgb``  : HOG riêng từng kênh R, G, B rồi nối ("chia bin theo màu RGB"), số chiều gấp 3.

Số chiều mỗi kênh = (resize // pixels_per_cell - cells_per_block + 1)^2 * cells_per_block^2 * orientations
(mặc định 128, 16, 2, 9 -> 7^2 * 4 * 9 = 1764; ``rgb`` -> 5292).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import hog

CHANNELS = {"gray": ("gray",), "rgb": ("R", "G", "B")}


@dataclass(frozen=True)
class HogExtractor:
    orientations: int = 9
    pixels_per_cell: int = 16
    cells_per_block: int = 2
    resize: int = 128
    color_mode: str = "rgb"
    use_mask: bool = True

    def __post_init__(self):
        if self.color_mode not in CHANNELS:
            raise ValueError(f"color_mode phải là {sorted(CHANNELS)}, nhận '{self.color_mode}'")

    @property
    def key(self) -> str:
        return (f"hog_o{self.orientations}_p{self.pixels_per_cell}_c{self.cells_per_block}_r{self.resize}"
                f"_{self.color_mode}_{'mask' if self.use_mask else 'full'}")

    @property
    def needs_mask(self) -> bool:
        return self.use_mask

    def __call__(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if self.use_mask and mask is not None and mask.any():
            rgb = np.where(mask[..., None], rgb, 255).astype(np.uint8)      # nền -> trắng
        img = cv2.resize(rgb, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        planes = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)] if self.color_mode == "gray" else [img[..., c] for c in range(3)]
        return np.concatenate([hog(p, orientations=self.orientations,
                                   pixels_per_cell=(self.pixels_per_cell,) * 2,
                                   cells_per_block=(self.cells_per_block,) * 2, block_norm="L2-Hys")
                               for p in planes]).astype(np.float32)

    def names(self) -> list[str]:
        per_channel = (self.resize // self.pixels_per_cell - self.cells_per_block + 1) ** 2 \
            * self.cells_per_block ** 2 * self.orientations
        return [f"hog_{ch}_{i}" for ch in CHANNELS[self.color_mode] for i in range(per_channel)]


__all__ = ["HogExtractor"]
