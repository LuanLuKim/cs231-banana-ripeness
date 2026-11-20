"""KHUNG — đặc trưng HOG (Histogram of Oriented Gradients).

>>> NGƯỜI PHỤ TRÁCH: Nguyên. File này chỉ là khung, Nguyên tự cài đặt phần TODO. <<<

Tài liệu: Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. CVPR.

Hợp đồng (contract) để dùng được với ``extract_features`` (cache theo image_id) và ``ImageFeatureModel``:

- ``key``        : chuỗi DUY NHẤT theo bộ tham số (đổi tham số -> đổi key -> cache mới).
- ``needs_mask`` : True nếu cần đọc mask (``extract_features`` sẽ truyền mask bool vào ``__call__``).
- ``__call__(rgb, mask)`` : ảnh RGB uint8 (224, 224, 3) + mask bool (224, 224) hoặc None
                            -> vector float32 1 chiều, độ dài cố định.
- ``names()``    : tên từng chiều (để ghi số chiều trên slide).

Gợi ý: ``skimage.feature.hog`` (đã có trên Kaggle/Colab); "chia bin theo màu RGB" Nguyên tự chốt
(vd. HOG từng kênh R, G, B rồi nối; hoặc nối thêm ``color_histogram`` trong ``color_histogram.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HogExtractor:
    # TODO(Nguyên): thêm/bớt tham số theo cách làm đã chốt
    orientations: int = 9
    pixels_per_cell: int = 16
    cells_per_block: int = 2
    resize: int = 128
    color_mode: str = "gray"
    use_mask: bool = True

    @property
    def key(self) -> str:
        return (f"hog_o{self.orientations}_p{self.pixels_per_cell}_c{self.cells_per_block}_r{self.resize}"
                f"_{self.color_mode}_{'mask' if self.use_mask else 'full'}")

    @property
    def needs_mask(self) -> bool:
        return self.use_mask

    def __call__(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        # TODO(Nguyên): tính HOG cho 1 ảnh, trả về np.float32 1 chiều
        raise NotImplementedError("HogExtractor.__call__ chưa cài đặt (phần của Nguyên)")

    def names(self) -> list[str]:
        # TODO(Nguyên): danh sách tên từng chiều, len == số chiều vector
        raise NotImplementedError("HogExtractor.names chưa cài đặt (phần của Nguyên)")


__all__ = ["HogExtractor"]
