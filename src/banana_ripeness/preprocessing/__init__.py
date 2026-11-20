"""Tiền xử lý: thu thập -> loại trùng -> nhóm -> chia tập -> tách nền -> lưu ảnh + manifest."""
from .pipeline import PreprocessResult, pad_square_replicate, run_preprocessing
from .segmentation import MaskResult, otsu_mask

__all__ = ["run_preprocessing", "PreprocessResult", "pad_square_replicate", "otsu_mask", "MaskResult"]
