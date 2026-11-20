"""Cấu hình chung cho toàn bộ dự án.

Mọi tham số (đường dẫn, ngưỡng, seed, ánh xạ nhãn) nằm ở đây hoặc trong file YAML
trong thư mục ``configs/``. Code và notebook KHÔNG hard-code các giá trị này.

Ví dụ::

    from banana_ripeness.config import load_config
    cfg = load_config("configs/kaggle.yaml")
    cfg = load_config("configs/kaggle.yaml", output_dir="/tmp/out")   # ghi đè 1 trường
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

CLASS_NAMES: list[str] = ["Unripe", "Half_Ripe", "Ripe", "Overripe"]

# Tên thư mục gốc (viết thường) -> nhãn chuẩn. Ánh xạ giữa các dataset là XẤP XỈ,
# nhất là "Class A-D" của BananaRipeness_GitHub (ghi rõ điều này trên slide).
DEFAULT_LABEL_MAPPING: dict[str, str] = {
    # BananaID
    "unripe": "Unripe", "half ripe": "Half_Ripe", "half_ripe": "Half_Ripe",
    "ripe": "Ripe", "overripe": "Overripe",
    # BananaImageBD
    "green": "Unripe", "semi ripe": "Half_Ripe", "semi-ripe": "Half_Ripe", "semi_ripe": "Half_Ripe",
    # BananaRipeness_GitHub
    "class a": "Unripe", "class b": "Half_Ripe", "class c": "Ripe", "class d": "Overripe",
}


@dataclass
class MaskConfig:
    """Tách nền bằng ngưỡng Otsu trên kênh Saturation (HSV).

    Nền ảnh (trắng / xám) có độ bão hoà thấp, vỏ chuối (xanh, vàng, nâu) có độ bão hoà cao,
    nên Otsu trên kênh S tách tốt hơn Otsu trên ảnh xám (chuối vàng và nền trắng cùng sáng).
    """
    channel: str = "s"            # "s" (HSV saturation) hoặc "gray"
    blur_ksize: int = 5           # Gaussian blur trước khi Otsu, giảm nhiễu
    morph_ksize: int = 5          # kernel mở/đóng hình thái học
    min_component_ratio: float = 0.10   # giữ thành phần liên thông >= 10% thành phần lớn nhất
    min_area: float = 0.02        # mask < 2% ảnh -> coi như thất bại, dùng cả ảnh
    max_area: float = 0.95        # mask > 95% ảnh -> coi như thất bại, dùng cả ảnh


@dataclass
class Config:
    # ---------- đường dẫn ----------
    dataset_dirs: dict[str, str] = field(default_factory=dict)
    output_dir: str = "outputs/preprocessed"

    # ---------- chung ----------
    seed: int = 42
    image_size: int = 224
    num_workers: int | None = None      # None = số core

    # ---------- thu thập ----------
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    label_mapping: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LABEL_MAPPING))
    excluded_dir_names: list[str] = field(
        default_factory=lambda: ["augmented", "augmentation", "synthetic", "varying illumination"])
    image_extensions: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"])
    eda_counts: dict[str, int] = field(default_factory=dict)   # để đối chiếu với báo cáo EDA

    # ---------- loại trùng ----------
    hash_size: int = 16          # dHash 16x16 = 256 bit
    dup_thr: int = 6             # <= 6 bit khác và cùng nhãn -> bản sao, giữ 1
    group_thr: int = 16          # <= 16 bit khác (bất kể nhãn) -> bắt buộc cùng split
    suspect_color_thr: float = 12.0
    drop_suspect_label_noise: bool = False
    compare_hash64: bool = True  # so sánh với dHash 64 bit mà notebook EDA đã dùng

    # ---------- chia tập ----------
    external_dataset: str = "BananaImageBD"   # dataset giữ ngoài cho kiểm định ngoài
    n_test_folds: int = 7        # test ~ 1/7
    n_val_folds: int = 6         # val ~ 1/6 phần còn lại
    n_cv_folds: int = 5          # fold trên tập train để chọn tham số + tính độ lệch chuẩn

    # ---------- tách nền ----------
    mask: MaskConfig = field(default_factory=MaskConfig)

    # ------------------------------------------------------------------
    @property
    def n_bits(self) -> int:
        return self.hash_size * self.hash_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
                              encoding="utf-8")


def _from_dict(data: dict[str, Any]) -> Config:
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Khoá cấu hình không hợp lệ: {sorted(unknown)}")
    data = dict(data)
    if isinstance(data.get("mask"), dict):
        data["mask"] = MaskConfig(**data["mask"])
    return Config(**data)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Đọc YAML (nếu có) rồi ghi đè bằng ``overrides``."""
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return _from_dict(data)
