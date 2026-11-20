"""Dữ liệu giả dùng chung cho mọi test (ảnh elip màu trên nền trắng, 3 dataset x 4 nhãn)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# (tên thư mục gốc, màu RGB) theo từng dataset giả
FAKE = {
    "BananaID": {"unripe": (60, 160, 40), "half ripe": (170, 190, 40),
                 "ripe": (230, 200, 30), "overripe": (150, 100, 30)},
    "BananaImageBD": {"green": (60, 160, 40), "semi-ripe": (170, 190, 40),
                      "ripe": (230, 200, 30), "overripe": (150, 100, 30)},
    "BananaRipeness_GitHub": {"class a": (60, 160, 40), "class b": (170, 190, 40),
                              "class c": (230, 200, 30), "class d": (150, 100, 30)},
}


def make_fake(root: Path, n_per_class: int = 30, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    dirs = {}
    for ds, classes in FAKE.items():
        dirs[ds] = str(root / ds)
        for folder, color in classes.items():
            d = root / ds / "ripeness" / folder
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                img = np.full((200, 300, 3), 245, np.uint8)
                img += rng.integers(0, 8, img.shape, dtype=np.uint8)
                c = np.clip(np.array(color) + rng.integers(-25, 25, 3), 0, 255).tolist()
                center = (int(rng.integers(90, 210)), int(rng.integers(70, 130)))
                axes = (int(rng.integers(50, 90)), int(rng.integers(18, 35)))
                cv2.ellipse(img, center, axes, float(rng.integers(0, 180)), 0, 360, c, -1)
                # hoạ tiết ngẫu nhiên để dHash các ảnh khác nhau
                for _ in range(6):
                    p = (int(rng.integers(0, 300)), int(rng.integers(0, 200)))
                    cv2.circle(img, p, int(rng.integers(3, 12)), [int(x) for x in rng.integers(150, 240, 3)], -1)
                cv2.imwrite(str(d / f"{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return dirs


@pytest.fixture(scope="session")
def prepped(tmp_path_factory):
    """(cfg, PreprocessResult) sau khi chạy preprocessing trên dữ liệu giả."""
    from banana_ripeness.config import load_config
    from banana_ripeness.preprocessing import run_preprocessing

    root = tmp_path_factory.mktemp("fake")
    dirs = make_fake(root / "raw")
    cfg = load_config(None, dataset_dirs=dirs, output_dir=str(root / "out"), num_workers=1)
    res = run_preprocessing(cfg, verbose=False)
    return cfg, res
