"""Kiểm thử preprocessing end-to-end trên dữ liệu giả (fixture ``prepped`` trong conftest). Chạy: pytest -q"""
from pathlib import Path

from src.data import get_split, load_manifest, read_mask

def test_manifest_and_files(prepped):
    cfg, res = prepped
    df = load_manifest(cfg.output_dir)
    assert len(df) > 300
    assert Path(df.image_path[0]).exists() and Path(df.mask_path[0]).exists()
    for scheme in ("mixed", "external"):
        tr = get_split(df, scheme, "train")
        assert set(tr[f"cv_fold_{scheme}"]) == set(range(cfg.n_cv_folds))
    ext = get_split(df, "external", "external")
    assert set(ext.dataset) == {"BananaImageBD"}
    assert "BananaImageBD" not in set(get_split(df, "external", "train").dataset)


def test_masks_reasonable(prepped):
    cfg, res = prepped
    df = load_manifest(cfg.output_dir)
    assert df.mask_fallback.mean() < 0.1
    m = read_mask(df.mask_path[0])
    assert m.shape == (cfg.image_size, cfg.image_size)
    assert 0.02 < m.mean() < 0.6
