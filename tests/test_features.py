"""Kiểm thử đặc trưng màu, cache đặc trưng và các hàm độ đo."""
import numpy as np
import pytest

from src.evaluation import bootstrap_std, compute_metrics, top_confusions
from src.features import ColorHistogramExtractor, HogExtractor, color_histogram, extract_features, feature_names, \
    histogram_stats


def _img(color=(230, 200, 30)):
    img = np.full((64, 64, 3), 250, np.uint8)
    img[16:48, 8:56] = color
    mask = np.zeros((64, 64), bool)
    mask[16:48, 8:56] = True
    return img, mask


@pytest.mark.parametrize("space", ["rgb", "hsv"])
def test_color_histogram_shape_and_norm(space):
    img, mask = _img()
    f = color_histogram(img, mask, space=space, bins=16)
    assert f.shape == (66,) and f.dtype == np.float32
    assert len(feature_names(space, 16)) == 66
    for c in range(3):                     # histogram mỗi kênh tổng = 1
        assert abs(f[c * 16:(c + 1) * 16].sum() - 1) < 1e-5
    assert color_histogram(img, None, space=space, bins=8, stats=False).shape == (24,)


def test_mask_removes_background():
    img, mask = _img((230, 200, 30))
    names = feature_names("rgb", 16)
    f = color_histogram(img, mask, "rgb", 16)
    # trong mask chỉ có 1 màu -> kênh R dồn vào 1 bin, std = 0, energy = 1
    assert f[:16].max() == pytest.approx(1.0)
    assert f[names.index("R_std")] == pytest.approx(0, abs=1e-6)
    assert f[names.index("R_energy")] == pytest.approx(1.0)
    assert f[names.index("R_mean")] == pytest.approx(230 / 255, abs=1e-6)
    f_full = color_histogram(img, None, "rgb", 16)
    assert f_full[names.index("R_std")] > 0.01


def test_hue_bins_cover_opencv_range():
    # H của OpenCV trong [0, 180): pixel H=179 phải rơi vào bin cuối
    hsv = np.zeros((4, 4, 3), np.uint8)
    hsv[..., 0], hsv[..., 1], hsv[..., 2] = 179, 255, 255
    import cv2
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    f = color_histogram(rgb, None, "hsv", 16, stats=False)
    assert f[15] == pytest.approx(1.0) or f[0] == pytest.approx(1.0)   # đỏ: 179 hoặc 0 sau làm tròn


def test_histogram_stats_uniform():
    p = np.full(256, 1 / 256)
    m, sd, skew, energy, smooth, ent = histogram_stats(p)
    assert m == pytest.approx(0.5) and abs(skew) < 1e-9
    assert energy == pytest.approx(1 / 256) and ent == pytest.approx(8.0)
    assert smooth == pytest.approx(1 - 1 / (1 + sd ** 2))


def test_feature_cache(prepped, tmp_path):
    from src.data import load_manifest
    cfg, _ = prepped
    df = load_manifest(cfg.output_dir).head(20)
    ex = ColorHistogramExtractor("hsv", 16, True, True)
    X1 = extract_features(ex, df.head(10), cache_dir=tmp_path, n_jobs=1, verbose=False)
    X2 = extract_features(ex, df, cache_dir=tmp_path, n_jobs=1, verbose=False)   # 10 ảnh mới
    assert X2.shape == (20, 66)
    np.testing.assert_allclose(X1, X2[:10])
    X3 = extract_features(ex, df.iloc[::-1], cache_dir=tmp_path, n_jobs=1, verbose=False)
    np.testing.assert_allclose(X3, X2[::-1])
    X4 = extract_features(ex, df, cache_dir=False, n_jobs=2, verbose=False)        # song song, không cache
    np.testing.assert_allclose(X4, X2)


def test_metrics_match_sklearn():
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 300)
    p = np.where(rng.random(300) < 0.7, y, rng.integers(0, 4, 300))
    m = compute_metrics(y, p)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, p))
    assert m["f1_macro"] == pytest.approx(f1_score(y, p, average="macro"))
    assert m["precision_macro"] == pytest.approx(precision_score(y, p, average="macro"))
    assert m["recall_macro"] == pytest.approx(recall_score(y, p, average="macro"))
    assert np.array(m["confusion_matrix"]).sum() == 300
    b = bootstrap_std(y, p, n_boot=200)
    assert 0 < b["f1_macro_std"] < 0.1
    assert b["f1_macro_ci_low"] < m["f1_macro"] < b["f1_macro_ci_high"]
    assert len(top_confusions(m["confusion_matrix"], k=3)) == 3


@pytest.mark.parametrize("mode,dims", [("gray", 1764), ("rgb", 5292)])
def test_hog_dims_and_mask(mode, dims):
    img, mask = _img()
    img[~mask] = np.random.default_rng(0).integers(0, 255, (int((~mask).sum()), 3))    # nền nhiễu: mask phải có tác dụng
    ex = HogExtractor(color_mode=mode)
    f = ex(img, mask)
    assert f.dtype == np.float32 and f.shape == (dims,) and len(ex.names()) == dims
    assert not np.allclose(f, HogExtractor(color_mode=mode, use_mask=False)(img, mask))   # mask đổi đặc trưng
    assert ex.key != HogExtractor(color_mode=mode, pixels_per_cell=32).key
