"""Đặc trưng thủ công: histogram màu (+ thống kê mô-men) và HOG, có cache theo image_id."""
from .cache import default_cache_dir, extract_features
from .color_histogram import (ColorHistogramExtractor, color_histogram, feature_names, feature_summary,
                              histogram_stats, plot_color_features)
from .hog import HogExtractor

__all__ = ["color_histogram", "histogram_stats", "feature_names", "ColorHistogramExtractor",
           "HogExtractor", "extract_features", "default_cache_dir", "feature_summary", "plot_color_features"]
