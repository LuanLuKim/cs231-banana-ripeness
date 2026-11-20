"""Chạy tiền xử lý từ dòng lệnh.

    python scripts/preprocess.py --config configs/kaggle.yaml
    python scripts/preprocess.py --config configs/local.yaml --output-dir data/preprocessed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):          # console Windows (cp1252) không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from banana_ripeness.config import load_config  # noqa: E402
from banana_ripeness.preprocessing import run_preprocessing  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Tiền xử lý dữ liệu độ chín chuối")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir")
    ap.add_argument("--num-workers", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config, output_dir=a.output_dir, num_workers=a.num_workers)
    res = run_preprocessing(cfg)
    return 1 if res.warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
