"""Chạy thí nghiệm từ dòng lệnh (cùng code với notebook).

    # cả danh sách thí nghiệm của 1 người trong configs/experiments.yaml
    python scripts/train.py --who luan

    # 1 mô hình, tham số ghi tay (list ngăn bằng dấu phẩy -> tổ hợp)
    python scripts/train.py --model svm_color --set kernel=linear,rbf space=hsv use_mask=true --schemes external

Dữ liệu: --data-root (thư mục có manifest.csv); bỏ trống -> tự tìm trong /kaggle/input, /content, outputs/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):          # console Windows (cp1252) không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

from banana_ripeness.experiment import compare_table, load_experiments, run_grid  # noqa: E402


def parse_set(items: list[str]) -> dict:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        vals = [yaml.safe_load(x) for x in v.split(",")]
        out[k] = vals if len(vals) > 1 else vals[0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Huấn luyện + đánh giá mô hình độ chín chuối")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--who", help="nhóm thí nghiệm trong experiments.yaml: luan / nguyen / quynh")
    g.add_argument("--model", help="tên mô hình: svm_color, knn_color, hog_svm, resnet18")
    ap.add_argument("--set", nargs="*", default=[], help="tham số mô hình key=value (value1,value2 để tổ hợp)")
    ap.add_argument("--experiments", default=str(ROOT / "configs" / "experiments.yaml"))
    ap.add_argument("--data-root")
    ap.add_argument("--results-dir")
    ap.add_argument("--schemes", nargs="+", default=["mixed", "external"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--owner")
    ap.add_argument("--no-skip", action="store_true", help="chạy lại cả lần đã có kết quả")
    a = ap.parse_args()

    specs = load_experiments(a.experiments, a.who) if a.who else [{"model": a.model, **parse_set(a.set)}]
    res = run_grid(specs, data_root=a.data_root, schemes=a.schemes, seeds=a.seeds, results_dir=a.results_dir,
                   skip_existing=not a.no_skip, owner=a.owner or a.who)
    if res.empty:
        print("Không có lần chạy nào thành công.")
        return 1
    for split in ("test", "external"):
        if split in set(res.split):
            print(f"\n===== {split} =====")
            print(compare_table(res, split=split).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
