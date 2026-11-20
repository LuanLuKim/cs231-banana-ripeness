"""Gộp kết quả của cả nhóm (mọi result.json tìm được) thành bảng so sánh chung.

    python scripts/compare.py /kaggle/input /kaggle/working/results --out reports
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):          # console Windows (cp1252) không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from banana_ripeness.experiment import compare_table, load_results  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Bảng so sánh kết quả các mô hình")
    ap.add_argument("paths", nargs="*", help="thư mục chứa kết quả (mặc định: results_dir + /kaggle/input)")
    ap.add_argument("--out", default="reports", help="thư mục ghi results_<scheme>_<split>.csv")
    ap.add_argument("--std", default="auto", choices=["auto", "boot", "seed", "cv"])
    a = ap.parse_args()

    res = load_results(a.paths or None)
    if res.empty:
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results_all_runs.csv", index=False)
    for scheme in sorted(res.scheme.unique()):
        for split in ("test", "external"):
            if not ((res.scheme == scheme) & (res.split == split)).any():
                continue
            t = compare_table(res, split=split, scheme=scheme, std=a.std,
                              save_path=out / f"results_{scheme}_{split}.csv")
            print(f"\n===== scheme={scheme} | split={split} =====")
            print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
