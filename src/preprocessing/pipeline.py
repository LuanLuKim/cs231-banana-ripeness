"""Chạy toàn bộ tiền xử lý: ``run_preprocessing(cfg)``.

Gọi từ dòng lệnh::

    python scripts/preprocess.py --config configs/kaggle.yaml

hoặc từ notebook::

    from src.preprocessing import run_preprocessing
    result = run_preprocessing(load_config("configs/kaggle.yaml"))
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import Config
from .collect import collect_images, compare_with_eda, read_fingerprints, tqdm
from .dedup import Deduplicator, assign_groups, hash64_comparison
from .reports import mask_qa_grid, pair_grid
from .segmentation import otsu_mask
from .split import assign_cv_folds, leakage_report, split_external, split_mixed


@dataclass
class PreprocessResult:
    output_dir: Path
    manifest: pd.DataFrame
    stats: dict
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def print_summary(self) -> None:
        print(f"Output: {self.output_dir}")
        print(f"Số ảnh cuối: {len(self.manifest)}")
        for name in ("dedup_log", "split_mixed", "split_external", "leakage", "mask_stats"):
            if name in self.tables:
                print(f"\n== {name} ==")
                print(self.tables[name].to_string())
        if self.warnings:
            print("\n!! CẢNH BÁO cần báo nhóm:")
            for w in self.warnings:
                print("  -", w)
        else:
            print("\nKhông có cảnh báo.")


# ---------------------------------------------------------------------------
def pad_square_replicate(rgb: np.ndarray) -> np.ndarray:
    """Pad ảnh thành hình vuông bằng cách lặp pixel biên (không làm méo quả chuối)."""
    h, w = rgb.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(rgb, top, s - h - top, left, s - w - left, cv2.BORDER_REPLICATE)


def mean_hue(rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Hue trung bình (độ, 0-360) của pixel đủ bão hoà (S>60, V>50), trong mask nếu có."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = (hsv[..., 1] > 60) & (hsv[..., 2] > 50)
    if mask is not None:
        m &= mask
    return float(hsv[..., 0][m].mean() * 2) if m.sum() >= 8 else float("nan")


def _process_one(args):
    src, img_out, mask_out, size, mask_cfg = args
    bgr = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_COLOR)
    rgb = pad_square_replicate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if rgb.shape[0] != size:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    res = otsu_mask(rgb, mask_cfg)
    Path(img_out).parent.mkdir(parents=True, exist_ok=True)
    Path(mask_out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(img_out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(mask_out, res.mask)
    m = res.mask > 0
    px = rgb.reshape(-1, 3).astype(np.float64) / 255.0
    return (res.area, res.threshold, res.fallback, mean_hue(rgb, m),
            px.sum(0), (px ** 2).sum(0), len(px))


def _hue_table(df: pd.DataFrame, col: str, class_names: list[str]) -> pd.DataFrame:
    return df.pivot_table(index="dataset", columns="label", values=col, aggfunc="mean")[class_names].round(1)


# ---------------------------------------------------------------------------
def run_preprocessing(cfg: Config, verbose: bool = True) -> PreprocessResult:
    out = Path(cfg.output_dir)
    rep = out / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config_used.yaml")
    tables: dict[str, pd.DataFrame] = {}
    figs: dict[str, Path] = {}
    warns: list[str] = []
    say = print if verbose else (lambda *a, **k: None)

    # ---- 1. thu thập ----------------------------------------------------
    df, skipped = collect_images(cfg)
    collected = df.copy()
    tables["skipped"] = pd.DataFrame(
        [{"dataset": k[0], "ly_do": k[1], "so_anh": v} for k, v in skipped.most_common()])
    if cfg.eda_counts:
        eda = compare_with_eda(df, cfg.eda_counts)
        tables["eda_check"] = eda
        for _, r in eda.iterrows():
            if abs(r.lech) > max(5, 0.01 * r.eda):
                warns.append(f"{r.dataset}: thu thập {r.preprocessing} ảnh nhưng EDA đếm {r.eda} (lệch {r.lech:+d})")
    say(f"[1] Thu thập: {len(df)} ảnh")

    # ---- 2. đọc ảnh -----------------------------------------------------
    df, H, H64, T, n_err = read_fingerprints(df, cfg)
    if n_err:
        warns.append(f"{n_err} ảnh đọc lỗi đã bị bỏ")
    say(f"[2] Đọc được {len(df)} ảnh, lỗi {n_err}")

    # ---- 3. loại trùng --------------------------------------------------
    if cfg.compare_hash64:
        s64, p64 = hash64_comparison(df, H, H64, cfg.group_thr)
        tables["hash64_vs_256"] = s64
        p64.to_csv(rep / "hash64_vs_256_pairs.csv", index=False)
        disagree = p64[p64["256_noi_khac"]]
        f = pair_grid(disagree, lambda r: (f"{r.label_a} | d64={r.hamming64}", f"{r.label_b} | d256={r.hamming256}"),
                      rep / "hash64_disagreement_review.png")
        if f:
            figs["hash64_disagreement_review"] = f
        for _, r in s64.iterrows():
            if r.so_anh and r.cum_khac_nhau_dhash64_le6 < 0.8 * r.so_anh and r.trong_do_256bit_noi_khac > 0:
                warns.append(
                    f"{r.dataset}: dHash 64 bit (cách EDA) chỉ còn {r.cum_khac_nhau_dhash64_le6}/{r.so_anh} ảnh khác nhau, "
                    f"nhưng {int(r.trong_do_256bit_noi_khac)} cặp bị 256 bit coi là khác hẳn. "
                    "Mở reports/hash64_disagreement_review.png để xem đó là bản sao thật hay hash 64 bit quá thô.")

    dd = Deduplicator(df, H, H64, T)
    dd.drop_md5()
    dd.drop_near_duplicates(cfg.dup_thr)
    cross = dd.cross_label_pairs(cfg.dup_thr, cfg.suspect_color_thr)
    cross.drop(columns=["idx_a", "idx_b"]).to_csv(rep / "cross_label_pairs.csv", index=False)
    f = pair_grid(cross, lambda r: (f"{r.label_a} (Δ={r.color_diff:.1f})", f"{r.label_b}"),
                  rep / "cross_label_review.png")
    if f:
        figs["cross_label_review"] = f
    if cfg.drop_suspect_label_noise and len(cross):
        dd.drop_suspect_noise(cross)
    elif cross.suspect_noise.any():
        warns.append(f"{int(cross.suspect_noise.sum())} cặp ảnh gần như y hệt nhưng khác nhãn "
                     "(reports/cross_label_review.png). Xem tay rồi đặt drop_suspect_label_noise: true nếu cần.")
    df, H = dd.df, dd.H
    tables["dedup_log"] = pd.DataFrame(dd.log)
    removed = dd.removed_df()
    removed.to_csv(rep / "removed_images.csv", index=False)
    say(f"[3] Sau loại trùng: {len(df)} ảnh")

    # ---- 4. nhóm --------------------------------------------------------
    df["group"] = assign_groups(H, cfg.group_thr)
    gsize = df.group.value_counts()
    if gsize.iloc[0] / len(df) > 0.03:
        warns.append(f"Nhóm lớn nhất có {gsize.iloc[0]} ảnh ({gsize.iloc[0] / len(df):.1%}), "
                     f"có thể do nối chuỗi A~B~C. Hạ group_thr (đang {cfg.group_thr}).")
    say(f"[4] {gsize.size} nhóm, nhóm lớn nhất {gsize.iloc[0]} ảnh")

    # ---- 5. chia tập ----------------------------------------------------
    schemes = {"mixed": split_mixed(df, cfg)}
    if cfg.external_dataset in set(df.dataset):
        schemes["external"] = split_external(df, cfg)
    else:
        warns.append(f"Không có '{cfg.external_dataset}' nên bỏ cách chia external")
    leak = []
    for name, sp in schemes.items():
        df[f"split_{name}"] = sp.to_numpy()
        df[f"cv_fold_{name}"] = assign_cv_folds(df, sp, cfg)
        leak.append(leakage_report(df, sp, H, cfg, name))
        tables[f"split_{name}"] = pd.crosstab(sp, df.label)[cfg.class_names]
        tables[f"split_{name}_dataset"] = pd.crosstab(sp, df.dataset)
    tables["leakage"] = pd.concat(leak, ignore_index=True)
    say("[5] Chia tập xong, không rò rỉ")

    # ---- 6. lưu ảnh + mask ----------------------------------------------
    df["label_idx"] = df.label.map({c: i for i, c in enumerate(cfg.class_names)}).astype(int)
    df["image_id"] = df.dataset + "_" + df.md5.str[:12]
    assert df.image_id.is_unique
    df["image_path"] = "images/" + df.dataset + "/" + df.label + "/" + df.image_id + ".png"
    df["mask_path"] = "masks/" + df.dataset + "/" + df.label + "/" + df.image_id + ".png"
    jobs = [(src, str(out / ip), str(out / mp), cfg.image_size, cfg.mask)
            for src, ip, mp in zip(df.src_path, df.image_path, df.mask_path)]
    workers = cfg.num_workers or os.cpu_count() or 1
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(tqdm(ex.map(_process_one, jobs, chunksize=16), total=len(jobs), desc="Lưu ảnh + mask"))
    else:
        res = [_process_one(j) for j in tqdm(jobs, desc="Lưu ảnh + mask")]
    df["mask_area"] = [r[0] for r in res]
    df["mask_threshold"] = [r[1] for r in res]
    df["mask_fallback"] = [bool(r[2]) for r in res]
    df["hue_masked"] = [r[3] for r in res]
    # hue trên thumbnail 16x16 (như notebook gốc) để đối chiếu với hue trong mask
    df["hue_thumb"] = [mean_hue(t) for t in dd.T]
    say(f"[6] Đã lưu {len(df)} ảnh + mask vào {out}")

    # ---- 7. kiểm tra gộp nhãn + mask ------------------------------------
    hue_t = _hue_table(df, "hue_thumb", cfg.class_names)
    hue_m = _hue_table(df, "hue_masked", cfg.class_names)
    tables["hue_thumb"], tables["hue_masked"] = hue_t, hue_m
    for ds, r in hue_m.iterrows():
        if not r.is_monotonic_decreasing:
            warns.append(f"{ds}: hue trong mask không giảm dần Unripe→Overripe ({r.tolist()}), xem lại ánh xạ nhãn/mask")
    ms = df.groupby(["dataset", "label"]).agg(
        mask_area_tb=("mask_area", "mean"), ti_le_fallback=("mask_fallback", "mean")).round(3)
    tables["mask_stats"] = ms
    for (ds, lb), r in ms.iterrows():
        if r.ti_le_fallback > 0.10:
            warns.append(f"Mask thất bại {r.ti_le_fallback:.0%} ở {ds}/{lb} (reports/mask_qa.png)")
    figs["mask_qa"] = mask_qa_grid(df, out, rep / "mask_qa.png", cfg.class_names, seed=cfg.seed)

    # ---- 8. thống kê train ----------------------------------------------
    s = np.stack([r[4] for r in res])
    s2 = np.stack([r[5] for r in res])
    npx = np.array([r[6] for r in res])
    stats = {"class_names": cfg.class_names, "image_size": cfg.image_size, "n_total": len(df), "schemes": {}}
    for name in schemes:
        sp = df[f"split_{name}"].to_numpy()
        tr = sp == "train"
        mean = s[tr].sum(0) / npx[tr].sum()
        std = np.sqrt(s2[tr].sum(0) / npx[tr].sum() - mean ** 2)
        cnt = df.loc[tr, "label"].value_counts()
        stats["schemes"][name] = {
            "mean_rgb": mean.round(4).tolist(), "std_rgb": std.round(4).tolist(),
            "class_weight_train": {c: round(float(tr.sum() / (len(cfg.class_names) * cnt[c])), 3)
                                   for c in cfg.class_names},
            "counts": {k: int(v) for k, v in Counter(sp).items()},
        }
    stats["warnings"] = warns
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 9. bảng trước/sau + manifest -----------------------------------
    summary = pd.DataFrame({"thu_thap": collected.dataset.value_counts(),
                            "con_lai": df.dataset.value_counts()}).fillna(0).astype(int)
    if len(removed):
        summary = summary.join(removed.pivot_table(index="dataset", columns="reason", values="src_path",
                                                   aggfunc="count")).fillna(0).astype(int)
    tables["summary_before_after"] = summary

    cols = ["image_id", "dataset", "label", "label_idx", "raw_label", "image_path", "mask_path", "group"]
    cols += [c for n in schemes for c in (f"split_{n}", f"cv_fold_{n}")]
    cols += ["mask_area", "mask_threshold", "mask_fallback", "hue_thumb", "hue_masked",
             "md5", "w", "h", "src_folder", "src_path"]
    manifest = df[cols]
    manifest.to_csv(out / "manifest.csv", index=False)
    for name, t in tables.items():
        t.to_csv(rep / f"{name}.csv")

    result = PreprocessResult(output_dir=out, manifest=manifest, stats=stats,
                              tables=tables, figures=figs, warnings=warns)
    if verbose:
        result.print_summary()
    return result
