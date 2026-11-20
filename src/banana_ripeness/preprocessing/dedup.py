"""Bước 3-4: loại trùng lặp và gom nhóm ảnh gần giống.

Quy tắc (giữ nguyên notebook gốc):

- MD5 trùng mà khác nhãn: xoá cả nhóm (không biết nhãn nào đúng).
- MD5 trùng cùng nhãn: giữ 1.
- dHash gần giống (<= ``dup_thr`` bit) VÀ cùng nhãn: giữ 1 ảnh độ phân giải cao nhất.
- dHash gần giống nhưng KHÁC nhãn: không tự xoá, xuất ra để xem tay.
- Gom nhóm (<= ``group_thr`` bit, bất kể nhãn): mọi ảnh trong 1 nhóm nằm cùng 1 split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def hamming_pairs(H: np.ndarray, max_bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mọi cặp (i, j, số bit khác) với i < j và khoảng cách <= ``max_bits``."""
    n_bits = H.shape[1]
    if len(H) < 2:
        e = np.array([], dtype=int)
        return e, e, e
    nn = NearestNeighbors(radius=max_bits / n_bits + 1e-9, metric="hamming",
                          algorithm="brute", n_jobs=-1).fit(H)
    dist, idx = nn.radius_neighbors(H)
    I, J, D = [], [], []
    for i, (d, nb) in enumerate(zip(dist, idx)):
        m = nb > i
        I.append(np.full(int(m.sum()), i))
        J.append(nb[m])
        D.append(np.rint(d[m] * n_bits).astype(int))
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def components(n: int, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Union-find: id cụm cho từng phần tử."""
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(I, J):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(n)])


class Deduplicator:
    """Giữ df, H, H64, T đồng bộ khi loại ảnh và ghi log ảnh bị loại."""

    def __init__(self, df: pd.DataFrame, H: np.ndarray, H64: np.ndarray, T: np.ndarray):
        self.df, self.H, self.H64, self.T = df, H, H64, T
        self.removed: list[pd.DataFrame] = []
        self.log: list[dict] = []

    def drop(self, mask: np.ndarray, reason: str) -> None:
        mask = np.asarray(mask, bool)
        if mask.any():
            self.removed.append(
                self.df.loc[mask, ["dataset", "src_path", "label", "md5"]].assign(reason=reason))
        self.df = self.df[~mask].reset_index(drop=True)
        self.H, self.H64, self.T = self.H[~mask], self.H64[~mask], self.T[~mask]
        self.log.append({"buoc": reason, "loai": int(mask.sum()), "con_lai": len(self.df)})

    def drop_md5(self) -> None:
        n_lab = self.df.groupby("md5")["label"].transform("nunique").to_numpy()
        self.drop(n_lab > 1, "md5_trung_khac_nhan")
        self.drop(self.df.duplicated("md5").to_numpy(), "md5_trung_cung_nhan")

    def drop_near_duplicates(self, dup_thr: int) -> None:
        I, J, _ = hamming_pairs(self.H, dup_thr)
        lab = self.df["label"].to_numpy()
        same = lab[I] == lab[J]
        comp = components(len(self.df), I[same], J[same])
        order = pd.DataFrame({"comp": comp, "area": (self.df.w * self.df.h).to_numpy()}).sort_values(
            ["comp", "area"], ascending=[True, False], kind="stable")
        keep = order.drop_duplicates("comp").index.to_numpy()
        self.drop(~np.isin(np.arange(len(self.df)), keep), "near_dup_cung_nhan")

    def cross_label_pairs(self, dup_thr: int, suspect_color_thr: float) -> pd.DataFrame:
        """Cặp gần giống nhưng khác nhãn. ``color_diff`` = phân vị 95 của |chênh lệch pixel|
        trên thumbnail 16x16 (không dùng trung bình vì nền trắng làm loãng khác biệt)."""
        I, J, D = hamming_pairs(self.H, dup_thr)
        lab = self.df["label"].to_numpy()
        cr = lab[I] != lab[J]
        I, J, D = I[cr], J[cr], D[cr]
        if len(I):
            pix = np.abs(self.T[I].astype(int) - self.T[J].astype(int)).mean(axis=3).reshape(len(I), -1)
            color_diff = np.percentile(pix, 95, axis=1)
        else:
            color_diff = np.array([])
        out = pd.DataFrame({
            "idx_a": I, "idx_b": J,
            "path_a": self.df.src_path.to_numpy()[I], "path_b": self.df.src_path.to_numpy()[J],
            "dataset_a": self.df.dataset.to_numpy()[I], "dataset_b": self.df.dataset.to_numpy()[J],
            "label_a": lab[I], "label_b": lab[J], "hamming": D, "color_diff": color_diff,
        })
        out["suspect_noise"] = out.color_diff < suspect_color_thr
        return out.sort_values("color_diff").reset_index(drop=True)

    def drop_suspect_noise(self, cross: pd.DataFrame) -> None:
        bad = set(cross.loc[cross.suspect_noise, "idx_a"]) | set(cross.loc[cross.suspect_noise, "idx_b"])
        self.drop(np.isin(np.arange(len(self.df)), list(bad)), "nghi_nhan_sai")

    def removed_df(self) -> pd.DataFrame:
        if self.removed:
            return pd.concat(self.removed, ignore_index=True)
        return pd.DataFrame(columns=["dataset", "src_path", "label", "md5", "reason"])


def assign_groups(H: np.ndarray, group_thr: int) -> np.ndarray:
    I, J, _ = hamming_pairs(H, group_thr)
    return components(len(H), I, J)


def hash64_comparison(df: pd.DataFrame, H: np.ndarray, H64: np.ndarray,
                      group_thr: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """So sánh dHash 64 bit (cách EDA đếm 'ảnh thực sự khác nhau') với dHash 256 bit.

    EDA báo BananaID chỉ còn 888/1960 ảnh khác nhau (64 bit, <= 6 bit), trong khi
    tiền xử lý 256 bit gần như không loại ảnh nào của BananaID. Bảng này giải thích độ lệch:
    các cặp '64 bit nói trùng, 256 bit nói khác hẳn' là ứng viên để xem tay xem
    đó là bản sao thật hay chỉ là ảnh nền trắng có hash 64 bit giống nhau.
    """
    I64, J64, D64 = hamming_pairs(H64, 6)
    comp64 = components(len(df), I64, J64)
    summary = []
    for ds, sub in df.groupby("dataset"):
        idx = sub.index.to_numpy()
        summary.append({
            "dataset": ds, "so_anh": len(idx),
            "cum_khac_nhau_dhash64_le6": int(len(np.unique(comp64[idx]))),
        })
    summary = pd.DataFrame(summary)
    if len(I64):
        d256 = np.count_nonzero(H[I64] != H[J64], axis=1)
    else:
        d256 = np.array([], dtype=int)
    pairs = pd.DataFrame({
        "path_a": df.src_path.to_numpy()[I64], "path_b": df.src_path.to_numpy()[J64],
        "dataset_a": df.dataset.to_numpy()[I64], "dataset_b": df.dataset.to_numpy()[J64],
        "label_a": df.label.to_numpy()[I64], "label_b": df.label.to_numpy()[J64],
        "hamming64": D64, "hamming256": d256,
    })
    pairs["256_noi_khac"] = pairs.hamming256 > group_thr
    disagree = pairs.groupby("dataset_a")["256_noi_khac"].agg(["size", "sum"]).rename(
        columns={"size": "cap_trung_64bit", "sum": "trong_do_256bit_noi_khac"})
    summary = summary.merge(disagree, left_on="dataset", right_index=True, how="left").fillna(0)
    return summary, pairs.sort_values("hamming256", ascending=False).reset_index(drop=True)
