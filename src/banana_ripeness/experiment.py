"""Chạy thí nghiệm + gộp kết quả. Đây là chỗ DUY NHẤT notebook cần gọi.

Một lần chạy = (mô hình + tham số) x scheme x seed. Kết quả mỗi lần chạy lưu CÙNG ĐỊNH DẠNG
dù là SVM, HOG hay ResNet18::

    <results_dir>/<run_name>/
        result.json          # tham số, tham số tốt nhất, CV, độ đo từng split (+ bootstrap std, theo dataset)
        pred_<split>.csv     # dự đoán từng ảnh (image_id, nhãn thật, nhãn đoán, xác suất nếu có)
        cm_<split>.png       # ma trận nhầm lẫn
        model.joblib|.pt     # mô hình (save_model=True)

Vì vậy mỗi người chạy notebook riêng, rồi notebook so sánh chỉ cần
``load_results(["/kaggle/input", ...])`` là có bảng chung của cả nhóm.

Ví dụ::

    from banana_ripeness.experiment import run_experiment, run_grid, load_results, compare_table
    r = run_experiment("svm_color", data_root=DATA_ROOT, scheme="external", kernel="rbf", space="hsv")
    res = run_grid([{"model": "svm_color", "kernel": ["linear", "rbf"], "space": ["rgb", "hsv"]}],
                   data_root=DATA_ROOT, schemes=["mixed", "external"])
    compare_table(load_results(), split="test")
"""
from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import CLASS_NAMES
from .data import find_data_root, get_split, labels_of, load_manifest, split_col
from .evaluation import MAIN_METRICS, bootstrap_std, compute_metrics, plot_confusion_matrix, results_table
from .models import BaseRipenessModel, build_model, load_model

RESULT_FILE = "result.json"


def default_results_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/results")
    if Path("/content").exists():
        return Path("/content/results")
    return Path("outputs/results")


# ---------------------------------------------------------------------------
@dataclass
class ExperimentResult:
    run_name: str
    run_dir: Path
    record: dict
    predictions: dict[str, pd.DataFrame] = field(default_factory=dict)
    model: BaseRipenessModel | None = None

    @property
    def metrics(self) -> dict[str, dict]:
        return self.record["splits"]

    def table(self) -> pd.DataFrame:
        return records_to_frame([self.record])

    def plot_confusion(self, split: str = "test", ax=None, normalize: bool = True):
        cm = self.metrics[split]["confusion_matrix"]
        return plot_confusion_matrix(cm, CLASS_NAMES, normalize=normalize, ax=ax,
                                     title=f"{self.run_name}\n{split}: macro-F1 {self.metrics[split]['f1_macro']:.3f}")

    def load_model(self) -> BaseRipenessModel:
        return self.model or load_model(self.run_dir)

    def __repr__(self) -> str:
        s = ", ".join(f"{k}: F1={v['f1_macro']:.3f}" for k, v in self.metrics.items())
        return f"ExperimentResult({self.run_name} | {s})"


def _as_model(model, scheme: str, seed: int | None, params: dict) -> BaseRipenessModel:
    if isinstance(model, BaseRipenessModel):
        if params:
            raise ValueError("Đã truyền đối tượng mô hình thì không truyền thêm tham số mô hình")
        model.scheme = scheme
        if seed is not None:
            model.seed = seed
        return model
    if isinstance(model, dict):
        params = {**{k: v for k, v in model.items() if k != "model"}, **params}
        model = model["model"]
    kw = {**params, "scheme": scheme}
    if seed is not None:
        kw["seed"] = seed
    return build_model(model, **kw)


def config_name_of(model: BaseRipenessModel) -> str:
    return f"{model.name}__{model.tag}__{model.scheme}"


def _jsonify(o):
    if isinstance(o, dict):
        return {str(k): _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.ndarray,)):
        return _jsonify(o.tolist())
    if isinstance(o, Path):
        return str(o)
    return o


def _prediction_frame(df: pd.DataFrame, pred: np.ndarray, proba: np.ndarray | None) -> pd.DataFrame:
    cols = [c for c in ("image_id", "dataset", "label", "label_idx", "image_path", "mask_path") if c in df]
    out = df[cols].copy()
    out["pred_idx"] = pred.astype(int)
    out["pred"] = [CLASS_NAMES[i] for i in out.pred_idx]
    out["correct"] = out.pred_idx == out.label_idx
    if proba is not None:
        for k, c in enumerate(CLASS_NAMES):
            out[f"prob_{c}"] = proba[:, k]
    return out


def evaluate_split(model: BaseRipenessModel, df: pd.DataFrame, n_boot: int = 1000, seed: int = 42
                   ) -> tuple[dict, pd.DataFrame]:
    """Độ đo + bootstrap std + độ đo theo từng dataset, và bảng dự đoán từng ảnh."""
    proba = model.predict_proba(df)
    pred = proba.argmax(1) if proba is not None else model.predict(df)
    y = labels_of(df)
    m = compute_metrics(y, pred, CLASS_NAMES)
    m.update(bootstrap_std(y, pred, len(CLASS_NAMES), n_boot=n_boot, seed=seed))
    if "dataset" in df and df["dataset"].nunique() > 1:
        m["by_dataset"] = {}
        for ds, idx in df.groupby("dataset").indices.items():
            sub = compute_metrics(y[idx], pred[idx], CLASS_NAMES)
            m["by_dataset"][ds] = {k: sub[k] for k in ("n", *MAIN_METRICS)}
    return m, _prediction_frame(df, pred, proba)


def default_eval_splits(scheme: str) -> list[str]:
    return ["val", "test", "external"] if scheme == "external" else ["val", "test"]


def run_experiment(model: str | dict | BaseRipenessModel, df: pd.DataFrame | None = None,
                   data_root: str | Path | None = None, scheme: str = "mixed", seed: int | None = None,
                   eval_splits: Sequence[str] | None = None, run_name: str | None = None,
                   results_dir: str | Path | None = None, save_model: bool = True, save_plots: bool = True,
                   n_boot: int = 1000, skip_existing: bool = False, owner: str | None = None,
                   verbose: bool = True, **model_params) -> ExperimentResult:
    """Train trên ``train`` của ``scheme``, đánh giá trên ``eval_splits``, lưu kết quả chuẩn.

    ``model``: tên trong registry (``"svm_color"``, ``"knn_color"``, ``"hog_svm"``, ``"resnet18"``),
    dict ``{"model": name, **params}``, hoặc đối tượng ``BaseRipenessModel`` đã tạo sẵn.
    ``owner``: tên người chạy (ghi vào bảng chung).
    """
    m = _as_model(model, scheme, seed, model_params)
    if df is None:
        data_root = Path(data_root) if data_root else find_data_root()
        df = load_manifest(data_root)
    m.set_data_root(data_root)
    config_name = config_name_of(m)
    run_name = run_name or f"{config_name}__s{m.seed}"
    results_dir = Path(results_dir) if results_dir else default_results_dir()
    run_dir = results_dir / run_name

    if skip_existing and (run_dir / RESULT_FILE).exists():
        if verbose:
            print(f"Bỏ qua (đã có kết quả): {run_dir}")
        return load_run(run_dir)

    if split_col(scheme) not in df:
        raise KeyError(f"manifest không có cột {split_col(scheme)}")
    train_df = get_split(df, scheme, "train")
    available = set(df[split_col(scheme)])
    eval_splits = [s for s in (eval_splits or default_eval_splits(scheme)) if s in available]
    val_df = get_split(df, scheme, "val") if "val" in available else None

    if verbose:
        print(f"=== {run_name} | train {len(train_df)} ảnh | đánh giá {eval_splits}")
    t0 = time.time()
    m.fit(train_df, val_df)
    fit_time = time.time() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    splits, preds = {}, {}
    for sp in eval_splits:
        sdf = get_split(df, scheme, sp)
        met, pdf = evaluate_split(m, sdf, n_boot=n_boot, seed=m.seed)
        splits[sp], preds[sp] = met, pdf
        pdf.drop(columns=[c for c in ("image_path", "mask_path") if c in pdf]).to_csv(
            run_dir / f"pred_{sp}.csv", index=False)
        if save_plots:
            import matplotlib.pyplot as plt
            fig = plot_confusion_matrix(met["confusion_matrix"], CLASS_NAMES,
                                        title=f"{config_name}\n{sp}: macro-F1 {met['f1_macro']:.3f}",
                                        save_path=run_dir / f"cm_{sp}.png")
            plt.close(fig)
        if verbose:
            boot = f" ± {met['f1_macro_std']:.3f}" if "f1_macro_std" in met else ""
            print(f"    {sp:9s} n={met['n']:5d}  acc {met['accuracy']:.3f}  P {met['precision_macro']:.3f}  "
                  f"R {met['recall_macro']:.3f}  macro-F1 {met['f1_macro']:.3f}{boot}")

    model_file = None
    if save_model:
        model_file = m.save(run_dir / f"model{m.file_suffix}").name

    fit_info = {k: v for k, v in m.fit_info.items()}
    record = _jsonify({
        "run_name": run_name, "config_name": config_name, "model": m.name, "model_class": type(m).__name__,
        "tag": m.tag, "scheme": scheme, "seed": m.seed, "owner": owner or os.environ.get("BANANA_OWNER"),
        "params": m.get_params(), "fit_info": fit_info, "fit_time_s": round(fit_time, 2),
        "n_train": len(train_df), "splits": splits, "model_file": model_file,
        "data_root": str(data_root) if data_root else None,
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    (run_dir / RESULT_FILE).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return ExperimentResult(run_name, run_dir, record, preds, m)


# ---------------------------------------------------------------------------
def expand_spec(spec: dict) -> list[dict]:
    """``{"model": "svm_color", "kernel": ["linear", "rbf"], "space": ["rgb", "hsv"]}`` -> 4 cấu hình.

    Giá trị dạng list được tổ hợp (tích Descartes). ``param_grid`` (dict) giữ nguyên.
    Muốn truyền 1 list làm giá trị thật thì bọc 2 lớp: ``[[16, 16]]``.
    """
    keys = list(spec)
    values = [v if isinstance(v, list) else [v] for v in spec.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_grid(specs: dict | Iterable[dict], df: pd.DataFrame | None = None, data_root: str | Path | None = None,
             schemes: Sequence[str] = ("mixed", "external"), seeds: Sequence[int] = (42,),
             skip_existing: bool = True, keep_models: bool = False, **kwargs) -> pd.DataFrame:
    """Chạy nhiều cấu hình x scheme x seed. Trả về bảng kết quả (1 dòng / lần chạy / split).

    Mỗi spec có thể tự ghi ``schemes`` / ``seeds`` riêng (vd. ResNet chạy 3 seed).
    ``skip_existing=True``: phiên Kaggle bị ngắt thì chạy lại sẽ bỏ qua các lần đã xong.
    """
    if isinstance(specs, dict):
        specs = [specs]
    if df is None:
        data_root = Path(data_root) if data_root else find_data_root()
        df = load_manifest(data_root)
    records, failed = [], []
    for spec in specs:
        spec = dict(spec)
        sp_schemes = spec.pop("schemes", schemes)
        sp_seeds = spec.pop("seeds", seeds)
        sp_schemes = [sp_schemes] if isinstance(sp_schemes, str) else sp_schemes
        sp_seeds = [sp_seeds] if isinstance(sp_seeds, int) else sp_seeds
        for cfg in expand_spec(spec):
            for scheme in sp_schemes:
                for seed in sp_seeds:
                    try:
                        r = run_experiment(cfg, df=df, data_root=data_root, scheme=scheme, seed=seed,
                                           skip_existing=skip_existing, **kwargs)
                        records.append(r.record)
                        if not keep_models:
                            r.model = None
                    except Exception as e:  # một cấu hình lỗi không làm hỏng cả lưới
                        failed.append({"spec": cfg, "scheme": scheme, "seed": seed, "error": repr(e)})
                        print(f"!! LỖI {cfg} | {scheme} | seed {seed}: {e!r}")
    if failed:
        print(f"\n!! {len(failed)} lần chạy lỗi:")
        for f in failed:
            print("   ", f)
    return records_to_frame(records)


def load_experiments(path: str | Path, who: str | None = None) -> list[dict]:
    """Đọc danh sách thí nghiệm từ YAML (``configs/experiments.yaml``). ``who``: ``luan``/``nguyen``/``quynh``."""
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    exps = data.get("experiments", data)
    if who is None:
        return [s for v in exps.values() for s in v]
    if who not in exps:
        raise KeyError(f"Không có nhóm thí nghiệm '{who}'. Có: {list(exps)}")
    return list(exps[who])


# ---------------------------------------------------------------------------
def load_run(run_dir: str | Path, with_predictions: bool = True) -> ExperimentResult:
    run_dir = Path(run_dir)
    rec = json.loads((run_dir / RESULT_FILE).read_text(encoding="utf-8"))
    preds = {}
    if with_predictions:
        for p in run_dir.glob("pred_*.csv"):
            preds[p.stem.removeprefix("pred_")] = pd.read_csv(p)
    return ExperimentResult(rec["run_name"], run_dir, rec, preds, None)


def find_runs(paths: str | Path | Sequence[str | Path] | None = None, max_depth: int = 6) -> list[Path]:
    """Mọi thư mục có ``result.json`` bên dưới ``paths`` (mặc định: results_dir + /kaggle/input)."""
    if paths is None:
        paths = [default_results_dir(), Path("/kaggle/input")]
    elif isinstance(paths, (str, Path)):
        paths = [paths]
    skip = {"images", "masks", "cache", ".git", "src", "__pycache__"}
    found = []
    for base in map(Path, paths):
        if not base.exists():
            continue
        base_depth = len(base.parts)
        for dirpath, dirnames, filenames in os.walk(base):
            if RESULT_FILE in filenames:
                found.append(Path(dirpath))
                dirnames[:] = []
                continue
            depth = len(Path(dirpath).parts) - base_depth
            dirnames[:] = [] if depth >= max_depth else [d for d in dirnames if d not in skip]
    return found


def records_to_frame(records: Iterable[dict]) -> pd.DataFrame:
    """Làm phẳng result.json -> 1 dòng / (lần chạy, split)."""
    rows = []
    for rec in records:
        base = {k: rec.get(k) for k in ("run_name", "config_name", "model", "tag", "scheme", "seed", "owner",
                                         "fit_time_s", "n_train", "created")}
        fi = rec.get("fit_info", {}) or {}
        base["best_params"] = json.dumps(fi.get("best_params")) if fi.get("best_params") is not None else None
        for m in MAIN_METRICS:
            base[f"cv_{m}_mean"] = fi.get(f"cv_{m}_mean")
            base[f"cv_{m}_std"] = fi.get(f"cv_{m}_std")
        if fi.get("best_epoch") is not None:
            base["best_epoch"] = fi.get("best_epoch")
        for k, v in (rec.get("params") or {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                base[f"param_{k}"] = v
        for sp, met in (rec.get("splits") or {}).items():
            row = {**base, "split": sp}
            for k, v in met.items():
                if isinstance(v, (int, float)) or v is None:
                    row[k] = v
            for c, pc in (met.get("per_class") or {}).items():
                row[f"f1_{c}"] = pc.get("f1")
                row[f"recall_{c}"] = pc.get("recall")
            for ds, dm in (met.get("by_dataset") or {}).items():
                row[f"f1_macro__{ds}"] = dm.get("f1_macro")
                row[f"accuracy__{ds}"] = dm.get("accuracy")
            rows.append(row)
    return pd.DataFrame(rows)


def load_results(paths: str | Path | Sequence[str | Path] | None = None) -> pd.DataFrame:
    """Gộp kết quả của MỌI người (mọi result.json tìm được) thành 1 bảng.

    Trên Kaggle: gắn output của các notebook 10/20/30 làm Input của notebook so sánh rồi gọi
    ``load_results()`` (tự tìm trong ``/kaggle/input`` và ``/kaggle/working/results``).
    Lần chạy trùng ``run_name`` chỉ giữ bản mới nhất.
    """
    recs = []
    for d in find_runs(paths):
        try:
            recs.append(json.loads((d / RESULT_FILE).read_text(encoding="utf-8")) | {"_dir": str(d)})
        except Exception as e:
            print(f"!! bỏ qua {d}: {e!r}")
    if not recs:
        print("Không tìm thấy result.json nào.")
        return pd.DataFrame()
    recs.sort(key=lambda r: r.get("created") or "")
    latest = {r["run_name"]: r for r in recs}
    df = records_to_frame(latest.values())
    dirs = {r["run_name"]: r["_dir"] for r in latest.values()}
    df["run_dir"] = df.run_name.map(dirs)
    return df


def summarize_runs(results: pd.DataFrame) -> pd.DataFrame:
    """Gộp các seed của cùng cấu hình: trung bình độ đo + ``*_seed_std`` (độ lệch giữa các seed)."""
    if results.empty:
        return results
    metric_cols = [c for c in results.columns if c in MAIN_METRICS or c.startswith(("f1_", "recall_", "accuracy__"))
                   or c.endswith("_std") or c.startswith("cv_") or c in ("f1_weighted", "n")]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(results[c])]
    keys = ["config_name", "scheme", "split"]
    first_cols = [c for c in results.columns if c not in metric_cols and c not in keys]
    g = results.groupby(keys, sort=False)
    out = g[metric_cols].mean()
    for m in MAIN_METRICS:
        out[f"{m}_seed_std"] = g[m].std(ddof=1)
    out["n_seeds"] = g["seed"].nunique()
    out = out.join(g[first_cols].first()).reset_index()
    out["run_name"] = out["config_name"]
    return out


def best_run(results: pd.DataFrame, scheme: str, by_split: str = "val", metric: str = "f1_macro",
             model: str | None = None) -> ExperimentResult:
    """Lần chạy tốt nhất theo ``metric`` trên ``by_split`` (mặc định **val**, KHÔNG chọn theo test)."""
    r = results[(results.scheme == scheme) & (results.split == by_split)]
    if model is not None:
        r = r[r.model == model]
    if r.empty:
        raise ValueError(f"Không có kết quả scheme={scheme}, split={by_split}, model={model}")
    row = r.sort_values([metric, "accuracy"], ascending=False).iloc[0]
    if "run_dir" in row and pd.notna(row.run_dir):
        return load_run(row.run_dir)
    raise ValueError("results thiếu cột run_dir, hãy dùng load_results(...)")


def compare_table(results: pd.DataFrame, split: str = "test", scheme: str | None = None,
                  metrics: Sequence[str] = MAIN_METRICS, std: str = "auto", digits: int = 3,
                  save_path: str | Path | None = None) -> pd.DataFrame:
    """Bảng "mean ± std" cho slide, gộp seed, sắp theo macro-F1 giảm dần."""
    r = results if scheme is None else results[results.scheme == scheme]
    s = summarize_runs(r)
    extra = [c for c in ("tag", "owner", "n_seeds", "best_params") if c in s]
    t = results_table(s, split=split, metrics=metrics, std=std, digits=digits, extra_cols=extra)
    front = ["model", "tag", "scheme"]
    t = t[[c for c in front if c in t] + [c for c in t if c not in front and c != "run_name"]]
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(save_path, index=False)
    return t


def plot_comparison(results: pd.DataFrame, metric: str = "f1_macro", split: str = "test",
                    scheme: str | None = None, std: str = "auto", ax=None, save_path: str | Path | None = None,
                    top: int | None = None):
    """Biểu đồ thanh ngang so sánh các cấu hình (thanh lỗi = std)."""
    import matplotlib.pyplot as plt
    r = results if scheme is None else results[results.scheme == scheme]
    s = summarize_runs(r)
    s = s[s.split == split].sort_values(metric, ascending=True)
    if top:
        s = s.tail(top)
    if std == "auto":
        err = s[f"{metric}_seed_std"].where(s.n_seeds > 1, s.get(f"{metric}_std"))
    elif std == "cv":
        err = s.get(f"cv_{metric}_std")
    elif std == "seed":
        err = s[f"{metric}_seed_std"]
    else:
        err = s.get(f"{metric}_std")
    labels = (s.model + " | " + s.tag + (" | " + s.scheme if scheme is None else "")).tolist()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(s) + 1.2))
    else:
        fig = ax.figure
    models = sorted(s.model.unique())
    cmap = plt.get_cmap("tab10")
    colors = [cmap(models.index(m) % 10) for m in s.model]
    ax.barh(range(len(s)), s[metric], xerr=None if err is None else err.fillna(0), color=colors, capsize=3)
    ax.set_yticks(range(len(s)), labels, fontsize=8)
    ax.set_xlabel(f"{metric} ({split})")
    ax.set_xlim(max(0.0, float(s[metric].min()) - 0.1) if len(s) else 0, 1.0)
    ax.grid(axis="x", alpha=0.3)
    if own:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
def prediction_grid(pred_df: pd.DataFrame, n: int = 8, true: str | None = None, pred: str | None = None,
                    correct: bool | None = None, seed: int = 42, ncols: int = 4, use_mask: bool = False,
                    data_root: str | Path | None = None, save_path: str | Path | None = None):
    """Lưới ảnh demo từ bảng dự đoán, lọc theo nhãn thật / nhãn đoán / đúng-sai.

    Vd. các ảnh Half_Ripe bị đoán thành Ripe: ``prediction_grid(p, true="Half_Ripe", pred="Ripe")``.
    ``pred_df`` từ ``ExperimentResult.predictions[split]`` (có image_path) hoặc ``pred_<split>.csv``
    (cần ``data_root`` để ghép lại đường dẫn ảnh).
    """
    import matplotlib.pyplot as plt

    from .data import apply_mask, read_mask, read_rgb
    p = pred_df
    if true is not None:
        p = p[p.label == true]
    if pred is not None:
        p = p[p.pred == pred]
    if correct is not None:
        p = p[p.correct == correct]
    if "image_path" not in p:
        if data_root is None:
            raise ValueError("pred_df không có image_path, cần truyền data_root")
        man = load_manifest(data_root).set_index("image_id")[["image_path", "mask_path"]]
        p = p.join(man, on="image_id")
    p = p.sample(min(n, len(p)), random_state=seed) if len(p) else p
    if p.empty:
        print("Không có ảnh nào thoả điều kiện lọc.")
        return None
    nrows = int(np.ceil(len(p) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    for ax, (_, r) in zip(axes, p.iterrows()):
        img = read_rgb(r.image_path)
        if use_mask:
            img = apply_mask(img, read_mask(r.mask_path))
        ax.imshow(img)
        conf = f" ({max(r[c] for c in prob_cols):.2f})" if all(c in r for c in prob_cols) else ""
        ax.set_title(f"{r.get('dataset', '')[:14]}\nthật {r.label} | đoán {r.pred}{conf}", fontsize=7,
                     color="green" if r.correct else "red")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


__all__ = ["ExperimentResult", "run_experiment", "run_grid", "expand_spec", "load_experiments", "evaluate_split",
           "load_run", "find_runs", "records_to_frame", "load_results", "summarize_runs", "best_run", "compare_table",
           "plot_comparison", "prediction_grid", "default_results_dir"]
