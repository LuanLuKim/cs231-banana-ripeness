"""Kiểm thử mô hình + runner chung + gộp kết quả (dữ liệu giả, lưới tham số nhỏ cho nhanh)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import get_split, labels_of, load_manifest
from src.experiment import (compare_table, expand_spec, load_experiments, load_results, load_run,
                                        plot_comparison, prediction_grid, run_experiment, run_grid)
from src.models import REGISTRY, BaseRipenessModel, build_model, get_model_class, load_model

EXPERIMENTS = Path(__file__).resolve().parents[1] / "configs" / "experiments.yaml"


@pytest.fixture(scope="module")
def data(prepped):
    cfg, _ = prepped
    return cfg.output_dir, load_manifest(cfg.output_dir)


def test_expand_spec():
    specs = expand_spec({"model": "svm_color", "kernel": ["linear", "rbf"], "space": ["rgb", "hsv"],
                         "param_grid": {"C": [1]}})
    assert len(specs) == 4 and all(s["param_grid"] == {"C": [1]} for s in specs)


def test_experiments_yaml_valid():
    for who in ("luan", "nguyen", "quynh"):
        specs = load_experiments(EXPERIMENTS, who)
        assert specs
        for s in specs:
            for cfg in expand_spec({k: v for k, v in s.items() if k not in ("schemes", "seeds")}):
                build_model(cfg.pop("model"), **cfg)          # tham số phải hợp lệ
    assert len([c for s in load_experiments(EXPERIMENTS, "luan") for c in expand_spec(s)]) == 8 + 4


def test_svm_color_end_to_end(data, tmp_path):
    root, df = data
    r = run_experiment("svm_color", df=df, data_root=root, scheme="external", kernel="rbf", space="hsv",
                       param_grid={"C": [1, 10], "gamma": ["scale"]}, results_dir=tmp_path, n_jobs=1,
                       cache_dir=tmp_path / "cache", n_boot=50, owner="test", verbose=False)
    assert set(r.metrics) == {"val", "test", "external"}
    assert r.metrics["test"]["f1_macro"] > 0.8                    # dữ liệu giả rất dễ
    assert set(r.metrics["test"]["by_dataset"]) == {"BananaID", "BananaRipeness_GitHub"}
    rec = json.loads((r.run_dir / "result.json").read_text(encoding="utf-8"))
    assert rec["fit_info"]["best_params"]["C"] in (1, 10)
    assert 0 <= rec["fit_info"]["cv_f1_macro_std"] < 0.5
    assert rec["fit_info"]["n_features"] == 66
    assert (r.run_dir / "pred_test.csv").exists() and (r.run_dir / "cm_test.png").exists()
    # nạp lại mô hình -> dự đoán giống hệt
    m2 = load_model(r.run_dir)
    test = get_split(df, "external", "test")
    np.testing.assert_array_equal(m2.predict(test), r.model.predict(test))
    assert len(r.model.cv_results()) == 2


def test_grid_and_shared_results(data, tmp_path):
    root, df = data
    common = dict(df=df, data_root=root, results_dir=tmp_path, n_boot=20, verbose=False)
    cache = str(tmp_path / "c")
    run_grid([{"model": "knn_color", "space": ["rgb", "hsv"], "use_mask": [True, False],
               "param_grid": {"n_neighbors": [1, 3]}, "n_jobs": 1, "cache_dir": cache},
              {"model": "svm_color", "kernel": "linear", "param_grid": {"C": [1]}, "n_jobs": 1,
               "cache_dir": cache, "schemes": ["mixed"]}],
             schemes=["mixed", "external"], owner="luan", **common)
    res = load_results(tmp_path)
    assert set(res.model) == {"knn_color", "svm_color"}
    assert res.run_name.nunique() == 4 * 2 + 1
    assert {"val", "test", "external"} <= set(res.split)
    t = compare_table(res, split="test", scheme="mixed")
    assert len(t) == 5 and "±" in t.f1_macro.iloc[0] and list(t.columns[:3]) == ["model", "tag", "scheme"]
    assert "±" in compare_table(res, split="test", std="cv").f1_macro.iloc[0]
    plot_comparison(res, split="test", scheme="mixed")
    # chạy lại -> bỏ qua lần đã có (resume khi phiên Kaggle bị ngắt)
    before = sorted(load_results(tmp_path).created.tolist())
    run_grid({"model": "knn_color", "space": "rgb", "use_mask": True, "param_grid": {"n_neighbors": [1, 3]},
              "n_jobs": 1, "cache_dir": cache}, schemes=["mixed"], **common)
    assert sorted(load_results(tmp_path).created.tolist()) == before
    run = load_run(res.run_dir.iloc[0])
    assert prediction_grid(run.predictions["test"], n=4, data_root=root) is not None


def test_teammate_skeletons_registered_but_not_implemented(data, tmp_path):
    root, df = data
    for name in ("hog_svm", "resnet18"):
        assert issubclass(get_model_class(name), BaseRipenessModel)
    # khung chưa cài đặt -> run_grid báo lỗi từng cấu hình nhưng không dừng cả lưới
    out = run_grid([{"model": "hog_svm"}, {"model": "resnet18", "epochs": 1}], df=df, data_root=root,
                   schemes=["mixed"], results_dir=tmp_path, verbose=False)
    assert out.empty
    m = build_model("resnet18", scheme="external")
    m.set_data_root(root)
    mean, std = m.train_mean_std()
    assert len(mean) == 3 and mean != [0.485, 0.456, 0.406]       # đọc được stats.json
    assert len(m.train_class_weight()) == 4


class _MajorityModel(BaseRipenessModel):
    """Mô hình đồ chơi: luôn đoán lớp đông nhất. Chứng minh khung: chỉ cần fit/predict_proba/save/load."""
    name = "majority"

    def fit(self, train_df, val_df=None):
        self.p_ = np.bincount(labels_of(train_df), minlength=4) / len(train_df)
        self.fit_info["history"] = [{"epoch": 1}]
        return self

    def predict_proba(self, df):
        return np.tile(self.p_, (len(df), 1))

    def save(self, path):
        Path(path).write_text(json.dumps(self.p_.tolist()))
        return Path(path)

    @classmethod
    def load(cls, path):
        m = cls()
        m.p_ = np.array(json.loads(Path(path).read_text()))
        return m

    def get_params(self):
        return {"scheme": self.scheme, "seed": self.seed}


def test_custom_model_joins_shared_table(data, tmp_path, monkeypatch):
    root, df = data
    monkeypatch.setitem(REGISTRY, "majority", (__name__, "_MajorityModel"))
    run_experiment(_MajorityModel(), df=df, data_root=root, scheme="mixed", results_dir=tmp_path / "quynh",
                   n_boot=10, verbose=False)
    run_experiment("svm_color", df=df, data_root=root, scheme="mixed", kernel="linear", param_grid={"C": [1]},
                   n_jobs=1, cache_dir=tmp_path / "c", results_dir=tmp_path / "luan", n_boot=10, verbose=False)
    res = load_results([tmp_path / "quynh", tmp_path / "luan"])
    t = compare_table(res, split="test")
    assert list(t.model) == ["svm_color", "majority"]             # sắp theo macro-F1 giảm dần
    assert isinstance(load_model(res[res.model == "majority"].run_dir.iloc[0]), _MajorityModel)
