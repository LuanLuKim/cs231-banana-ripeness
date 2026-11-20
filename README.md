# CS231 — Phân loại độ chín chuối

Ảnh một quả chuối → một trong 4 mức: `Unripe`, `Half_Ripe`, `Ripe`, `Overripe`.

| Người | Mô hình | Trạng thái |
|---|---|---|
| Luân | Histogram màu RGB/HSV (66 chiều) + SVM tuyến tính / RBF, KNN làm mốc | xong code |
| Nguyên | HOG + SVM | khung (`TODO(Nguyên)`) |
| Quỳnh | ResNet18 + Grad-CAM | khung (`TODO(Quỳnh)`) |

Cả 3 mô hình dùng **chung** dữ liệu đã tiền xử lý, chung cách chia, chung hàm đánh giá và lưu kết quả cùng
định dạng, nên gộp được thành một bảng so sánh.

## Cấu trúc

```
configs/
  kaggle.yaml          đường dẫn 3 dataset trên Kaggle + tham số preprocessing
  experiments.yaml     danh sách thí nghiệm của từng người (luan / nguyen / quynh)
src/banana_ripeness/
  config.py            tham số chung
  data.py              đọc manifest + ảnh + mask, find_data_root()
  preprocessing/       thu thập, loại trùng, chia tập, tách nền Otsu, lưu output
  features/            color_histogram.py (Luân), hog.py (khung Nguyên), cache.py
  models/              base.py (wrapper chung), svm_color.py (Luân),
                       hog_svm.py (khung Nguyên), resnet18.py + gradcam.py (khung Quỳnh)
  evaluation.py        accuracy, precision/recall/F1 macro, bootstrap std, ma trận nhầm lẫn
  experiment.py        run_experiment / run_grid / load_results / compare_table
scripts/               preprocess.py, train.py, compare.py (cùng code với notebook)
notebooks/             01_eda, 02_preprocessing, 10_svm, 20_hog, 30_resnet18, 90_compare
                       (chỉ import và gọi hàm), legacy/ = notebook cũ để đối chiếu
tests/                 pytest trên dữ liệu giả
```

## Chạy trên Kaggle (hoặc Colab)

Mọi lần chạy thật đều trên Kaggle. Notebook nào cũng mở đầu bằng cell setup tự `git clone` repo này vào `/tmp`
(cần bật **Settings → Internet = On**).

1. **Tiền xử lý (1 lần cho cả nhóm):** `notebooks/02_preprocessing.ipynb`, Add Input 3 dataset
   (`anhnguyen7508/bananaid`, `anhnguyen7508/bananaimagebd`, `kimlunl/bananaripeness-github`).
   Chạy hết → Save Version → Output → **New Dataset** (vd. `banana-preprocessed`).
2. **Mỗi người chạy notebook của mình** (`10_…`, `20_…`, `30_…`) với Add Input = dataset ở bước 1.
   Kết quả nằm ở `/kaggle/working/results/<run_name>/`. Chạy xong → Save Version.
3. **So sánh:** `notebooks/90_compare_results.ipynb`, Add Input = output của các notebook 10/20/30.
   `load_results()` gộp mọi `result.json` → bảng `mean ± std` cho slide.

Sửa code → push lên GitHub → chạy lại cell setup (tự `git pull`) là notebook dùng code mới.

**Không dùng git — upload thẳng code `.py`:** nén `src/` + `configs/` (+ `scripts/` nếu cần):

```bash
git archive --format=zip -o cs231-code.zip HEAD src configs scripts
```

Kaggle → Datasets → New Dataset → upload `cs231-code.zip` (Kaggle tự giải nén). Trong notebook: Add Input
dataset này cùng dataset output preprocessing. Cell setup tự tìm `src/banana_ripeness/` trong `/kaggle/input`
(Colab: upload vào `/content`), chép sang `/tmp` rồi import; không cần Internet. Đổi code thì upload
**New Version** của dataset code. Muốn ép một nguồn: đặt `CODE_SOURCE = "upload"` hoặc `"git"` trong cell setup.

## Dùng trong notebook

```python
from banana_ripeness.data import find_data_root, load_manifest
from banana_ripeness.experiment import run_experiment, run_grid, load_experiments, load_results, compare_table

DATA_ROOT = find_data_root()                    # tự tìm manifest.csv trong /kaggle/input
df = load_manifest(DATA_ROOT)

# 1 mô hình
r = run_experiment("svm_color", df=df, data_root=DATA_ROOT, scheme="external", kernel="rbf", space="hsv")
r.metrics["external"]["f1_macro"]; r.plot_confusion("external")

# cả danh sách thí nghiệm trong configs/experiments.yaml
res = run_grid(load_experiments("configs/experiments.yaml", "luan"), df=df, data_root=DATA_ROOT)

# bảng chung của cả nhóm
compare_table(load_results(), split="test", scheme="mixed")
```

Tên mô hình: `svm_color`, `knn_color`, `hog_svm`, `resnet18` (xem `models/__init__.py`).

**Cách chia (`scheme`):** `mixed` = gộp 3 dataset rồi chia train/val/test; `external` = train/val/test từ
BananaID + GitHub, BananaImageBD làm tập **kiểm định ngoài**. Chia theo nhóm ảnh gần trùng (dHash), không có rò rỉ.

**Chọn tham số:** GridSearchCV trên 5 fold theo nhóm của **tập train** (`cv_fold_<scheme>`), scoring macro-F1.
**Std:** bootstrap 1000 lần trên tập đánh giá; std 5 fold CV; std giữa các seed (ResNet).

## Kết quả một lần chạy

```
results/<model>__<tag>__<scheme>__s<seed>/
  result.json       tham số, tham số tốt nhất, CV mean/std, độ đo từng split (+ bootstrap std, theo dataset, theo lớp)
  pred_<split>.csv  dự đoán từng ảnh
  cm_<split>.png    ma trận nhầm lẫn
  model.joblib|.pt  mô hình
```

## Thêm / hoàn thiện một mô hình

Kế thừa `BaseRipenessModel` (cài `fit`, `predict_proba` hoặc `predict`, `save`, `load`, `get_params`, `tag`),
hoặc `ImageFeatureModel` nếu là đặc trưng thủ công + sklearn (chỉ cần `make_extractor`, `make_estimator`,
`default_param_grid`). Đăng ký tên trong `models/__init__.py::REGISTRY`, khai báo thí nghiệm trong
`configs/experiments.yaml`. Ví dụ hoàn chỉnh: `models/svm_color.py`.

## Chạy bằng dòng lệnh

```bash
python scripts/preprocess.py --config configs/kaggle.yaml
python scripts/train.py --who luan
python scripts/train.py --model svm_color --set kernel=linear,rbf space=hsv --schemes external
python scripts/compare.py /kaggle/input /kaggle/working/results --out reports
```

## Test (trên máy, dữ liệu giả)

```bash
pip install -r requirements.txt
pytest -q
```
