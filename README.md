# CS231 — Phân loại độ chín chuối

Ảnh một quả chuối → một trong 4 mức: `Unripe`, `Half_Ripe`, `Ripe`, `Overripe`.

| Người | Mô hình | Trạng thái |
|---|---|---|
| Luân | Histogram màu RGB/HSV (66 chiều) + SVM tuyến tính / RBF, KNN làm mốc | xong code |
| Nguyên | HOG (từng kênh R, G, B hoặc xám) + SVM tuyến tính / RBF | xong code |
| Quỳnh | ResNet18 + Grad-CAM | khung (`TODO(Quỳnh)`) |

Cả 3 mô hình dùng **chung** dữ liệu đã tiền xử lý, chung cách chia, chung hàm đánh giá và lưu kết quả cùng
định dạng, nên gộp được thành một bảng so sánh.

## Nguồn dữ liệu

Ba dataset tham khảo từ Mendeley Data (cùng giấy phép **CC BY 4.0**). Nhóm dùng bản sao trên Kaggle để chạy,
**chỉ lấy ảnh gốc** của phần độ chín; bỏ ảnh tăng cường (augmented) và ảnh tổng hợp (synthetic).

| Tên trong code | Dataset gốc (Mendeley) | Bản trên Kaggle | Ảnh độ chín gốc | Nhãn gốc → nhãn chung |
|---|---|---|---|---|
| `BananaID` | Mutrofin, S., Fatichah, C., Yuniarti, A., & Setiawan, E. (2025). *BananaID: An image dataset of banana varieties and ripeness stages in Indonesia designed for classification* (V1). [doi:10.17632/h6n5srjjyw.1](https://data.mendeley.com/datasets/h6n5srjjyw) | `anhnguyen7508/bananaid` | 1.960 (Unripe 408, Half-ripe 488, Ripe 616, Overripe 448); 384×256 JPG, chụp DSLR ở Surabaya, Indonesia | giữ nguyên |
| `BananaImageBD` | Ferdaus, M. H., Prito, R. H., Rasel, A. A. S. R., Ahmed, M., Saykot, M. J. H., Shanta, S. S., Akter, S., Das, A. C., Islam, M. M., Hasan, M., & Ali, S. (2024). *BananaImageBD: A Comprehensive Image Dataset of Common Banana Varieties with Different Ripeness Stages in Bangladesh* (V2). [doi:10.17632/ptfscwtnyz.2](https://data.mendeley.com/datasets/ptfscwtnyz) | `anhnguyen7508/bananaimagebd` | 820 (Green 212, Semi-ripe 204, Ripe 201, Overripe 203); 256×256 JPEG, chụp điện thoại | Green → Unripe, Semi-ripe → Half_Ripe |
| `BananaRipeness_GitHub` | Chuquimarca Jimenez, L. E., Vintimilla, B., & Velastin, S. (2023). *Banana Ripeness Images Datasets* (V1). [doi:10.17632/66kjghpjgy.1](https://data.mendeley.com/datasets/66kjghpjgy/1). Trang Mendeley chỉ có metadata, ảnh lấy từ [github.com/luischuquim/BananaRipeness](https://github.com/luischuquim/BananaRipeness). | `kimlunl/bananaripeness-github` | 3.495 ảnh thật (chuối Cavendish); bỏ 161.280 ảnh tổng hợp | Class A–D → Unripe, Half_Ripe, Ripe, Overripe (**xấp xỉ**) |

Bài báo đi kèm dataset thứ 3: Chuquimarca, L. E., Vintimilla, B. X., & Velastin, S. A. (2023). Banana Ripeness Level
Classification Using a Simple CNN Model Trained with Real and Synthetic Datasets. *VISIGRAPP 2023 — VISAPP*, 536–543.
doi:10.5220/0011654600003417.

Lưu ý khi báo cáo:
- Ánh xạ nhãn giữa 3 dataset là **xấp xỉ**, nhất là Class A–D của dataset thứ 3 (`src/config.py::DEFAULT_LABEL_MAPPING`).
- Không có ID từng quả chuối. "Chia theo nhóm" nghĩa là chia theo nhóm ảnh gần trùng (dHash), không phải theo quả.
- Số ảnh sau khi loại trùng xem trong `reports/summary_before_after.csv` (output preprocessing).

## Cấu trúc

```
configs/
  kaggle.yaml          đường dẫn 3 dataset trên Kaggle + tham số preprocessing
  experiments.yaml     danh sách thí nghiệm của từng người (luan / nguyen / quynh)
src/                   package Python: import bằng `from src.xxx import ...` (thêm THƯ MỤC GỐC repo vào sys.path)
  config.py            định nghĩa tham số + giá trị mặc định (giá trị cụ thể nằm ở configs/*.yaml)
  data.py              đọc manifest + ảnh + mask, find_data_root()
  preprocessing/       thu thập, loại trùng, chia tập, tách nền Otsu, lưu output
  features/            color_histogram.py (Luân), hog.py (Nguyên), cache.py
  models/              base.py (wrapper chung), svm_color.py (Luân),
                       hog_svm.py (Nguyên), resnet18.py + gradcam.py (khung Quỳnh)
  evaluation.py        accuracy, precision/recall/F1 macro, bootstrap std, ma trận nhầm lẫn
  experiment.py        run_experiment / run_grid / load_results / compare_table
scripts/               preprocess.py, train.py, compare.py (cùng code với notebook)
notebooks/             eda, preprocessing, svm_color_histogram, hog_svm, resnet18_gradcam, compare_results
                       (chỉ import và gọi hàm), legacy/ = notebook cũ để đối chiếu
tests/                 pytest trên dữ liệu giả
```

## Chạy trên Kaggle (hoặc Colab)

Mọi lần chạy thật đều trên Kaggle. Notebook nào cũng mở đầu bằng cell setup tự `git clone` repo này vào `/tmp`
(cần bật **Settings → Internet = On**).

1. **Tiền xử lý (1 lần cho cả nhóm):** `notebooks/preprocessing.ipynb`, Add Input 3 dataset
   (`anhnguyen7508/bananaid`, `anhnguyen7508/bananaimagebd`, `kimlunl/bananaripeness-github`).
   Chạy hết → Save Version → Output → **New Dataset** (vd. `banana-preprocessed`).
2. **Mỗi người chạy notebook của mình** (`svm_color_histogram`, `hog_svm`, `resnet18_gradcam`) với Add Input = dataset ở bước 1.
   Kết quả nằm ở `/kaggle/working/results/<run_name>/`. Chạy xong → Save Version.
3. **So sánh:** `notebooks/compare_results.ipynb`, Add Input = output của 3 notebook mô hình.
   `load_results()` gộp mọi `result.json` → bảng `mean ± std` cho slide.

Sửa code → push lên GitHub → chạy lại cell setup (tự `git pull`) là notebook dùng code mới.

**Không dùng git — upload thẳng code `.py`:** nén `src/` + `configs/` (+ `scripts/` nếu cần):

```bash
git archive --format=zip -o cs231-code.zip HEAD src configs scripts
```

Kaggle → Datasets → New Dataset → upload `cs231-code.zip` (Kaggle tự giải nén). Trong notebook: Add Input
dataset này cùng dataset output preprocessing. Cell setup tự tìm thư mục `src/` (package, có `__init__.py`) trong `/kaggle/input`
(Colab: upload vào `/content`), chép sang `/tmp` rồi import; không cần Internet. Đổi code thì upload
**New Version** của dataset code. Muốn ép một nguồn: đặt `CODE_SOURCE = "upload"` hoặc `"git"` trong cell setup.

## Dùng trong notebook

```python
from src.data import find_data_root, load_manifest
from src.experiment import run_experiment, run_grid, load_experiments, load_results, compare_table

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

## Tài liệu tham khảo (phương pháp và code)

Code trong repo do nhóm tự viết. Phần tiền xử lý chuyển từ notebook của nhóm sang package `.py`: notebook EDA
(`notebooks/eda.ipynb`) và notebook tiền xử lý cũ (`notebooks/legacy/cs231-r12-data-preprocessing.ipynb`, giữ để đối chiếu).
Các phương pháp lấy từ những nguồn sau:

| Bước | Phương pháp | Nguồn | File |
|---|---|---|---|
| Loại trùng, gom nhóm ảnh gần trùng | dHash (difference hash) 16×16 = 256 bit, khoảng cách Hamming; so sánh với 64 bit của notebook EDA | [3] | `src/preprocessing/collect.py`, `dedup.py` |
| Chia tập | Chia theo nhóm, phân tầng theo (nhãn × dataset); 5 fold CV trên train | [10] (`StratifiedGroupKFold`) | `src/preprocessing/split.py` |
| Đưa ảnh về 224×224 | Pad vuông bằng lặp pixel biên + resize INTER_AREA | [11] | `src/preprocessing/pipeline.py` |
| Tách nền | Ngưỡng Otsu trên kênh S của HSV, mở/đóng hình thái học, giữ thành phần liên thông lớn, lấp lỗ | [1], [2], [11] | `src/preprocessing/segmentation.py` |
| Đặc trưng màu | Histogram từng kênh RGB / HSV + mô-men thống kê của histogram (mean, std, skewness, energy, smoothness, entropy) | [4], [2] | `src/features/color_histogram.py` |
| Phân loại (Luân) | SVM tuyến tính / RBF; KNN làm mốc; chọn tham số bằng GridSearchCV theo fold nhóm, scoring macro-F1 | [5], [6], [10] | `src/models/svm_color.py`, `base.py` |
| Đặc trưng HOG (Nguyên) | Histogram of Oriented Gradients | [7], [12] | `src/features/hog.py`, `src/models/hog_svm.py` |
| CNN (Quỳnh) | ResNet18 tiền huấn luyện ImageNet, fine-tune | [8], [14], [13] | `src/models/resnet18.py` |
| Giải thích (Quỳnh) | Grad-CAM | [9] | `src/models/gradcam.py` |
| Đánh giá | accuracy, precision / recall / F1 macro, ma trận nhầm lẫn; độ lệch chuẩn bootstrap | [15], [10] | `src/evaluation.py` |

1. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
2. Smith, A. R. (1978). Color gamut transform pairs. *ACM SIGGRAPH Computer Graphics*, 12(3), 12–19. (không gian màu HSV)
3. Krawetz, N. (2013). *Kind of Like That*. The Hacker Factor Blog. https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html (dHash)
4. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson. (mô tả kết cấu bằng mô-men thống kê của histogram)
5. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
6. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.
7. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR 2005*, 886–893.
8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*, 770–778.
9. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV 2017*, 618–626.
10. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
11. Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
12. van der Walt, S., et al. (2014). scikit-image: image processing in Python. *PeerJ*, 2, e453.
13. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 2019*. (PyTorch, torchvision)
14. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR 2009*.
15. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
