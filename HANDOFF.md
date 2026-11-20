# HANDOFF — tiếp tục dự án CS231 (phân loại độ chín chuối)

> File này để Claude (bản desktop, có quyền vào thư mục máy) đọc và làm tiếp.
> Người dùng: **Luân** (Kim Luân), nhóm 3 người. Trả lời bằng **tiếng Việt**.
> Thư mục làm việc trên máy: `D:\Luan\Study\CS231\DoAn\` (repo này nằm trong đó).
> Tài khoản GitHub: **LuanLuKim**, tên repo đề xuất: `cs231-banana-ripeness`.

## 1. Bối cảnh

- Đề tài: ảnh 1 quả chuối → 1 trong 4 mức `Unripe, Half_Ripe, Ripe, Overripe`.
- 3 dataset trên Kaggle:
  - `anhnguyen7508/bananaid` (BananaID)
  - `anhnguyen7508/bananaimagebd` (BananaImageBD)
  - `kimlunl/bananaripeness-github` (BananaRipeness_GitHub)
- Phân công (mỗi người một notebook, cùng dùng output của preprocessing):
  - **Luân**: SVM tuyến tính + SVM RBF trên histogram màu RGB (đối chứng HSV), có/không tách nền, KNN làm mốc.
  - **Nguyên**: HOG ("chia bin theo màu RGB", để Nguyên tự chốt cách làm).
  - **Quỳnh**: ResNet18 + Grad-CAM.
- Độ đo trên slide: accuracy, precision, recall, macro-F1, kèm **độ lệch chuẩn**. Tham số phải **chọn trong tập train**.
- Dàn ý slide: (1) bìa; (2) lý do; (3) bài toán; (4) sơ đồ pipeline; (5) tiền xử lý + đặc trưng (Otsu, histogram HSV/RGB, thống kê mean/skewness/energy/smoothness, **ghi rõ nguồn, số bin, số chiều**); (6) bộ phân loại; (7) dataset + thiết lập (BananaID chính, BananaImageBD kiểm định ngoài); (8) kết quả có ± std, nói rõ ánh xạ nhãn là xấp xỉ; (9) nhận xét, nhầm Half_Ripe ↔ Ripe; (10) tài liệu tham khảo.

## 2. Quyết định đã chốt

1. **`.py` là nguồn chính, notebook chỉ import và gọi.** EDA vẫn là notebook (cần hình ngay trong file).
2. Preprocessing là **package** `src/banana_ripeness/preprocessing`, chạy bằng `scripts/preprocess.py --config ...`
   hoặc gọi `run_preprocessing(cfg)` trong notebook. Cấu hình nằm ở `configs/*.yaml`, không hard-code.
3. Output preprocessing **lưu thành 1 Kaggle Dataset** để cả nhóm gắn vào, dùng **chung một cách chia**.
4. **Dữ liệu chia theo manifest** (`manifest.csv`), KHÔNG chia theo cấu trúc thư mục. Có 2 cách chia:
   - `split_mixed`: gộp 3 dataset.
   - `split_external`: BananaImageBD = `external` (kiểm định ngoài), train/val/test lấy từ 2 dataset còn lại.
   - Mỗi cách chia có thêm `cv_fold_*` (5 fold theo nhóm, trên train) để chọn tham số và tính std.
5. Mỗi ảnh có **mask Otsu** lưu sẵn (trên kênh S của HSV), để 3 người dùng chung một mask.
6. Mô hình: **một lớp wrapper chung** `BaseRipenessModel` (fit / predict / evaluate / save / load), dữ liệu
   vào là DataFrame của manifest. Có 3 nhánh kế thừa: color-histogram SVM/KNN, HOG, ResNet18.

## 3. Đã làm xong (có trong repo này)

| File | Trạng thái |
|---|---|
| `src/banana_ripeness/config.py` | xong: dataclass `Config`/`MaskConfig`, `load_config(yaml, **override)` |
| `configs/kaggle.yaml`, `configs/local.yaml` | xong |
| `src/banana_ripeness/data.py` | xong: `load_manifest`, `get_split(df, scheme, split)`, `read_rgb`, `read_mask`, `apply_mask` |
| `preprocessing/collect.py` | xong: quét thư mục, ánh xạ nhãn, đối chiếu số ảnh với EDA, đọc MD5 + dHash 256 bit + dHash 64 bit + thumbnail |
| `preprocessing/dedup.py` | xong: loại trùng MD5 / near-dup cùng nhãn, cặp khác nhãn để xem tay, gom nhóm, **so sánh hash 64 bit với 256 bit** |
| `preprocessing/split.py` | xong: chia mixed + external, cv_fold, kiểm tra rò rỉ (assert) |
| `preprocessing/segmentation.py` | xong: `otsu_mask` (S-channel → blur → Otsu → mở/đóng → giữ thành phần lớn → lấp lỗ → fallback) |
| `preprocessing/reports.py` | xong: lưới cặp ảnh, lưới QA mask |
| `preprocessing/pipeline.py` | xong: `run_preprocessing(cfg)` → `PreprocessResult` (manifest, stats, bảng, hình, **danh sách cảnh báo**) |
| `scripts/preprocess.py` | xong (trả mã thoát 1 nếu có cảnh báo) |
| `tests/test_preprocessing.py` | xong, **2 test pass** trên dữ liệu giả |

Output của preprocessing (`cfg.output_dir`):
```
manifest.csv  stats.json  config_used.yaml
images/<dataset>/<label>/<image_id>.png    # 224x224, pad replicate + resize INTER_AREA
masks/<dataset>/<label>/<image_id>.png     # 0/255
reports/  eda_check, skipped, dedup_log, removed_images, cross_label_pairs(+png),
          hash64_vs_256(+pairs csv, png), split_*, leakage, hue_thumb, hue_masked,
          mask_stats, mask_qa.png, summary_before_after
```
Cột manifest: `image_id, dataset, label, label_idx, raw_label, image_path, mask_path, group,
split_mixed, cv_fold_mixed, split_external, cv_fold_external, mask_area, mask_threshold,
mask_fallback, hue_thumb, hue_masked, md5, w, h, src_folder, src_path`.

## 4. Việc còn lại (theo thứ tự)

1. **Dựng repo trên máy**: chép thư mục này vào `D:\Luan\Study\CS231\DoAn\` (thư mục cũ tên `banana`; được phép đổi tên
   thành `cs231-banana-ripeness`). Chuyển notebook EDA cũ thành `notebooks/01_eda.ipynb`, notebook preprocessing cũ thành
   `notebooks/legacy/cs231-r12-data-preprocessing.ipynb` (giữ để đối chiếu, không xoá).
   Tạo repo GitHub: `gh repo create LuanLuKim/cs231-banana-ripeness --private --source . --push`
   (người dùng tự đăng nhập `gh auth login`, Claude không nhập mật khẩu hay token).
2. **`notebooks/02_preprocessing.ipynb`**: cell setup (thêm `src/` vào `sys.path`, hoặc `git clone` trên Kaggle),
   `run_preprocessing(load_config("configs/kaggle.yaml"))`, hiển thị `result.tables[...]`, các ảnh trong
   `result.figures`, in `result.warnings`.
3. **`src/banana_ripeness/features/color_histogram.py`**: `color_histogram(rgb, mask=None, space="rgb"|"hsv", bins=16)`
   - histogram 1D mỗi kênh trên pixel trong mask, chuẩn hoá tổng = 1 → 3×bins chiều (16 bin → 48 chiều).
   - Thống kê mỗi kênh, tính từ histogram 256 bin với z ∈ [0,1]: mean, std, **skewness** (μ3/σ³), **energy/uniformity**
     Σp², **smoothness** R = 1 − 1/(1+σ²), entropy → 6×3 = 18 chiều. Tổng 66 chiều. Nguồn: Gonzalez & Woods,
     *Digital Image Processing*, mục mô tả kết cấu bằng thống kê (statistical moments).
   - HSV của OpenCV: H ∈ [0,180), S, V ∈ [0,256). Trả về kèm `feature_names()` để giải thích trên slide.
   - Cache đặc trưng theo `image_id` (npz) để các thí nghiệm không tính lại.
4. **`src/banana_ripeness/models/base.py`**: `BaseRipenessModel(ABC)` có `fit(train_df, val_df=None)`, `predict(df)`,
   `evaluate(df)`, `save/load`. Có thêm `ImageFeatureModel` (đặc trưng + sklearn Pipeline[StandardScaler, clf]):
   chọn tham số bằng `GridSearchCV(cv=PredefinedSplit(train_df.cv_fold_<scheme>), scoring="f1_macro")`, nên tham số
   được chọn **trong train, theo nhóm**. Std lấy từ `std_test_score` của 5 fold ở tham số tốt nhất, thêm bootstrap std trên test.
5. **`models/svm_color.py`** (phần của Luân): `ColorHistogramSVM(kernel="linear"|"rbf", space, use_mask, bins)`,
   lưới tham số: linear C ∈ {0.01,0.1,1,10,100}; rbf C ∈ {0.1,1,10,100} × gamma ∈ {"scale",0.001,0.01,0.1,1};
   `class_weight="balanced"`. Thêm `ColorHistogramKNN` (k ∈ {1,3,5,7,9,15}) làm mốc.
6. **`models/hog_svm.py`** (khung cho Nguyên) và **`models/resnet18.py` + `gradcam.py`** (khung cho Quỳnh): kế thừa
   `BaseRipenessModel`, chạy được với tham số mặc định, ghi rõ "người phụ trách chỉnh". ResNet: dùng `stats.json`
   (mean/std, class_weight), early stopping theo val macro-F1. Đã kiểm tra được là PyTorch cài được để test.
7. **`src/banana_ripeness/evaluation.py`**: `compute_metrics` (accuracy, precision/recall/F1 macro, báo cáo theo lớp,
   ma trận nhầm lẫn), `bootstrap_std`, `plot_confusion_matrix`, `results_table` → `reports/results_*.csv`.
8. **`notebooks/10_svm_color_histogram.ipynb`**: bảng {RGB, HSV} × {linear, RBF} × {có, không mask} + KNN; trên
   `split_mixed` (test) và `split_external` (external = BananaImageBD); ma trận nhầm lẫn của mô hình tốt nhất;
   vài ảnh demo đoán đúng + phân tích nhầm Half_Ripe ↔ Ripe. Notebook chỉ import và gọi.
9. `20_hog_svm.ipynb`, `30_resnet18_gradcam.ipynb`: khung tương tự cho Nguyên và Quỳnh.
10. Test cho đặc trưng/mô hình (`tests/`), cập nhật README, commit.

## 5. Vấn đề dữ liệu CẦN BÁO NHÓM SỚM

1. **EDA và preprocessing mâu thuẫn về số ảnh trùng của BananaID.** EDA (dHash 64 bit, ≤ 6 bit) chỉ còn
   888/1960 ảnh khác nhau và 1757 cặp gần trùng khác nhãn. Preprocessing (dHash 256 bit) gần như không loại ảnh BananaID nào.
   Ảnh BananaID nền trắng, chuối chiếm ít diện tích (≈10% pixel vàng), nên hash 64 bit có thể quá thô. Nhưng cũng có thể
   đó là nhiều ảnh chụp **cùng một quả**. Pipeline mới sinh `reports/hash64_vs_256*.{csv,png}` và cảnh báo để xem tay.
   Kết luận ảnh hưởng trực tiếp đến độ tin cậy của kết quả test.
2. **Kiểm định ngoài:** notebook gốc đưa BananaImageBD vào train, trái với slide 7. Đã thêm `split_external`.
3. Slide ghi **"chia theo quả"** nhưng thực tế là chia theo nhóm ảnh gần trùng (không có ID quả). Nên sửa chữ trên slide.
4. Slide chưa nhắc **BananaRipeness_GitHub** (≈3462 ảnh, hơn nửa dữ liệu). Ánh xạ Class A–D → 4 mức là xấp xỉ.
   Ảnh GitHub **nền tối** (brightness ≈ 0.31), khác 2 dataset nền trắng. Cần xem `reports/mask_qa.png`: Otsu trên kênh S
   có thể kém ở dataset này. Nếu `mask_fallback` cao thì báo lại.
5. Notebook gốc: 6274 ảnh đọc → loại 41 → 6233 ảnh; hue giảm dần theo độ chín ở cả 3 dataset (gộp nhãn có cơ sở).

## 6. Quy ước

- Không commit ảnh/dữ liệu (`.gitignore`), chỉ commit code + `reports/*.csv|png` nhỏ.
- Seed 42 ở mọi nơi. Tên biến/hàm tiếng Anh, chú thích và in ra màn hình bằng tiếng Việt.
- Chạy test: `pytest -q` (dữ liệu giả, khoảng 6 giây).
