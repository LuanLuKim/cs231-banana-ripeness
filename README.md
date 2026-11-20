# CS231 — Phân loại độ chín chuối

Ảnh một quả chuối → một trong 4 mức: `Unripe`, `Half_Ripe`, `Ripe`, `Overripe`.

## Cấu trúc

```
configs/            cấu hình YAML (kaggle.yaml, local.yaml)
src/banana_ripeness/
  config.py         tham số chung
  data.py           đọc manifest + ảnh + mask (mọi mô hình dùng chung)
  preprocessing/    thu thập, loại trùng, chia tập, tách nền Otsu, lưu output
scripts/preprocess.py
notebooks/          01_eda, 02_preprocessing, 10_svm, 20_hog, 30_resnet18 (chỉ import và gọi)
tests/              pytest trên dữ liệu giả
HANDOFF.md          trạng thái + việc còn lại
```

## Chạy tiền xử lý

```bash
pip install -r requirements.txt
python scripts/preprocess.py --config configs/kaggle.yaml     # trên Kaggle
python scripts/preprocess.py --config configs/local.yaml      # trên máy
```

Trên Kaggle (bật Internet):
```python
!git clone https://github.com/LuanLuKim/cs231-banana-ripeness.git
import sys; sys.path.insert(0, "cs231-banana-ripeness/src")
from banana_ripeness import load_config
from banana_ripeness.preprocessing import run_preprocessing
result = run_preprocessing(load_config("cs231-banana-ripeness/configs/kaggle.yaml"))
```
Lưu `/kaggle/working/banana_preprocessed` thành Kaggle Dataset để cả nhóm gắn vào.

## Dùng output

```python
from banana_ripeness.data import load_manifest, get_split
df = load_manifest("/kaggle/input/banana-preprocessed")
train = get_split(df, "external", "train")      # hoặc scheme "mixed"
ext   = get_split(df, "external", "external")   # BananaImageBD, kiểm định ngoài
```

## Test

```bash
pytest -q
```
