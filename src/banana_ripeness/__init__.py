"""banana_ripeness — phân loại 4 mức độ chín của chuối (đồ án CS231).

Các module chính:

- ``config``         : cấu hình chung (đường dẫn, ngưỡng, seed, ánh xạ nhãn)
- ``preprocessing``  : thu thập, loại trùng, chia tập, tách nền, lưu ảnh + manifest
- ``data``           : đọc manifest và ảnh/mask đã tiền xử lý (dùng chung cho mọi mô hình)
- ``features``       : đặc trưng thủ công (histogram màu, HOG)
- ``models``         : wrapper chung ``BaseRipenessModel`` và các mô hình cụ thể
- ``evaluation``     : độ đo, ma trận nhầm lẫn, bảng kết quả
"""
from .config import CLASS_NAMES, Config, load_config

__all__ = ["CLASS_NAMES", "Config", "load_config"]
__version__ = "0.1.0"
