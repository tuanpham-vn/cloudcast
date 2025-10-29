# Script tạo dataset từ TIF files cho CloudCast (Sliding Window Approach)

Script `create_tiff_dataset.py` giúp bạn tạo dataset từ các file TIF lớn (2922×1776) bằng cách sử dụng **cửa sổ trượt với overlap 10%** để tạo nhiều patch đa dạng mà không mất thông tin.

## Cách tiếp cận đặc biệt

### **Sliding Window với Overlap 10%**
- **Ảnh gốc**: 2922×1776 pixels
- **Patch size**: 512×512 pixels (có thể tùy chỉnh)
- **Overlap**: 10% (stride ≈ 461 pixels)
- **Ưu điểm**:
  - Không mất thông tin từ việc resize
  - Tạo nhiều patch đa dạng từ cùng một ảnh
  - Cải thiện khả năng generalization của mô hình
  - Xử lý tốt các edge cases

## Yêu cầu

```bash
pip install -r create_tiff_dataset_requirements.txt
```

## Cách sử dụng

### Cấu trúc thư mục dữ liệu

Đặt các file TIF của bạn theo cấu trúc:

```
your_data/
├── 2024-01-01_0000.tif
├── 2024-01-01_0010.tif
├── 2024-01-01_0020.tif
└── ...
```

**Định dạng tên file**: `YYYY-MM-DD_HHMM.tif` (cách nhau 10 phút)

### Chạy script

```bash
python create_tiff_dataset.py \
  --input_dir data \
  --output_dir output \
  --start_date 2024-01-01 \
  --end_date 2024-01-31 \
  --img_size 512x512 \
  --n_channels 4 \
  --leadtime_conditioning 18 \
  --time_interval 10
```

### Các tham số quan trọng

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--input_dir` | Bắt buộc | Thư mục chứa TIF files |
| `--output_dir` | Bắt buộc | Thư mục đầu ra |
| `--start_date` | Bắt buộc | Ngày bắt đầu (YYYY-MM-DD) |
| `--end_date` | Bắt buộc | Ngày kết thúc (YYYY-MM-DD) |
| `--img_size` | 512x512 | Kích thước patch (WxH) |
| `--n_channels` | 4 | Số kênh đầu vào |
| `--leadtime_conditioning` | 18 | Số bước dự báo tối đa (3 giờ) |
| `--time_interval` | 10 | Khoảng thời gian giữa các file (phút) |

## Cách tiếp cận mới: Individual Files per Image

Script này tạo **một file .npz riêng biệt cho mỗi ảnh TIF gốc**, với định dạng giống như `create-dataset.py`:

### **Định dạng đầu ra**:
```python
# Mỗi file .npz chứa:
arr_0.shape = (N, 512, 512, 1)     # N patches từ 1 ảnh gốc
arr_1.shape = (N,)                  # Cùng timestamp cho tất cả patches
arr_1.dtype = '<U15'                # String timestamps
```

### **Ví dụ thực tế**:

Với dữ liệu của bạn:
- **Kích thước gốc**: 2922×1776 pixels
- **Patch size**: 512×512 pixels (tùy chỉnh được)
- **4 ảnh đầu vào**: Dự báo độ che phủ mây
- **18 bước dự báo**: Dự báo tối đa 3 giờ
- **Sliding window**: Overlap 10% tạo nhiều patches từ mỗi ảnh

```bash
python create_tiff_dataset.py \
  --input_dir /data/tif_files \
  --output_dir /data/cloudcast_datasets \
  --start_date 2024-01-01 \
  --end_date 2024-01-31 \
  --img_size 512x512 \
  --n_channels 4 \
  --leadtime_conditioning 18 \
  --time_interval 10 \
  --output_format npz \
  --dtype float32
```

### **Ví dụ đầu ra thực tế**:

Với 1 ảnh gốc 2922×1776:
- **Sliding window (overlap 10%)**: Khoảng 25-30 patches khác nhau

**Kết quả cuối cùng**:
```
Found 1000 TIF files out of 1008 expected
Processed 1000 files
Creating dataset for 2025-09-28_2010_acc10.tif: 25 patches, shape (25, 512, 512, 1)
Creating dataset for 2025-09-28_2020_acc10.tif: 25 patches, shape (25, 512, 512, 1)
...
Individual files created: 1000
```

## Đầu ra

### Files dataset riêng biệt
```
2025-09-28_2010_acc10_patches_512_float32.npz
2025-09-28_2020_acc10_patches_512_float32.npz
2025-09-28_2030_acc10_patches_512_float32.npz
...
```

### Nội dung mỗi file
```python
import numpy as np
data = np.load('2025-09-28_2010_acc10_patches_512_float32.npz')
images = data['arr_0']     # Shape: (N, 512, 512, 1) - patches từ 1 ảnh
timestamps = data['arr_1']  # Shape: (N,) - cùng timestamp cho tất cả patches
```

## Sử dụng với CloudCast

### **Cách 1: Sử dụng từng file riêng biệt**
```bash
python cloudcast/cloudcast-unet.py \
  --dataseries_file /data/cloudcast_datasets/2025-09-28_2010_acc10_patches_512_float32.npz \
  --n_channels 4 \
  --leadtime_conditioning 18 \
  --preprocess img_size=512x512 \
  --loss_function bcl1 \
  --include_sun_elevation_angle \
  --label tiff_individual_512_4ch_18lt
```

### **Cách 2: Kết hợp nhiều files** (cho dataset lớn hơn)
```bash
# Tạo danh sách các file cần kết hợp
file_list = [
    "2025-09-28_2010_acc10_patches_512_float32.npz",
    "2025-09-28_2020_acc10_patches_512_float32.npz",
    "2025-09-28_2030_acc10_patches_512_float32.npz"
]

# Kết hợp thành dataset lớn hơn
python combine_datasets.py --input_files file_list --output combined_dataset.npz
```

## Ưu điểm của Individual Files Approach

### **So với cách tạo một file duy nhất**:

| Tiêu chí | Một file tổng hợp | Individual Files |
|----------|------------------|------------------|
| **Cấu trúc** | Tất cả patches trong 1 file | Mỗi ảnh 1 file riêng |
| **Tương thích** | Khó tương thích với CloudCast | Hoàn toàn tương thích |
| **Debug** | Khó debug từng ảnh | Dễ debug từng ảnh |
| **Memory** | Có thể lớn | Nhỏ hơn, dễ quản lý |

### **Cách hoạt động**:
```python
# Với ảnh 2025-09-28_2010_acc10.tif:
# - Tạo 25 patches với sliding window
# - Tất cả patches có cùng timestamp: 2025-09-28T2010
# - Lưu vào file: 2025-09-28_2010_acc10_patches_512_float32.npz
# - Định dạng giống hệt create-dataset.py
```

### **Ví dụ sử dụng với CloudCast**:
```python
# File này có định dạng giống hệt output của create-dataset.py
# Có thể dùng trực tiếp với cloudcast-unet.py
python cloudcast/cloudcast-unet.py \
  --dataseries_file 2025-09-28_2010_acc10_patches_512_float32.npz \
  --n_channels 4 \
  --leadtime_conditioning 18
```

## Lưu ý quan trọng

1. **Individual files**: Mỗi ảnh gốc tạo ra 1 file .npz riêng biệt
2. **Tương thích hoàn toàn**: Định dạng giống hệt `create-dataset.py`
3. **Dễ quản lý**: Có thể train từng file riêng hoặc kết hợp nhiều files
4. **Timestamp extraction**: Tự động trích xuất từ tên file (YYYY-MM-DD_HHMM_acc10.tif)

## Troubleshooting

- **Memory error**: Nếu quá nhiều patches, giảm kích thước ảnh đầu ra (ví dụ: 256x256)
- **Individual files**: Script tạo nhiều file riêng biệt, không phải một file tổng hợp
- **Missing files**: Script tự động bỏ qua và báo cáo các file thiếu
- **Import errors**: Đảm bảo đã cài đặt Pillow và numpy
- **File not found**: Kiểm tra đường dẫn và định dạng tên file (YYYY-MM-DD_HHMM_acc10.tif)

## Kết luận

Đây là công cụ thiết yếu để chuẩn bị dữ liệu huấn luyện từ dữ liệu TIF lớn thành định dạng phù hợp cho mô hình CloudCast.

Với **Individual Files Approach**, bạn có thể:
- ✅ Tạo nhiều file .npz riêng biệt cho mỗi ảnh gốc
- ✅ Định dạng giống hệt `create-dataset.py` (tương thích hoàn toàn)
- ✅ Sử dụng trực tiếp với `cloudcast-unet.py` mà không cần sửa đổi
- ✅ Dễ dàng debug và quản lý từng file riêng biệt
- ✅ Không mất thông tin từ việc resize toàn bộ ảnh

### **Ưu điểm chính**:
1. **Tương thích hoàn toàn** với CloudCast gốc
2. **Dễ quản lý** - mỗi ảnh một file riêng
3. **Có thể kết hợp** nhiều files cho dataset lớn hơn
4. **Timestamp chính xác** - trích xuất từ tên file
5. **Sliding window** - tạo patches đa dạng với overlap

Script này giúp bạn dễ dàng chuyển đổi dữ liệu TIF riêng (2922×1776) thành định dạng phù hợp để huấn luyện mô hình CloudCast! 🎉
