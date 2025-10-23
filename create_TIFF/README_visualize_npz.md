# Script Visualize NPZ Training Data

Script `visualize_npz_data_training.py` giúp bạn kiểm tra và visualize dữ liệu training đã tạo từ script `create_tiff_dataset.py`.

## 📁 Files trong bộ công cụ

1. **`create_tiff_dataset.py`** - Script tạo dataset từ TIF files với sliding window
2. **`visualize_npz_data_training.py`** - Script visualize và kiểm tra dữ liệu đã tạo
3. **`debug_example.py`** - Script demo để debug từng bước trong editor
4. **`example_visualization.py`** - Script demo cách hoạt động của sliding window
5. **`visualize_requirements.txt`** - Dependencies cho visualization
6. **`README_visualize_npz.md`** - Hướng dẫn chi tiết sử dụng

## Mục đích

- **Kiểm tra chất lượng dữ liệu**: Đảm bảo patches được tạo đúng cách
- **Hiểu cấu trúc dữ liệu**: Xem cách sliding window hoạt động
- **Debug training data**: Phát hiện vấn đề với overlap và patches

## Cách sử dụng

### 1. Cài đặt dependencies

```bash
pip install numpy matplotlib
```

### 2. Chạy script

#### **Từ command line (khuyên dùng)**:
```bash
python visualize_npz_data_training.py --npz_file path/to/your/dataset.npz
```

#### **Từ editor (PyCharm, VSCode) để debug**:
```python
# Chỉ cần chạy file trực tiếp trong editor
# Script sẽ tự động sử dụng default arguments
python visualize_npz_data_training.py
```

### 3. Tương tác

Script sẽ hiển thị:
- Thông tin về dataset đã load
- Prompt để nhập index của sample muốn xem
- Visualization của sample được chọn

## Ví dụ thực tế

Giả sử bạn đã tạo dataset với `create_tiff_dataset.py`:

```bash
# Tạo dataset
python create_tiff_dataset.py \
  --input_dir /data/tif_files \
  --output_dir /data/datasets \
  --start_date 2024-01-01 \
  --end_date 2024-01-31 \
  --img_size 512x512 \
  --n_channels 4 \
  --leadtime_conditioning 18

# Kết quả:
# tiff_sliding_2024-01-01_2024-01-31_patch512_overlap10pct_ch4_lt18_float32.npz
```

Bây giờ visualize:

#### **Từ command line**:
```bash
python visualize_npz_data_training.py \
  --npz_file /data/datasets/tiff_sliding_2024-01-01_2024-01-31_patch512_overlap10pct_ch4_lt18_float32.npz
```

#### **Từ editor để debug**:
```bash
# Chỉ cần chạy file trong editor (PyCharm, VSCode)
# Script sẽ tự động sử dụng default path
python visualize_npz_data_training.py
```

## Debug Mode

### **Đặc điểm của Debug Mode**:
- ✅ **Không cần command line arguments**
- ✅ **Sử dụng default file path**
- ✅ **Có thể đặt breakpoints** trong editor
- ✅ **Debug từng bước** một cách dễ dàng

### **Default Configuration**:
```python
# Khi chạy từ editor, script sử dụng:
npz_file = "output/tiff_sliding_2025-09-28_2025-09-29_patch512_overlap10pct_ch4_lt18_float32.npz"
max_samples = 10
```

## Đầu ra Visualization

### **Thông tin hiển thị**:
```
Dataset loaded: 25000 samples of shape (512, 512, 1)
Note: This dataset uses sliding window approach with 10% overlap

============================================================
INTERACTIVE VISUALIZATION
============================================================
Dataset contains 25000 samples
Enter sample index to visualize (0 to exit):
Valid range: 0 to 24999
```

### **Khi nhập sample index (ví dụ: 100)**:
- Hiển thị patch hiện tại (Current Patch)
- Hiển thị patch liền kề với vùng overlap (Adjacent Patch)
- Thông tin metadata đầy đủ

## Cấu trúc Visualization

### **Sliding Window Visualization**:
```
┌─────────────────────────────────────────────────────────┐
│ Sample #101 | 2024-01-01_0010_acc10.tif | Patch 3/25   │
├─────────────────┬───────────────────────────────────────┤
│ Current Patch   │ Adjacent Patch (10% Overlap)         │
│                 │                                       │
│ [512x512 img]   │ [512x512 img]                         │
│ 2024-01-01T0010 │ with red rectangle showing overlap    │
└─────────────────┴───────────────────────────────────────┘
```

## Thông tin hiển thị

### **Metadata**:
- **Sample index**: Số thứ tự trong dataset
- **Original file**: File TIF gốc tạo ra patch này
- **Patch index**: Thứ tự patch trong ảnh gốc (1/25)
- **Timestamp**: Thời gian của file gốc

### **Thông số kỹ thuật**:
- **Patch size**: 512×512 pixels
- **Overlap ratio**: 10% (stride ≈ 461 pixels)
- **Total patches**: Tổng số patches từ tất cả ảnh

## Lợi ích của việc visualize

### **1. Kiểm tra chất lượng patches**:
- Đảm bảo patches không bị cắt xén
- Kiểm tra overlap hoạt động đúng
- Phát hiện patches lỗi hoặc missing

### **2. Hiểu cách sliding window hoạt động**:
- Thấy rõ cách patches chồng lấp nhau
- Hiểu tại sao có nhiều patches từ cùng một ảnh
- Visualize vùng overlap để đảm bảo tính liên tục

### **3. Debug training data**:
- Phát hiện vấn đề với kích thước ảnh
- Kiểm tra normalization dữ liệu
- Đảm bảo metadata được lưu đúng

## Debug trong Editor

### **Cách đặt breakpoints để debug**:

1. **Mở file trong PyCharm/VSCode**
2. **Đặt breakpoint** tại các dòng quan trọng:
   ```python
   # Đặt breakpoint tại đây để kiểm tra dữ liệu đầu vào
   def main(args=None):

   # Đặt breakpoint tại đây để kiểm tra quá trình load
   images, metadata = load_npz_data(args.npz_file)

   # Đặt breakpoint tại đây để kiểm tra visualization
   fig = create_training_sample_visualization(sample_idx, images, metadata)
   ```

3. **Chạy script** từ editor (không cần command line arguments)
4. **Debug từng bước** để kiểm tra:
   - Dữ liệu có load đúng không
   - Metadata có đúng định dạng không
   - Visualization có hiển thị đúng không

### **Debug từng bước với debug_example.py**:

Để dễ dàng hơn, sử dụng file `debug_example.py`:

```bash
python debug_example.py
```

**Script này sẽ kiểm tra từng bước**:
- ✅ Load default arguments
- ✅ Load dữ liệu từ file NPZ
- ✅ Tạo visualization object
- ✅ Cho phép đặt breakpoints tại các điểm quan trọng

### **Troubleshooting**

#### **Lỗi thường gặp**:
- **File not found**: Kiểm tra đường dẫn đến file .npz
- **Memory error**: Dataset quá lớn, giảm `--max_samples`
- **Import error**: Cài đặt matplotlib và numpy
- **Pickle error**: Script sử dụng `allow_pickle=True` để load metadata

#### **Lỗi Object arrays cannot be loaded**:
Nếu gặp lỗi:
```
ERROR loading NPZ file: Object arrays cannot be loaded when allow_pickle=False
```

**Nguyên nhân**: File .npz chứa metadata dạng dictionary, numpy mặc định không cho phép load vì lý do bảo mật.

**Giải pháp**: Script đã tự động sử dụng `allow_pickle=True` để xử lý trường hợp này.

**⚠️ Lưu ý bảo mật**: `allow_pickle=True` có thể gây rủi ro bảo mật nếu file .npz đến từ nguồn không tin cậy. Chỉ sử dụng với file do chính bạn tạo ra.

### **Kiểm tra dữ liệu**:
```bash
# Xem thông tin cơ bản
python -c "import numpy as np; data=np.load('dataset.npz', allow_pickle=True); print(data['arr_0'].shape)"

# Xem metadata của sample đầu tiên
print(data['arr_1'][0])
```

### **Debug từng bước**:
```bash
# Chạy debug example để kiểm tra từng bước
python debug_example.py
```

**Debug example sẽ kiểm tra**:
- ✅ Load default arguments
- ✅ Load dữ liệu từ file NPZ
- ✅ Tạo visualization object

## Kết luận

Script này giúp bạn:
- ✅ Kiểm tra chất lượng dữ liệu training
- ✅ Hiểu cách sliding window hoạt động
- ✅ Debug vấn đề với patches và overlap
- ✅ Đảm bảo dữ liệu phù hợp cho huấn luyện mô hình

Đây là công cụ thiết yếu để đảm bảo chất lượng dữ liệu trước khi huấn luyện mô hình CloudCast với sliding window approach!
