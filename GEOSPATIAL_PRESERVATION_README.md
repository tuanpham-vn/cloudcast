# Geospatial Metadata Preservation in CloudCast

## Tổng quan

Tính năng này cho phép bảo tồn thông tin địa lý (geospatial metadata) từ file TIF đầu vào sang file TIF đầu ra sau khi thực hiện quá trình cắt, dự đoán và ghép lại.

## Yêu cầu

```bash
pip install GDAL>=3.0.0
```

## Cách sử dụng

### Bật bảo tồn thông tin địa lý (mặc định)
```bash
python predict_radar_from_tif.py \
  --timestamp "2025-10-07-13-40" \
  --input_dir "data" \
  --output_dir "predictions" \
  --model_dir "models/my_model" \
  --preserve_geospatial
```

### Tắt bảo tồn thông tin địa lý
```bash
python predict_radar_from_tif.py \
  --timestamp "2025-10-07-13-40" \
  --input_dir "data" \
  --output_dir "predictions" \
  --model_dir "models/my_model" \
  --no-preserve_geospatial
```

## Lợi ích

1. **Tọa độ chính xác**: File đầu ra giữ nguyên hệ tọa độ của file gốc
2. **Tích hợp GIS**: Có thể mở trực tiếp trong phần mềm GIS (QGIS, ArcGIS)
3. **Phân tích không gian**: Hỗ trợ phân tích và overlay với dữ liệu địa lý khác
4. **Tương thích**: Tương thích với các chuẩn geospatial hiện có

## Cách thức hoạt động

1. **Đọc metadata**: Đọc thông tin địa lý từ file TIF đầu tiên
2. **Xử lý patches**: Thực hiện cắt, dự đoán và ghép như bình thường
3. **Lưu với metadata**: Áp dụng lại thông tin địa lý cho file đầu ra

## Fallback

- Nếu GDAL không có sẵn: Tự động chuyển về PIL (không có thông tin địa lý)
- Nếu file gốc không có metadata: Lưu như file TIF thông thường
- Nếu có lỗi GDAL: Tự động fallback về PIL

## Kiểm tra

Chạy script test để kiểm tra tính năng:
```bash
python test_geospatial_preservation.py
```

## Lưu ý kỹ thuật

- Thông tin địa lý được đọc từ file đầu tiên trong sequence
- Tất cả file đầu ra sẽ có cùng thông tin địa lý
- Kích thước và vị trí ảnh được giữ nguyên (512x512 trong trường hợp của bạn)
- Overlap và center crop không ảnh hưởng đến tọa độ tổng thể
