# Hướng Dẫn Tái Tạo Môi Trường CloudCast với GPU

## Yêu Cầu Hệ Thống

- **OS**: Linux (Ubuntu 20.04+ hoặc tương đương)
- **CUDA**: 12.x (server hỗ trợ CUDA 12.9 trở xuống)
- **GPU**: NVIDIA GPU với driver hỗ trợ CUDA 12.x
- **RAM**: Tối thiểu 16GB (khuyến nghị 32GB+)
- **Python**: 3.10
- **Conda/Miniconda**: Đã cài đặt

## Bước 1: Cài Đặt Miniconda (nếu chưa có)

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Cài đặt
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# Khởi tạo conda
~/miniconda3/bin/conda init bash

# Reload shell
source ~/.bashrc
```

## Bước 2: Tạo Môi Trường Conda

```bash
# Tạo môi trường với Python 3.10
conda create -n cloudcast python=3.10 -y

# Kích hoạt môi trường
conda activate cloudcast
```

## Bước 3: Cài Đặt TensorFlow với CUDA 12

```bash
# Cài TensorFlow 2.16.1 với hỗ trợ CUDA 12 và NumPy 1.x
pip install --no-cache-dir 'tensorflow[and-cuda]==2.16.1' 'numpy<2.0,>=1.23.5'
```

## Bước 4: Cài Đặt Các Package Python Khác

```bash
# Cài đặt các package cần thiết
pip install --no-cache-dir \
    botocore \
    boto3 \
    numba \
    'opencv-python==4.6.0.66' \
    'scikit-image==0.19.3' \
    'scikit-learn==1.1.2' \
    tensorflow-datasets \
    tensorflow-metadata \
    matplotlib \
    PyWavelets

# Cài eccodes
pip install --no-cache-dir eccodes==1.6.1
```

## Bước 5: Cài Đặt GDAL và Basemap qua Conda

```bash
# Cài GDAL
conda install -c conda-forge gdal=3.5 -y

# Cài Basemap
conda install -c conda-forge basemap -y
```

## Bước 6: Thiết Lập Biến Môi Trường

Tạo file `~/.bashrc_cloudcast` hoặc thêm vào `~/.bashrc`:

```bash
# CloudCast Environment Variables
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
```

**Lưu ý**: Thay `/usr/local/cuda-12.9` bằng đường dẫn CUDA thực tế của bạn.

## Bước 7: Kiểm Tra GPU

```bash
# Kích hoạt môi trường
conda activate cloudcast

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

# Kiểm tra GPU
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('Number of GPUs:', len(gpus)); [print(f'GPU {i}: {gpu.name}') for i, gpu in enumerate(gpus)]"
```

Kết quả mong đợi:
```
TensorFlow version: 2.16.1
Number of GPUs: 4
GPU 0: /physical_device:GPU:0
GPU 1: /physical_device:GPU:1
GPU 2: /physical_device:GPU:2
GPU 3: /physical_device:GPU:3
```

## Bước 8: Chạy Training

```bash
cd /path/to/cloudcast

# Thiết lập môi trường
source setup_env.sh

# Chạy training
bash train_cloudcast.sh --dataseries_file data/patch000_2025-09-03_2025-10-08_patches_512512_float32.npz --label "test_4gpu"
```

## Script Tự Động Hóa

### File: `install_environment.sh`

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "CloudCast Environment Installation"
echo "=========================================="

# Tạo môi trường conda
echo "Creating conda environment..."
conda create -n cloudcast python=3.10 -y

# Kích hoạt môi trường
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cloudcast

# Cài TensorFlow với CUDA 12
echo "Installing TensorFlow 2.16 with CUDA 12..."
pip install --no-cache-dir 'tensorflow[and-cuda]==2.16.1' 'numpy<2.0,>=1.23.5'

# Cài các package Python
echo "Installing Python packages..."
pip install --no-cache-dir \
    botocore \
    boto3 \
    numba \
    'opencv-python==4.6.0.66' \
    'scikit-image==0.19.3' \
    'scikit-learn==1.1.2' \
    tensorflow-datasets \
    tensorflow-metadata \
    matplotlib \
    PyWavelets \
    eccodes==1.6.1

# Cài GDAL và Basemap
echo "Installing GDAL and Basemap..."
conda install -c conda-forge gdal=3.5 -y
conda install -c conda-forge basemap -y

echo "=========================================="
echo "Installation completed!"
echo "Activate environment with: conda activate cloudcast"
echo "=========================================="
```

### File: `setup_env.sh`

Đã tạo sẵn tại `setup_env.sh` - sử dụng để thiết lập biến môi trường trước khi chạy training.

## Export/Import Môi Trường

### Export môi trường hiện tại:

```bash
# Export danh sách package
conda activate cloudcast
conda env export > environment.yml
pip list --format=freeze > requirements_full.txt
```

### Import trên máy mới:

```bash
# Tạo từ environment.yml
conda env create -f environment.yml

# Hoặc tạo thủ công và cài từ requirements
conda create -n cloudcast python=3.10 -y
conda activate cloudcast
pip install -r requirements_full.txt
```

## Ghi Chú Quan Trọng

1. **CUDA Version**: TensorFlow 2.16.1 hỗ trợ CUDA 12.x. Nếu server của bạn có CUDA 11.x, sử dụng TensorFlow 2.14.0.

2. **LD_LIBRARY_PATH**: Phải thiết lập đúng đường dẫn đến:
   - Thư viện CUDA của hệ thống
   - Thư viện cuDNN và cuBLAS từ pip packages

3. **NumPy Version**: Phải dùng NumPy 1.x (1.23.5 - 1.26.4) để tương thích với TensorFlow 2.16.

4. **Python Version**: Python 3.10 là phiên bản tối ưu cho TensorFlow 2.16 và các package khác.

## Troubleshooting

### Lỗi: "Could not find cuda drivers"

```bash
# Kiểm tra NVIDIA driver
nvidia-smi

# Kiểm tra CUDA path
ls -la /usr/local/cuda*

# Set LD_LIBRARY_PATH đúng
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
```

### Lỗi: "Cannot dlopen some GPU libraries"

```bash
# Thêm đường dẫn đến thư viện NVIDIA từ pip
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
```

### Lỗi NumPy version conflict

```bash
# Downgrade NumPy
pip install --no-cache-dir 'numpy<2.0,>=1.23.5'
```

## Tổng Kết

Môi trường đã được thiết lập với:

- ✅ Python 3.10
- ✅ TensorFlow 2.16.1 với CUDA 12 support
- ✅ 4x NVIDIA RTX 4090 GPUs
- ✅ Tất cả dependencies từ requirements.txt
- ✅ GDAL 3.5 và Basemap
- ✅ NumPy 1.26.4 (tương thích)

Sử dụng script `setup_env.sh` trước mỗi lần chạy training để đảm bảo môi trường được thiết lập đúng cách.

