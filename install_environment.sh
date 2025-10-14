#!/bin/bash
set -e

echo "=========================================="
echo "CloudCast Environment Installation"
echo "=========================================="

# Detect CUDA version
echo "Detecting CUDA version..."
CUDA_VERSION=""
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "Found CUDA version: $CUDA_VERSION"
    echo "CUDA compiler path: $(which nvcc)"
    # Kiểm tra NVIDIA driver
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "NVIDIA driver detected:"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits | head -1
    fi
else
    echo "CUDA not found, will install CPU-only TensorFlow"
    CUDA_VERSION="none"
fi

# Tạo môi trường conda
echo "Creating conda environment..."
conda create -n cloudcast python=3.10 -y

# Kích hoạt môi trường
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cloudcast

# Cài TensorFlow phù hợp với CUDA version
echo "Installing TensorFlow based on CUDA version..."

if [[ "$CUDA_VERSION" == "none" ]]; then
    echo "Installing CPU-only TensorFlow..."
    pip install --no-cache-dir 'tensorflow==2.15.0' 'numpy<2.0,>=1.23.5'
elif [[ "$CUDA_VERSION" == "13.0" ]] || [[ "$CUDA_VERSION" == "13.1" ]] || [[ "$CUDA_VERSION" == "13.2" ]]; then
    echo "Installing TensorFlow 2.15 (compatible with CUDA 13.x)..."
    # CUDA 13.x thường cần CPU-only TensorFlow vì compatibility issues
    pip install --no-cache-dir 'tensorflow==2.15.0' 'numpy<2.0,>=1.23.5'
elif [[ "$CUDA_VERSION" == "12."* ]]; then
    echo "Installing TensorFlow 2.15 with CUDA 12 support..."
    # Thử cài với CUDA support trước
    if pip install --no-cache-dir 'tensorflow[and-cuda]==2.15.0' 'numpy<2.0,>=1.23.5' 2>/dev/null; then
        echo "Successfully installed TensorFlow with CUDA support"
    else
        echo "CUDA support failed, installing CPU-only version..."
        pip install --no-cache-dir 'tensorflow==2.15.0' 'numpy<2.0,>=1.23.5'
    fi
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    echo "Installing TensorFlow 2.13 with CUDA 11 support..."
    pip install --no-cache-dir 'tensorflow==2.13.0' 'numpy<2.0,>=1.23.5'
else
    echo "Unknown CUDA version $CUDA_VERSION, installing TensorFlow 2.15..."
    pip install --no-cache-dir 'tensorflow==2.15.0' 'numpy<2.0,>=1.23.5'
fi

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
# echo "Installing GDAL and Basemap..."
# conda install -c conda-forge gdal=3.5 -y
# conda install -c conda-forge basemap -y

# Kiểm tra GPU support
echo "=========================================="
echo "Verifying GPU support..."
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('Built with GPU:', tf.test.is_built_with_gpu_support())
gpus = tf.config.list_physical_devices('GPU')
print('Available GPUs:', len(gpus))
for i, gpu in enumerate(gpus):
    print(f'  GPU {i}: {gpu.name}')
if len(gpus) > 0:
    print('✅ GPU support is working!')
else:
    print('⚠️  No GPUs detected - will use CPU only')
"

echo "=========================================="
echo "Installation completed!"
echo ""
echo "To use the environment:"
echo "  1. conda activate cloudcast"
echo "  2. source setup_env.sh"
echo "  3. bash train_cloudcast.sh --dataseries_file <your_data.npz> --label <your_label>"
echo ""
echo "CUDA version detected: $CUDA_VERSION"
echo "TensorFlow version: 2.15.0 (compatible with CUDA 12.x and 13.x)"
echo "=========================================="

