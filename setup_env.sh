#!/bin/bash

# CloudCast GPU Environment Setup Script
# This script activates the conda environment and sets up environment variables for GPU training

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cloudcast

# Set PYTHONPATH to include project root
export PYTHONPATH=/workspace/cloudcast:$PYTHONPATH

# Set LD_LIBRARY_PATH for CUDA and cuDNN libraries
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.9/lib64:/venv/cloudcast/lib/python3.10/site-packages/nvidia/cudnn/lib:/venv/cloudcast/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

# Suppress TensorFlow warnings and XLA logs
export TF_CPP_MIN_LOG_LEVEL=2
export TF_XLA_FLAGS='--tf_xla_auto_jit=2 --tf_xla_min_cluster_size=4'

# Verify GPU availability
echo "=========================================="
echo "CloudCast Environment Information"
echo "=========================================="
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "Number of GPUs: $(python -c 'import tensorflow as tf; print(len(tf.config.list_physical_devices("GPU")))')"
echo "=========================================="

