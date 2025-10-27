#!/bin/bash

# CloudCast Training Script
# This script handles the Python path issues automatically

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sử dụng môi trường hiện tại
echo "Sử dụng môi trường hiện tại: $CONDA_DEFAULT_ENV"

# Verify TensorFlow is available and show GPUs
echo "Checking TensorFlow installation..."
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))" || {
    echo "ERROR: TensorFlow not available in this environment!"
    exit 1
}

# Set PYTHONPATH to include the project root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Suppress TensorFlow warnings (0=all, 1=filter INFO, 2=filter INFO+WARNING, 3=filter all except ERROR)
export TF_CPP_MIN_LOG_LEVEL=2

# Disable XLA by default to avoid excessive logs and potential instability on some combos
unset TF_XLA_FLAGS

# NCCL tuning for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Limit TF threads to mitigate pthread_create failures
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2

# Optional: restrict visible GPUs via CUDA_VISIBLE_DEVICES if needed
# export CUDA_VISIBLE_DEVICES=0,1

# Change to project root directory
cd "$SCRIPT_DIR"

# Create logs directory if not exists
mkdir -p logs

# Default parameters
LOSS_FUNCTION="ssim"
N_CHANNELS=9
LEADTIME_CONDITIONING=20
IMG_SIZE="512x512"
DATASERIES_FILE=""
DATASERIES_DIRECTORY="data"
LABEL=""
SEQUENCE_STRIDE_MINUTES=10
SEQUENCE_OFFSET_MINUTES=0
CHECKPOINT_PATH=""
LEARNING_RATE=""
MIXED_PRECISION=true  # Mặc định bật mixed precision

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataseries_file)
            DATASERIES_FILE="$2"
            shift 2
            ;;
        --dataseries_directory)
            DATASERIES_DIRECTORY="$2"
            shift 2
            ;;
        --label)
            LABEL="$2"
            shift 2
            ;;
        --n_channels)
            N_CHANNELS="$2"
            shift 2
            ;;
        --loss_function)
            LOSS_FUNCTION="$2"
            shift 2
            ;;
        --leadtime_conditioning)
            LEADTIME_CONDITIONING="$2"
            shift 2
            ;;
        --sequence_stride_minutes)
            SEQUENCE_STRIDE_MINUTES="$2"
            shift 2
            ;;
        --sequence_offset_minutes)
            SEQUENCE_OFFSET_MINUTES="$2"
            shift 2
            ;;
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --no_mixed_precision)
            MIXED_PRECISION=false
            shift
            ;;
        --force_load_weights)
            FORCE_LOAD_WEIGHTS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dataseries_file FILE       Path to single NPZ file (must have .npz extension)"
            echo "  --dataseries_directory DIR   Path to directory with multiple NPZ files (must have .npz extension)"
            echo "  --label LABEL               Training label (default: $LABEL)"
            echo "  --n_channels N              Number of input channels (default: $N_CHANNELS)"
            echo "  --loss_function NAME        Loss: MeanSquaredError | ssim | msssim | bcl1 | fss | ks | coss | mae | ssim_mae[_wSSIM_wMAE]"
            echo "  --leadtime_conditioning N   Lead time conditioning (default: $LEADTIME_CONDITIONING)"
            echo "  --sequence_stride_minutes N Sequence stride in minutes (default: $SEQUENCE_STRIDE_MINUTES)"
            echo "  --sequence_offset_minutes N Sequence offset in minutes (default: $SEQUENCE_OFFSET_MINUTES)"
            echo "  --checkpoint_path PATH      Path to checkpoint for fine-tuning"
            echo "  --learning_rate LR          Learning rate (default: auto - 5e-4 for SSIM, 1e-3 for others, 1e-5 for fine-tune)"
            echo "  --no_mixed_precision        Disable mixed precision training (default: enabled, bfloat16 for SSIM)"
            echo "  --force_load_weights        Force load weights even with warnings (for fine-tuning)"
            echo "  --help                      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if dataseries file or directory is provided
if [ -z "$DATASERIES_FILE" ] && [ -z "$DATASERIES_DIRECTORY" ]; then
    echo "ERROR: Either --dataseries_file or --dataseries_directory must be provided"
    exit 1
fi

if [ -n "$DATASERIES_FILE" ]; then
    if [ ! -f "$DATASERIES_FILE" ]; then
        echo "ERROR: Dataseries file not found: $DATASERIES_FILE"
        echo "Available files in output/:"
        ls -la output/ 2>/dev/null || echo "No output directory found"
        exit 1
    fi
    DATA_SOURCE="File: $DATASERIES_FILE"
    DATA_ARG="--dataseries_file $DATASERIES_FILE"
elif [ -n "$DATASERIES_DIRECTORY" ]; then
    if [ ! -d "$DATASERIES_DIRECTORY" ]; then
        echo "ERROR: Dataseries directory not found: $DATASERIES_DIRECTORY"
        exit 1
    fi
    NPZ_COUNT=$(ls -1 "$DATASERIES_DIRECTORY"/*.npz 2>/dev/null | wc -l)
    DATA_SOURCE="Directory: $DATASERIES_DIRECTORY ($NPZ_COUNT NPZ files)"
    DATA_ARG="--dataseries_directory $DATASERIES_DIRECTORY"
fi

echo "=========================================="
echo "CloudCast Training"
echo "=========================================="
echo "Project directory: $SCRIPT_DIR"
echo "Data source:       $DATA_SOURCE"
echo "Label:             $LABEL"
echo "N channels:        $N_CHANNELS"
echo "Lead time:         $LEADTIME_CONDITIONING"
echo "Image size:        $IMG_SIZE"
echo "Sequence stride:   $SEQUENCE_STRIDE_MINUTES minutes"
echo "Sequence offset:   $SEQUENCE_OFFSET_MINUTES minutes"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint:        $CHECKPOINT_PATH (fine-tuning mode)"
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "✅ CHECKPOINT FILE EXISTS"
    else
        echo "❌ CHECKPOINT FILE NOT FOUND"
        echo "🛑 STOPPING TRAINING - CHECKPOINT FILE REQUIRED FOR FINE-TUNING"
        exit 1
    fi
fi
if [ -n "$LEARNING_RATE" ]; then
    echo "Learning rate:     $LEARNING_RATE"
else
    if [ -n "$CHECKPOINT_PATH" ]; then
        echo "Learning rate:     1e-5 (auto: fine-tuning)"
    elif [[ "$LOSS_FUNCTION" == ssim* ]] || [[ "$LOSS_FUNCTION" == msssim* ]]; then
        echo "Learning rate:     5e-4 (auto: SSIM loss)"
    else
        echo "Learning rate:     1e-3 (auto: default)"
    fi
fi
echo "=========================================="

# Build optional arguments
OPTIONAL_ARGS=""
if [ -n "$CHECKPOINT_PATH" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --checkpoint_path $CHECKPOINT_PATH"
fi
if [ -n "$LEARNING_RATE" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --learning_rate $LEARNING_RATE"
fi

# Create log filename with timestamp
TS_COMPACT=$(date +%Y%m%d_%H%M%S)
TS_CHECKPOINT=$(date +%Y_%m_%d_%H%M)
LOG_FILE="logs/training_${TS_COMPACT}.log"
echo "Training log will be saved to: $LOG_FILE"
echo ""

# Show live GPU status before training
echo "GPU snapshot before training:"
nvidia-smi || true
echo ""

# Run the training - output to both terminal and log file
export CLOUDCAST_TIMESTAMP="$TS_CHECKPOINT"
python cloudcast/cloudcast-unet.py \
  --loss_function "$LOSS_FUNCTION" \
  --n_channels "$N_CHANNELS" \
  --leadtime_conditioning "$LEADTIME_CONDITIONING" \
  --preprocess "img_size=$IMG_SIZE" \
  --sequence_stride_minutes "$SEQUENCE_STRIDE_MINUTES" \
  --sequence_offset_minutes "$SEQUENCE_OFFSET_MINUTES" \
  ${CHECKPOINT_PATH:+--checkpoint_path "$CHECKPOINT_PATH"} \
  $([ "$MIXED_PRECISION" = "false" ] && echo "--no_mixed_precision") \
  ${FORCE_LOAD_WEIGHTS:+--force_load_weights} \
  $DATA_ARG \
  $OPTIONAL_ARGS 2>&1 | tee "$LOG_FILE"

echo ""
echo "Training completed!"
echo "Log saved to: $LOG_FILE"

# ========================================
# EXAMPLE COMMANDS
# ========================================
#
# 0. Training với loss SSIM + MAE (50%/50% mặc định):
# bash train_cloudcast.sh \
#   --dataseries_directory input \
#   --label ssim_mae_70_30 \
#   --loss_function ssim_mae
#   # Lưu ý: thiết lập loss trong Python bằng tham số --loss_function ssim_mae khi gọi trực tiếp
#
#   # Nếu dùng trực tiếp python:
#   # python cloudcast/cloudcast-unet.py \
#   #   --dataseries_directory data \
#   #   --label ssim_mae_50_50 \
#   #   --loss_function ssim_mae
#
# 0b. Fine-tune với SSIM + MAE (50%/50%) và LR mặc định 1e-5:
# ./train_cloudcast.sh \
#   --dataseries_directory data \
#   --label "ft_ssim_mae_50_50" \
#   --checkpoint_path checkpoints/your_model/model.weights.h5
  # (không cần --learning_rate, script và Python sẽ dùng mặc định 1e-5 cho fine-tuning)
#
# 0c. Training với SSIM + MAE trọng số 70%/30%:
# # Dùng trực tiếp python để truyền loss:
#    bash train_cloudcast.sh \
#    --dataseries_directory  /home/databourg/workspace/cloudcast/output\
#    --label ssim_mae_70_30 \
#    --loss_function ssim_mae_0.7_0.3
#
# 1. Training với single NPZ file:
# ./train_cloudcast.sh --dataseries_file data/patch000_2025-09-03_2025-10-08_patches_512512_float32.npz --label "single_patch_10min"
#
# 2. Training với multiple NPZ patches:
# ./train_cloudcast.sh --dataseries_directory output/patches --label "multi_patch_10min"
#
# 3. Training với custom parameters:
# ./train_cloudcast.sh \
#   --dataseries_directory output/patches \
#   --label "custom_10min" \
#   --n_channels 4 \
#   --leadtime_conditioning 18 \
#   --sequence_stride_minutes 10 \
#   --sequence_offset_minutes 0
#
# 4. Training với 10-minute stride (mặc định):
# ./train_cloudcast.sh \
#   --dataseries_directory output/patches_10min \
#   --label "model_10min"
#
# 5. Training với even/odd chains (10-minute stride):
# # Even chain (0, 10, 20 minutes):
# ./train_cloudcast.sh \
#   --dataseries_directory output/patches \
#   --label "even_chain_10min" \
#   --sequence_stride_minutes 10 \
#   --sequence_offset_minutes 0
#
# # Odd chain (5, 15, 25 minutes):
# ./train_cloudcast.sh \
#   --dataseries_directory output/patches \
#   --label "odd_chain_10min" \
#   --sequence_stride_minutes 10 \
#   --sequence_offset_minutes 5
#
# 6. Fine-tuning from checkpoint (default LR = 5e-4):
# ./train_cloudcast.sh \
#   --dataseries_directory output/patches \
#   --label "finetuned_model" \
#   --checkpoint_path checkpoints/original_model/cp.ckpt
#
# 7. Fine-tuning with custom learning rate:
# ./train_cloudcast.sh \
#   --dataseries_directory output \
#   --label "finetuned_model" \
#   --checkpoint_path checkpoints/unet-bcl1-hist%3D4-dt%3DFalse-topo%3DFalse-terrain%3DFalse-lc%3D18-oh%3DFalse-sun%3DFalse-img_size%3D512x512/cp.ckpt \
#   --learning_rate 5e-5
#
# 8. Fine-tuning from saved model directory:
# ./train_cloudcast.sh \
#   --dataseries_directory output/patches \
#   --label "finetuned_model" \
#   --checkpoint_path models/original_model/variables/variables
#


#'''
#bash train_cloudcast.sh \
#--label "multi_patch_10min" \
#--loss_function ssim_mae_0.7_0.3 \
#--checkpoint_path '/home/databourg/workspace/cloudcast2/checkpoints/unet-ssim_mae_0.8_0.2-hist=4-lc=18-oh=False-img_size=512x512/model.weights.h5' \
#--dataseries_directory '/home/databourg/workspace/cloudcast/output_test'
#
#'''


# bash train_cloudcast.sh \
#   --dataseries_directory data \
#   --label "ssim" \
#   --checkpoint_path checkpoints/model.weights.h5
#   --loss_function ssim
#   --learning_rate 5e-4