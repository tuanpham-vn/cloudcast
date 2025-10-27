#!/bin/bash


# Kích hoạt môi trường
echo "Activating cloudcast environment..."
conda activate cloudcast


bash train_cloudcast.sh \
  --dataseries_directory data \
  --loss_function "ssim" \
  --sequence_offset_minutes 0 \
  --learning_rate 1e-4 \
  --checkpoint_path checkpoints/model.weights.h5