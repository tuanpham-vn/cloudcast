#!/bin/bash

# CloudCast Training Script với tất cả tham số mặc định
cd cloudcast
# Kiểm tra môi trường conda đã tồn tại chưa
if conda env list | grep -q "cloudcast"; then
    echo "Environment 'cloudcast' already exists."
    read -p "Do you want to recreate it? This will remove the existing environment. (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment 'cloudcast'..."
        conda env remove -n cloudcast -y
        echo "Creating new environment from cloudcast-environment.yml..."
        conda env create -f cloudcast-environment.yml
    else
        echo "Using existing environment 'cloudcast'..."
    fi
else
    echo "Creating new environment 'cloudcast' from cloudcast-environment.yml..."
    conda env create -f cloudcast-environment.yml
fi

# Kích hoạt môi trường
echo "Activating cloudcast environment..."
conda deactivate
conda activate cloudcast
