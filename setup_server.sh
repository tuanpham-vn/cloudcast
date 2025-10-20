#!/bin/bash

# Script to setup TensorFlow on VAST.AI server
echo "Connecting to VAST.AI server..."

# Check current environment
echo "Current user: $(whoami)"
echo "Current hostname: $(hostname)"

# Initialize conda
echo "Initializing conda..."
source /opt/conda/etc/profile.d/conda.sh

# Check conda environments
echo "Available conda environments:"
conda info --envs

# Check if cloudcast environment exists
if conda info --envs | grep -q "cloudcast"; then
    echo "Activating cloudcast environment..."
    conda activate cloudcast
else
    echo "Creating cloudcast environment..."
    conda create -n cloudcast python=3.9 -y
    conda activate cloudcast
fi

# Check CUDA version
echo "CUDA version:"
nvcc --version

# Check current TensorFlow installation
echo "Current TensorFlow version:"
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" 2>/dev/null || echo "TensorFlow not installed"

# Install TensorFlow with CUDA 12.8 support (using 2.15.0 as lowest available)
echo "Installing TensorFlow with CUDA 12.8 support..."
pip uninstall tensorflow -y
pip install tensorflow==2.15.0

# Verify installation
echo "Verifying TensorFlow installation:"
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('CUDA available:', tf.test.is_built_with_cuda())
print('GPU available:', tf.test.is_gpu_available())
print('GPU devices:', tf.config.list_physical_devices('GPU'))
"

# Test training capability
echo "Testing training capability..."
python -c "
import tensorflow as tf
import numpy as np

# Create simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate dummy data
x_train = np.random.random((1000, 784))
y_train = np.random.randint(0, 2, (1000, 1))

# Train model
print('Training model...')
history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
print('Training completed successfully!')
print('Final loss:', history.history['loss'][0])
print('Final accuracy:', history.history['accuracy'][0])
"

echo "Setup completed!"
