#!/bin/bash
# Script to install OpenCV using conda

echo "Installing OpenCV using conda..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Get the current conda environment name
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')

echo "Current conda environment: $CURRENT_ENV"
echo "Installing OpenCV in environment: $CURRENT_ENV"

# Install OpenCV
conda install -y -c conda-forge opencv

# Verify installation
echo "Verifying OpenCV installation..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

if [ $? -eq 0 ]; then
    echo "OpenCV installation successful!"
else
    echo "OpenCV installation verification failed."
    echo "You may need to activate your conda environment and try again."
fi

echo "Done."
