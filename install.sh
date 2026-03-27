#!/bin/bash
set -e
echo "Installing Author Photo Pipeline dependencies..."
echo ""

pip3 install --break-system-packages \
  Pillow \
  rembg \
  onnxruntime \
  super-image \
  opencv-python-headless \
  scikit-image \
  numpy \
  scipy \
  torch \
  torchvision

echo ""
echo "=== Installation complete ==="
echo "Run 'bash run.sh' to start the pipeline."
