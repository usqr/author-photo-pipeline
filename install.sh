#!/bin/bash
set -e
echo "Installing Author Photo Pipeline dependencies..."
echo ""

pip3 install --break-system-packages \
  Pillow \
  rembg \
  onnxruntime \
  opencv-python-headless \
  scikit-image \
  numpy \
  scipy \
  google-genai \
  google-auth

echo ""
echo "=== Installation complete ==="
echo "Run 'bash run.sh' to start the pipeline."
