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
  google-auth \
  torch \
  torchvision \
  timm \
  tqdm \
  huggingface_hub

echo ""

# Clone CorridorKey if not present
if [ ! -d "CorridorKey" ]; then
  echo "Cloning CorridorKey green screen keyer..."
  git clone https://github.com/nikopueringer/CorridorKey.git
fi

echo ""
echo "=== Installation complete ==="
echo "Run 'bash run.sh' to start the pipeline."
echo "Note: CorridorKey model (~300MB) auto-downloads on first run."
