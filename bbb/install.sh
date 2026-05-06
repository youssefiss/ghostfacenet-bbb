#!/bin/bash
# install_bbb.sh — Installation BBB Debian Bookworm
set -e
echo "======================================"
echo "  Installation BBB — GhostFaceNet"
echo "======================================"

# Dépendances système
sudo apt-get update -q
sudo apt-get install -y \
    python3-pip \
    python3-numpy \
    python3-opencv \
    libopencv-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    python3-scipy \
    python3-flask \
    --no-install-recommends

echo "[INFO] Dépendances système installées"

# tflite-runtime + autres via pip avec --break-system-packages
# (nécessaire sur Debian Bookworm)
pip3 install \
    tflite-runtime \
    --break-system-packages \
    --no-cache-dir

echo "[INFO] tflite-runtime installé"

# Vérification
python3 -c "
import cv2, numpy as np, flask, scipy
print('[OK] OpenCV :', cv2.__version__)
print('[OK] NumPy  :', np.__version__)
print('[OK] Flask  :', flask.__version__)
print('[OK] SciPy  : OK')
try:
    import tflite_runtime.interpreter as tflite
    print('[OK] tflite-runtime : OK')
except ImportError:
    print('[WARN] tflite-runtime absent, on utilisera tensorflow.lite')
print()
print('Tout est prêt. Lance : python3 recognize_bbb.py')
"

echo "======================================"
echo "  Installation terminée"
echo "======================================"
