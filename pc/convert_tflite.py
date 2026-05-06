"""
convert_tflite.py — Conversion GhostFaceNet H5 → TFLite INT8
Entrée/Sortie : INT8 (identique au convert_tflite.py qui fonctionne)
Usage : python convert_tflite.py
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import os

# ─── CONFIG ───────────────────────────────────────────────
MODEL_H5_PATH   = "ghostfacenet.h5"
OUTPUT_TFLITE   = "ghostfacenet_int8.tflite"
CALIBRATION_DIR = "known_users"
IMG_SIZE        = (112, 112)
N_CALIBRATION   = 100
SUPPORTED       = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
# ──────────────────────────────────────────────────────────

def imread_safe(path):
    with open(path, 'rb') as f:
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

print(f"\n{'='*55}")
print("  CONVERSION GHOSTFACENET → TFLITE INT8")
print(f"{'='*55}\n")

# 1. Charger modèle
print(f"[INFO] Chargement : {MODEL_H5_PATH}")
model = tf.keras.models.load_model(MODEL_H5_PATH)
print(f"[INFO] Input  : {model.input_shape}")
print(f"[INFO] Output : {model.output_shape}")

# 2. Dataset calibration
# Preprocessing IDENTIQUE à main.py : /255 puis (x-0.5)/0.5
print(f"\n[INFO] Collecte images calibration depuis '{CALIBRATION_DIR}/'...")
calib_images = []

for root, dirs, files in os.walk(CALIBRATION_DIR):
    for fname in files:
        if not fname.lower().endswith(SUPPORTED):
            continue
        img = imread_safe(os.path.join(root, fname))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5              # même normalisation que main.py
        calib_images.append(img)
        if len(calib_images) >= N_CALIBRATION:
            break
    if len(calib_images) >= N_CALIBRATION:
        break

print(f"[INFO] {len(calib_images)} images de calibration")

def representative_dataset():
    for img in calib_images:
        yield [np.expand_dims(img, axis=0)]

# 3. Conversion INT8
print(f"\n[INFO] Conversion INT8 (1-2 min)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations              = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset     = representative_dataset
converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type       = tf.int8    # ← INT8 entrée
converter.inference_output_type      = tf.int8    # ← INT8 sortie

tflite_model = converter.convert()

with open(OUTPUT_TFLITE, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(OUTPUT_TFLITE) / 1024 / 1024
print(f"[OK] {OUTPUT_TFLITE}  ({size_mb:.2f} MB)")

# 4. Vérification
print(f"\n[INFO] Vérification...")
interp = tf.lite.Interpreter(model_path=OUTPUT_TFLITE)
interp.allocate_tensors()
inp  = interp.get_input_details()[0]
outp = interp.get_output_details()[0]

print(f"[INFO] Input  : shape={inp['shape']}  dtype={inp['dtype'].__name__}"
      f"  scale={inp['quantization'][0]:.6f}  zp={inp['quantization'][1]}")
print(f"[INFO] Output : shape={outp['shape']} dtype={outp['dtype'].__name__}"
      f"  scale={outp['quantization'][0]:.6f}  zp={outp['quantization'][1]}")

# Test sur image noire
test = np.zeros(inp['shape'], dtype=np.int8)
interp.set_tensor(inp['index'], test)
interp.invoke()
out = interp.get_tensor(outp['index'])
print(f"[INFO] Test OK → output shape : {out.shape}")

print(f"\n{'='*55}")
print(f"  Fichier : {OUTPUT_TFLITE}  ({size_mb:.2f} MB)")
print(f"  → Lance : python precalculate_embeddings.py")
print(f"{'='*55}\n")