"""
evaluate.py — Évaluation de la précision de reconnaissance faciale
Méthode : Leave-One-Out Cross-Validation (LOOCV)
Pour chaque photo : on l'exclut, on calcule les embeddings du reste,
puis on teste si elle est correctement reconnue.
"""

import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from face_align import align_face, imread_unicode

TFLITE_PATH = "ghostfacenet_int8.tflite"
KNOWN_DIR   = "known_users"
THRESHOLD   = 0.5   # Seuil cosinus (< seuil = reconnu)
SUPPORTED   = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

# ── Charger TFLite ────────────────────────────────────────
print(f"[INFO] Chargement TFLite : {TFLITE_PATH}")
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp_det  = interp.get_input_details()[0]
outp_det = interp.get_output_details()[0]
input_scale      = inp_det['quantization'][0]
input_zero_point = inp_det['quantization'][1]
output_scale     = outp_det['quantization'][0]
output_zero_point= outp_det['quantization'][1]


def get_embedding(aligned_face):
    img = cv2.resize(aligned_face, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img_int8 = (img / input_scale + input_zero_point).astype(np.int8)
    img_int8 = np.expand_dims(img_int8, axis=0)
    interp.set_tensor(inp_det['index'], img_int8)
    interp.invoke()
    output = interp.get_tensor(outp_det['index'])
    emb = (output.astype(np.float32) - output_zero_point) * output_scale
    emb = emb.flatten()
    return emb / (np.linalg.norm(emb) + 1e-6)


def cosine_distance(a, b):
    return 1.0 - np.dot(a, b)


# ── Charger toutes les images ─────────────────────────────
print("[INFO] Chargement des images...")
dataset = {}  # { nom: [emb1, emb2, ...] }

persons = sorted([
    d for d in os.listdir(KNOWN_DIR)
    if os.path.isdir(os.path.join(KNOWN_DIR, d)) and not d.startswith('.')
])

for person_dir in persons:
    name = person_dir.capitalize()
    path = os.path.join(KNOWN_DIR, person_dir)
    imgs = sorted([f for f in os.listdir(path) if f.lower().endswith(SUPPORTED)])

    embs = []
    for fname in imgs:
        img = imread_unicode(os.path.join(path, fname))
        if img is None:
            continue
        aligned, ok = align_face(img)
        emb = get_embedding(aligned)
        embs.append((fname, emb))

    if embs:
        dataset[name] = embs
        print(f"  {name} : {len(embs)} image(s) chargée(s)")

print(f"\n[INFO] {len(dataset)} personne(s), "
      f"{sum(len(v) for v in dataset.values())} images au total")


# ── Leave-One-Out Cross-Validation ────────────────────────
print("\n" + "─" * 55)
print("  ÉVALUATION  (Leave-One-Out Cross-Validation)")
print("─" * 55)

total      = 0
correct    = 0
incorrect  = 0
unknown    = 0

errors = []  # liste des erreurs pour affichage final

for true_name, emb_list in dataset.items():
    for i, (fname, test_emb) in enumerate(emb_list):

        # ── Construire la galerie SANS cette image ────────
        gallery = {}
        for name, others in dataset.items():
            refs = [e for j, (_, e) in enumerate(others) if not (name == true_name and j == i)]
            if refs:
                mean_emb = np.mean(refs, axis=0)
                gallery[name] = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)

        # ── Comparer avec la galerie ──────────────────────
        best_name = None
        best_dist = float('inf')

        for name, ref_emb in gallery.items():
            dist = cosine_distance(test_emb, ref_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        # ── Évaluer le résultat ───────────────────────────
        total += 1
        if best_dist > THRESHOLD:
            result = "INCONNU"
            unknown += 1
            errors.append((true_name, fname, result, best_dist))
        elif best_name == true_name:
            result = "✓"
            correct += 1
        else:
            result = f"✗ → {best_name}"
            incorrect += 1
            errors.append((true_name, fname, result, best_dist))

        status = "OK" if best_name == true_name and best_dist <= THRESHOLD else "ERREUR"
        print(f"  [{status:6}] {true_name}/{fname:<25} "
              f"prédit={best_name or '?':<12} dist={best_dist:.4f}")


# ── Résultats finaux ──────────────────────────────────────
print("\n" + "═" * 55)
print("  RÉSULTATS")
print("═" * 55)
print(f"  Total images testées  : {total}")
print(f"  ✓ Correctes           : {correct}  ({100*correct/total:.1f}%)")
print(f"  ✗ Mauvaise identité   : {incorrect}  ({100*incorrect/total:.1f}%)")
print(f"  ? Non reconnus        : {unknown}  ({100*unknown/total:.1f}%)")
print(f"\n  Précision globale     : {100*correct/total:.1f}%")
print("═" * 55)

if errors:
    print("\n  Détail des erreurs :")
    for true_name, fname, result, dist in errors:
        print(f"    • {true_name}/{fname} → {result}  (dist={dist:.4f})")

print(f"\n[INFO] Seuil utilisé : {THRESHOLD} "
      f"(augmenter = plus tolérant, diminuer = plus strict)")