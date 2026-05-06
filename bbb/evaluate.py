"""
evaluate_bbb.py — Évaluation de précision sur BeagleBone Black
Modèle   : ghostfacenet_int8.tflite
Méthode  : Repeated Random Split (80% train / 20% test, 5 répétitions)

Pipeline IDENTIQUE à main.py :
  - FaceMesh (face_align_lite.py) pour détecter et aligner le visage
  - known_embeddings.pkl comme galerie fixe (calculé avec align_face)
  - 2 seuils : THRESHOLD_HIGH=0.50 (high) et THRESHOLD_LOW=0.605 (low)
  - cosine_similarity (dot / norms)

Copier sur BBB :
  scp evaluate_bbb.py debian@BBB_IP:~/ghost2/
  scp -r known_users  debian@BBB_IP:~/ghost2/
  ssh debian@BBB_IP
  cd ~/ghost2
  python3 evaluate_bbb.py
"""

import os
import cv2
import numpy as np
import pickle
import random
import time

# ── Charger TFLite ────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print("[INFO] tflite_runtime chargé.")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("[INFO] tensorflow.lite chargé.")

# ── FaceMesh — identique à main.py ───────────────────────
from face_align_lite import align_face
print("[INFO] FaceMesh chargé via face_align_lite.py")

# ── CONFIG — identique à main.py ─────────────────────────
TFLITE_MODEL   = "ghostfacenet_int8.tflite"
EMBEDDINGS_PKL = "known_embeddings.pkl"
KNOWN_DIR      = "known_users"
OUTPUT_TXT     = "evaluation_results.txt"
SUPPORTED      = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

THRESHOLD_HIGH = 0.650
THRESHOLD_LOW  = 0.80
TEST_RATIO     = 0.20
N_REPEATS      = 5
RANDOM_SEED    = 42


# ── Charger TFLite ────────────────────────────────────────
print(f"\n[INFO] Chargement TFLite : {TFLITE_MODEL}")
interpreter = Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale,  input_zero_point  = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']
print(f"  Input  : scale={input_scale:.6f}  zp={input_zero_point}")
print(f"  Output : scale={output_scale:.6f}  zp={output_zero_point}")


# ── get_embedding — identique à main.py ──────────────────
def get_embedding(roi):
    img = cv2.resize(roi, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img_int8 = (img / input_scale + input_zero_point).astype(np.int8)
    img_int8 = np.expand_dims(img_int8, axis=0)

    t0 = time.time()
    interpreter.set_tensor(input_details[0]['index'], img_int8)
    interpreter.invoke()
    elapsed = time.time() - t0

    output = interpreter.get_tensor(output_details[0]['index'])
    emb = (output.astype(np.float32) - output_zero_point) * output_scale
    emb = emb.flatten()
    return emb / (np.linalg.norm(emb) + 1e-6), elapsed


# ── cosine_similarity — identique à main.py ──────────────
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


# ── Charger known_embeddings.pkl ─────────────────────────
print(f"\n[INFO] Chargement {EMBEDDINGS_PKL}...")
with open(EMBEDDINGS_PKL, 'rb') as f:
    pkl_data = pickle.load(f)
known_embeddings = pkl_data['embeddings']
known_names      = pkl_data['names']
print(f"  {pkl_data['n_persons']} personne(s) : {', '.join(known_names)}")


# ── Charger les images avec FaceMesh ─────────────────────
print(f"\n[INFO] Chargement images depuis {KNOWN_DIR}...")
print(f"[INFO] Détection : FaceMesh (face_align_lite) = main.py")

dataset         = {}
inference_times = []
n_no_face       = 0

persons = sorted([
    d for d in os.listdir(KNOWN_DIR)
    if os.path.isdir(os.path.join(KNOWN_DIR, d)) and not d.startswith('.')
])

for person_dir in persons:
    name = person_dir.capitalize()
    path = os.path.join(KNOWN_DIR, person_dir)
    imgs = sorted([f for f in os.listdir(path)
                   if f.lower().endswith(SUPPORTED)])
    embs = []

    for fname in imgs:
        img = cv2.imread(os.path.join(path, fname))
        if img is None:
            print(f"  [WARN] Impossible de lire {fname}")
            continue

        # align_face — identique à main.py
        aligned, ok = align_face(img)

        if not ok:
            n_no_face += 1
            # Fallback : image entière si FaceMesh ne détecte rien
            aligned = img

        emb, t = get_embedding(aligned)
        inference_times.append(t)
        embs.append((fname, emb, ok))
        print("." if ok else "o", end="", flush=True)

    print()
    if embs:
        dataset[name] = embs
        n_test = max(1, int(len(embs) * TEST_RATIO))
        n_test = min(n_test, len(embs) - 1)
        n_ok_face = sum(1 for _, _, ok in embs if ok)
        print(f"  {name} : {len(embs)} images  "
              f"(FaceMesh OK={n_ok_face}/{len(embs)})  "
              f"test={n_test} / train={len(embs)-n_test}")

names_list = sorted(dataset.keys())
avg_ms = np.mean(inference_times) * 1000 if inference_times else 0
min_ms = np.min(inference_times)  * 1000 if inference_times else 0
max_ms = np.max(inference_times)  * 1000 if inference_times else 0

print(f"\n[INFO] Inférence BBB : "
      f"moy={avg_ms:.0f}ms  min={min_ms:.0f}ms  max={max_ms:.0f}ms")
if n_no_face > 0:
    print(f"[WARN] {n_no_face} image(s) sans visage FaceMesh → image entière utilisée")
print(f"[INFO] {len(dataset)} personne(s)  "
      f"{sum(len(v) for v in dataset.values())} images chargées")
print(f"[INFO] Seuils : HIGH={THRESHOLD_HIGH}  LOW={THRESHOLD_LOW}")
print(f"[INFO] Split {int(TEST_RATIO*100)}/{int((1-TEST_RATIO)*100)}"
      f"  x  {N_REPEATS} répétitions\n")


# ── Repeated Random Split ─────────────────────────────────
all_results    = []
all_genuine    = []
all_impostor   = []
per_person_acc = {n: [] for n in names_list}
repeat_acc     = []

for repeat in range(N_REPEATS):
    seed = RANDOM_SEED + repeat
    random.seed(seed)
    print(f"{'─'*55}")
    print(f"  Répétition {repeat+1}/{N_REPEATS}  (seed={seed})")
    print(f"{'─'*55}")

    rep_results   = []
    per_person_ok = {n: {'ok': 0, 'total': 0} for n in names_list}

    # Split test/train
    split_test = {}
    for name, emb_list in dataset.items():
        idx = list(range(len(emb_list)))
        random.shuffle(idx)
        n_test = max(1, int(len(idx) * TEST_RATIO))
        n_test = min(n_test, len(idx) - 1)
        split_test[name] = [emb_list[i] for i in sorted(idx[:n_test])]
        print(f"  {name:<22} test={len(split_test[name])}")

    print()

    # Test contre known_embeddings.pkl (galerie fixe = production)
    for true_name, test_list in split_test.items():
        for (fname, test_emb, face_ok) in test_list:

            # cosine_similarity — identique à main.py
            sims     = [cosine_similarity(test_emb, e)
                        for e in known_embeddings]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            dist     = 1.0 - best_sim
            pred_name_raw = known_names[best_idx]

            # 2 seuils — identique à main.py
            if dist < THRESHOLD_HIGH:
                pred_name = pred_name_raw
                level     = 'high'
            elif dist < THRESHOLD_LOW:
                pred_name = pred_name_raw + "?"
                level     = 'low'
            else:
                pred_name = "Inconnu"
                level     = 'unknown'

            # Distances genuine / impostor
            for i, sim in enumerate(sims):
                d = 1.0 - sim
                if known_names[i] == true_name:
                    all_genuine.append(d)
                else:
                    all_impostor.append(d)

            correct = (pred_name_raw == true_name and level != 'unknown')
            all_results.append((true_name, pred_name_raw, dist, level, correct))
            rep_results.append((true_name, pred_name_raw, dist, level, correct))
            per_person_ok[true_name]['total'] += 1
            if correct:
                per_person_ok[true_name]['ok'] += 1

            if correct and level == 'high':
                tag = "OK-H"
            elif correct and level == 'low':
                tag = "OK-L"
            elif level == 'unknown':
                tag = "UNK "
            else:
                tag = "ERR "

            face_tag = "" if face_ok else " [no-mesh]"
            print(f"  [{tag}] {true_name}/{fname:<25} "
                  f"-> {pred_name:<22} dist={dist:.4f}{face_tag}")

    for name in names_list:
        s   = per_person_ok[name]
        acc = (s['ok'] / s['total'] * 100) if s['total'] > 0 else 0
        per_person_acc[name].append(acc)

    n_ok  = sum(1 for *_, c in rep_results if c)
    n_tot = len(rep_results)
    r_acc = 100 * n_ok / n_tot
    repeat_acc.append(r_acc)
    print(f"\n  -> Répétition {repeat+1} : {n_ok}/{n_tot}  ({r_acc:.1f}%)\n")


# ── Métriques finales ─────────────────────────────────────
genuine_dists  = np.array(all_genuine)
impostor_dists = np.array(all_impostor)

total   = len(all_results)
correct = sum(1 for *_, c in all_results if c)
n_high  = sum(1 for _, _, _, l, c in all_results if l == 'high' and c)
n_low   = sum(1 for _, _, _, l, c in all_results if l == 'low'  and c)
n_unk   = sum(1 for _, _, _, l, _ in all_results if l == 'unknown')
n_wrong = sum(1 for tn, pn, _, l, _ in all_results
              if l != 'unknown' and pn != tn)

tp = correct
fn = sum(1 for *_, c in all_results if not c)
fp = n_wrong
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0
accuracy  = correct / total

far_high = float(np.mean(impostor_dists < THRESHOLD_HIGH) * 100)
frr_high = float(np.mean(genuine_dists  >= THRESHOLD_HIGH) * 100)
far_low  = float(np.mean(impostor_dists < THRESHOLD_LOW)  * 100)
frr_low  = float(np.mean(genuine_dists  >= THRESHOLD_LOW)  * 100)

thresholds_arr = np.linspace(0.05, 1.0, 600)
far_a = np.clip([np.mean(impostor_dists < t)*100 for t in thresholds_arr], 0, 100)
frr_a = np.clip([np.mean(genuine_dists  >= t)*100 for t in thresholds_arr], 0, 100)
eer_i   = np.argmin(np.abs(np.array(far_a) - np.array(frr_a)))
eer_val = (far_a[eer_i] + frr_a[eer_i]) / 2
eer_thr = thresholds_arr[eer_i]

person_mean = {n: float(np.mean(per_person_acc[n])) for n in names_list}
person_std  = {n: float(np.std(per_person_acc[n]))  for n in names_list}


# ── Rapport texte ─────────────────────────────────────────
SEP = "=" * 58
sep = "-" * 58

lines = [
    SEP,
    f"  RESULTATS BBB  ({N_REPEATS} rep., split "
    f"{int(TEST_RATIO*100)}/{int((1-TEST_RATIO)*100)})",
    SEP,
    f"  Plateforme      : BeagleBone Black",
    f"  Modele          : {TFLITE_MODEL}",
    f"  Detection       : FaceMesh (face_align_lite.py)",
    f"  Galerie         : {EMBEDDINGS_PKL} (fixe, calculé avec FaceMesh)",
    f"  Pipeline        : IDENTIQUE a main.py",
    f"  Inference moy.  : {avg_ms:.0f} ms  "
    f"(min={min_ms:.0f}  max={max_ms:.0f})",
    f"  Images no-mesh  : {n_no_face} (image entiere utilisee en fallback)",
    f"  Total tests     : {total}  ({total//N_REPEATS} par rep.)",
    sep,
    "  METRIQUES GLOBALES :",
    f"  Accuracy        : {accuracy*100:.1f}%  "
    f"(moy rep: {np.mean(repeat_acc):.1f}% +/- {np.std(repeat_acc):.1f}%)",
    f"  Precision       : {precision*100:.1f}%",
    f"  Rappel          : {recall*100:.1f}%",
    f"  F1-score        : {f1*100:.1f}%",
    sep,
    "  DETAIL PAR NIVEAU (comme main.py) :",
    f"  OK haute conf.  : {n_high}   (dist < {THRESHOLD_HIGH})",
    f"  OK faible conf. : {n_low}   (dist < {THRESHOLD_LOW}, label='nom?')",
    f"  Inconnus        : {n_unk}",
    f"  Mauvais nom     : {n_wrong}",
    sep,
    "  FAR / FRR :",
    f"  @ seuil HIGH {THRESHOLD_HIGH} : FAR={far_high:.2f}%  FRR={frr_high:.2f}%",
    f"  @ seuil LOW  {THRESHOLD_LOW} : FAR={far_low:.2f}%  FRR={frr_low:.2f}%",
    f"  EER             : {eer_val:.2f}%  @ seuil {eer_thr:.3f}",
    sep,
    "  Accuracy par rep. :",
]
for i, acc in enumerate(repeat_acc):
    lines.append(f"    Rep {i+1} : {acc:.1f}%")

lines += [sep, "  Accuracy par personne (moy +/- std) :"]
for n in sorted(person_mean, key=person_mean.get):
    bar  = "#" * int(person_mean[n] / 5)
    warn = "!" if person_mean[n] < 95 or person_std[n] > 5 else " "
    lines.append(f"  {warn} {n:<24} "
                 f"{person_mean[n]:5.1f}% +/- {person_std[n]:.1f}%  {bar}")

lines += [
    SEP,
    "",
    "  LEGENDE :",
    "  [OK-H]     = reconnu haute confiance (dist < 0.50)",
    "  [OK-L]     = reconnu faible confiance (dist < 0.605)",
    "  [UNK ]     = non reconnu (dist >= 0.605)",
    "  [ERR ]     = mauvaise identite",
    "  [no-mesh]  = FaceMesh n'a pas detecte de visage",
    SEP,
]

report = "\n".join(lines)
print("\n" + report)

with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n[OK] Resultats sauvegardes : {OUTPUT_TXT}")
print(f"\nPour recuperer sur PC :")
print(f"  scp debian@BBB_IP:~/ghost2/{OUTPUT_TXT} .")
