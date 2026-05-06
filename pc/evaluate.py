"""
evaluate.py — Évaluation de la précision de reconnaissance faciale
Modèle   : ghostfacenet_int8.tflite  (INT8 TFLite — identique à app_bbb.py)
Méthode  : Repeated Random Split  (80% train / 20% test, 5 répétitions)
Pipeline : IDENTIQUE à precalculate_embeddings.py
           (quantisation INT8, augmentation miroir, moyenne L2-normalisée)
Sortie   : evaluation_report.png

Pourquoi 5 répétitions ?
  Un seul split peut tomber par hasard sur des photos faciles ou difficiles.
  En répétant 5 fois avec un mélange différent, on obtient un résultat
  plus stable et plus représentatif de la vraie performance.
"""

import os
import cv2
import numpy as np
import random
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc, confusion_matrix
from face_align import align_face, imread_unicode

TFLITE_PATH  = "ghostfacenet_int8.tflite"
KNOWN_DIR    = "known_users"
THRESHOLD    = 0.69
OUTPUT_IMG   = "evaluation_report.png"
SUPPORTED    = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
TEST_RATIO   = 0.20    # 20% des photos pour le test
N_REPEATS    = 5       # nombre de répétitions
RANDOM_SEED  = 42

# ── Charger TFLite ────────────────────────────────────────
print(f"[INFO] Chargement TFLite : {TFLITE_PATH}")
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp_det  = interp.get_input_details()[0]
outp_det = interp.get_output_details()[0]
input_scale       = inp_det['quantization'][0]
input_zero_point  = inp_det['quantization'][1]
output_scale      = outp_det['quantization'][0]
output_zero_point = outp_det['quantization'][1]


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
    return 1.0 - float(np.dot(a, b))


# ── Charger toutes les images ─────────────────────────────
print("[INFO] Chargement des images...")
dataset = {}   # { nom: [(fname, emb, emb_flip), ...] }

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
        emb      = get_embedding(aligned)
        emb_flip = get_embedding(cv2.flip(aligned, 1))
        embs.append((fname, emb, emb_flip))
        print("." if ok else "o", end="", flush=True)
    print()
    if embs:
        dataset[name] = embs
        print(f"  {name} : {len(embs)} image(s) "
              f"→ test≈{max(1,int(len(embs)*TEST_RATIO))} / "
              f"train≈{len(embs)-max(1,int(len(embs)*TEST_RATIO))}")

names_list = sorted(dataset.keys())
print(f"\n[INFO] {len(dataset)} personne(s), "
      f"{sum(len(v) for v in dataset.values())} images au total")
print(f"[INFO] Split : {int(TEST_RATIO*100)}% test / "
      f"{int((1-TEST_RATIO)*100)}% train  ×  {N_REPEATS} répétitions\n")


# ── Repeated Random Split ─────────────────────────────────
all_results    = []   # tous les (true, pred, dist) de toutes les répétitions
all_genuine    = []
all_impostor   = []
per_person_acc = {n: [] for n in names_list}   # accuracy par répétition

for repeat in range(N_REPEATS):
    seed = RANDOM_SEED + repeat
    random.seed(seed)
    print(f"{'─'*55}")
    print(f"  Répétition {repeat+1}/{N_REPEATS}  (seed={seed})")
    print(f"{'─'*55}")

    rep_results  = []
    rep_genuine  = []
    rep_impostor = []
    per_person_ok = {n: {'ok': 0, 'total': 0} for n in names_list}

    # ── Split train/test pour cette répétition ────────────
    split = {}   # { nom: {'train': [...], 'test': [...]} }
    for name, emb_list in dataset.items():
        indices = list(range(len(emb_list)))
        random.shuffle(indices)
        n_test  = max(1, int(len(indices) * TEST_RATIO))
        # S'assurer qu'il reste au moins 1 image en train
        n_test  = min(n_test, len(indices) - 1)
        test_idx  = set(indices[:n_test])
        train_idx = set(indices[n_test:])
        split[name] = {
            'train': [emb_list[i] for i in sorted(train_idx)],
            'test':  [emb_list[i] for i in sorted(test_idx)],
        }
        print(f"  {name:<20} train={len(train_idx)}  test={len(test_idx)}")

    print()

    # ── Construire la galerie (train uniquement) ──────────
    gallery = {}
    for name, s in split.items():
        refs = []
        for (_, e, e_flip) in s['train']:
            refs.append(e)
            refs.append(e_flip)
        if refs:
            mean_emb = np.mean(refs, axis=0)
            gallery[name] = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)

    # ── Tester sur les photos de test ─────────────────────
    for true_name, s in split.items():
        for (fname, test_emb, _) in s['test']:

            best_name = None
            best_dist = float('inf')
            for name, ref_emb in gallery.items():
                dist = cosine_distance(test_emb, ref_emb)
                if name == true_name:
                    rep_genuine.append(dist)
                else:
                    rep_impostor.append(dist)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

            rep_results.append((true_name, best_name, best_dist))
            per_person_ok[true_name]['total'] += 1
            if best_name == true_name and best_dist <= THRESHOLD:
                per_person_ok[true_name]['ok'] += 1

            status = "OK" if best_name == true_name and best_dist <= THRESHOLD else "ERR"
            print(f"  [{status}] {true_name}/{fname}  →  {best_name}  dist={best_dist:.4f}")

    # Accuracy de cette répétition par personne
    for name in names_list:
        s = per_person_ok[name]
        acc = (s['ok'] / s['total'] * 100) if s['total'] > 0 else 0
        per_person_acc[name].append(acc)

    n_ok  = sum(1 for tn, pn, d in rep_results if tn == pn and d <= THRESHOLD)
    n_tot = len(rep_results)
    print(f"\n  → Répétition {repeat+1} : {n_ok}/{n_tot} correctes "
          f"({100*n_ok/n_tot:.1f}%)\n")

    all_results.extend(rep_results)
    all_genuine.extend(rep_genuine)
    all_impostor.extend(rep_impostor)

genuine_dists  = np.array(all_genuine)
impostor_dists = np.array(all_impostor)


# ── Métriques globales (moyennées sur toutes les répétitions) ─
thresholds = np.linspace(0.05, 1.0, 600)
far_arr, frr_arr = [], []
for t in thresholds:
    far_arr.append(np.mean(impostor_dists <= t) * 100)
    frr_arr.append(np.mean(genuine_dists  >  t) * 100)
far_arr = np.clip(np.array(far_arr), 0, 100)
frr_arr = np.clip(np.array(frr_arr), 0, 100)

eer_idx = np.argmin(np.abs(far_arr - frr_arr))
eer_val = (far_arr[eer_idx] + frr_arr[eer_idx]) / 2
eer_thr = thresholds[eer_idx]

far_cur = float(np.mean(impostor_dists <= THRESHOLD) * 100)
frr_cur = float(np.mean(genuine_dists  >  THRESHOLD) * 100)

y_true   = [1]*len(genuine_dists) + [0]*len(impostor_dists)
y_scores = np.concatenate([1-genuine_dists, 1-impostor_dists])
fpr_roc, tpr_roc, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr_roc, tpr_roc)

total   = len(all_results)
correct = sum(1 for tn, pn, d in all_results if tn == pn and d <= THRESHOLD)
tp = correct
fp = sum(1 for tn, pn, d in all_results if tn != pn and d <= THRESHOLD)
fn = sum(1 for tn, pn, d in all_results if tn == pn and d  > THRESHOLD)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0
accuracy  = correct / total

pred_names_flat = [pn if d <= THRESHOLD else "Inconnu"
                   for _, pn, d in all_results]
true_names_flat = [r[0] for r in all_results]
labels_cm = names_list + (["Inconnu"] if "Inconnu" in pred_names_flat else [])
cm = confusion_matrix(true_names_flat, pred_names_flat, labels=labels_cm)

# Accuracy par personne = moyenne des 5 répétitions ± écart-type
person_mean = {n: float(np.mean(per_person_acc[n])) for n in names_list}
person_std  = {n: float(np.std(per_person_acc[n]))  for n in names_list}
person_sorted = dict(sorted(person_mean.items(), key=lambda x: x[1]))


# ── Figure ────────────────────────────────────────────────
print("\n[INFO] Génération du rapport visuel...")

plt.style.use('dark_background')
fig = plt.figure(figsize=(22, 14), facecolor='#0d0d0d')
fig.suptitle(
    f"Evaluation GhostFaceNet + MediaPipe + Alignement FaceMesh"
    f"  |  Split {int(TEST_RATIO*100)}/{int((1-TEST_RATIO)*100)}  ×  {N_REPEATS} répétitions",
    fontsize=13, color='white', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.40, wspace=0.32,
                       left=0.06, right=0.97,
                       top=0.93, bottom=0.05)

CYAN   = '#00e5ff'
GREEN  = '#00e676'
RED    = '#ff1744'
ORANGE = '#ff9100'
YELLOW = '#ffd600'
GREY   = '#424242'


# 1 — FAR / FRR vs Seuil
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#111111')
ax1.plot(thresholds, far_arr, color=RED,   lw=2, label='FAR')
ax1.plot(thresholds, frr_arr, color=GREEN, lw=2, label='FRR')
ax1.axvline(eer_thr,   color=YELLOW, lw=1.2, linestyle='--',
            label=f'EER={eer_val:.1f}% @ {eer_thr:.2f}')
ax1.axvline(THRESHOLD, color=CYAN,   lw=1.2, linestyle=':',
            label=f'Seuil={THRESHOLD}')
ax1.set_title('FAR / FRR vs Seuil', color=CYAN, fontsize=11)
ax1.set_xlabel('Seuil (distance cosinus)', color='white', fontsize=8)
ax1.set_ylabel('%', color='white', fontsize=8)
ax1.set_ylim(0, 105)
ax1.set_xlim(thresholds[0], thresholds[-1])
ax1.tick_params(colors='white', labelsize=7)
ax1.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='white')
for sp in ax1.spines.values(): sp.set_edgecolor(GREY)


# 2 — Distribution Genuine vs Impostor
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#111111')
bins = np.linspace(
    min(genuine_dists.min(), impostor_dists.min()) - 0.02,
    max(genuine_dists.max(), impostor_dists.max()) + 0.02,
    80)
ax2.hist(genuine_dists,  bins=bins, color=GREEN, alpha=0.7,
         label=f'Genuine (n={len(genuine_dists)})')
ax2.hist(impostor_dists, bins=bins, color=RED,   alpha=0.6,
         label=f'Impostor (n={len(impostor_dists)})')
ax2.axvline(THRESHOLD, color=CYAN, lw=1.5, linestyle='--',
            label=f'Seuil={THRESHOLD}')
ax2.set_title('Distribution Genuine vs Impostor', color=CYAN, fontsize=11)
ax2.set_xlabel('Distance cosinus', color='white', fontsize=8)
ax2.set_ylabel('Nombre de paires', color='white', fontsize=8)
ax2.tick_params(colors='white', labelsize=7)
ax2.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='white')
for sp in ax2.spines.values(): sp.set_edgecolor(GREY)


# 3 — Matrice de Confusion
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('#0d0d0d')
n_cls  = len(labels_cm)
cm_norm = cm.astype(float)
row_sums = cm_norm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cm_norm /= row_sums
cmap_blue = LinearSegmentedColormap.from_list('blk_blue',
            ['#0d0d0d','#003366','#0055aa','#00aaff'])
ax3.imshow(cm_norm, cmap=cmap_blue, aspect='auto', vmin=0, vmax=1)
fs = max(3.5, min(6, 80/n_cls))
ax3.set_xticks(range(n_cls))
ax3.set_yticks(range(n_cls))
ax3.set_xticklabels(labels_cm, rotation=90, fontsize=fs, color='white')
ax3.set_yticklabels(labels_cm, fontsize=fs, color='white')
for i in range(n_cls):
    for j in range(n_cls):
        val = cm[i, j]
        if val > 0:
            ax3.text(j, i, str(val), ha='center', va='center',
                     fontsize=max(3, fs-1),
                     color='white' if cm_norm[i,j] < 0.6 else '#001122')
ax3.set_title('Matrice de Confusion (cumulée)', color=CYAN, fontsize=11)
ax3.tick_params(colors='white', labelsize=fs)
for sp in ax3.spines.values(): sp.set_edgecolor(GREY)


# 4 — Métriques
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor('#111111')
ax4.axis('off')
ax4.set_title('Métriques @ seuil actuel', color=CYAN, fontsize=11, pad=10)
metrics = [
    ("Accuracy",      f"{accuracy*100:.1f}%",  GREEN),
    ("Précision",     f"{precision*100:.1f}%", GREEN),
    ("Rappel",        f"{recall*100:.1f}%",    GREEN),
    ("F1-score",      f"{f1*100:.1f}%",        GREEN),
    ("FAR",           f"{far_cur:.2f}%",        RED),
    ("FRR",           f"{frr_cur:.2f}%",        RED),
    ("EER",           f"{eer_val:.2f}%",        ORANGE),
    ("Seuil EER",     f"{eer_thr:.3f}",         YELLOW),
    ("Seuil actuel",  f"{THRESHOLD:.3f}",       CYAN),
    ("Répétitions",   f"{N_REPEATS}×",          GREY),
]
y0 = 0.95
for label, value, color in metrics:
    ax4.text(0.05, y0, label, transform=ax4.transAxes,
             fontsize=10, color='white', va='top')
    ax4.text(0.72, y0, value, transform=ax4.transAxes,
             fontsize=10, color=color, va='top', fontweight='bold')
    y0 -= 0.093


# 5 — Courbe ROC
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor('#111111')
ax5.plot(fpr_roc*100, tpr_roc*100, color=CYAN, lw=2,
         label=f'AUC={roc_auc:.4f}')
ax5.plot([0,100],[0,100], color=GREY, lw=1, linestyle='--')
ax5.set_title(f'Courbe ROC (AUC={roc_auc:.4f})', color=CYAN, fontsize=11)
ax5.set_xlabel('FAR (%)', color='white', fontsize=8)
ax5.set_ylabel('TAR (%) = 1 - FRR', color='white', fontsize=8)
ax5.tick_params(colors='white', labelsize=7)
ax5.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='white',
           loc='lower right')
ax5.set_xlim(0, 100)
ax5.set_ylim(0, 100)
for sp in ax5.spines.values(): sp.set_edgecolor(GREY)


# 6 — Accuracy par personne (moyenne ± std sur N_REPEATS)
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#111111')
pnames  = list(person_sorted.keys())
pmeans  = [person_mean[n] for n in pnames]
pstds   = [person_std[n]  for n in pnames]
colors_bar = [GREEN if v >= 95 else ORANGE if v >= 80 else RED for v in pmeans]
bars = ax6.barh(pnames, pmeans, xerr=pstds,
                color=colors_bar, height=0.65,
                error_kw=dict(ecolor='white', capsize=3, lw=1.2))
for bar, val, std in zip(bars, pmeans, pstds):
    ax6.text(min(val + std + 1, 102),
             bar.get_y() + bar.get_height()/2,
             f'{val:.0f}%', va='center', ha='left',
             fontsize=7, color='white')
ax6.set_xlim(0, 115)
ax6.set_title(f'Accuracy par personne  (moy. ± std  ×{N_REPEATS})',
              color=CYAN, fontsize=11)
ax6.set_xlabel('Accuracy (%)', color='white', fontsize=8)
fs_y = max(5, min(9, 200/max(len(pnames), 1)))
ax6.tick_params(colors='white', labelsize=fs_y)
for sp in ax6.spines.values(): sp.set_edgecolor(GREY)


# ── Sauvegarder ───────────────────────────────────────────
plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print(f"\n[OK] Rapport sauvegardé : {OUTPUT_IMG}")
print(f"\n{'═'*55}")
print(f"  RÉSULTATS FINAUX  ({N_REPEATS} répétitions, split {int(TEST_RATIO*100)}/{int((1-TEST_RATIO)*100)})")
print(f"{'═'*55}")
print(f"  Total tests     : {total}  ({total//N_REPEATS} par répétition)")
print(f"  ✓ Correctes     : {correct}  ({accuracy*100:.1f}%)")
print(f"  Précision       : {precision*100:.1f}%")
print(f"  Rappel          : {recall*100:.1f}%")
print(f"  F1-score        : {f1*100:.1f}%")
print(f"  FAR             : {far_cur:.2f}%")
print(f"  FRR             : {frr_cur:.2f}%")
print(f"  EER             : {eer_val:.2f}%  @ seuil {eer_thr:.3f}")
print(f"  AUC-ROC         : {roc_auc:.4f}")
print(f"  Seuil utilisé   : {THRESHOLD}")
print(f"\n  Accuracy par personne (moyenne ± std) :")
for n in sorted(person_mean, key=person_mean.get):
    bar = "█" * int(person_mean[n] / 5)
    print(f"    {n:<22} {person_mean[n]:5.1f}% ± {person_std[n]:.1f}%  {bar}")
print(f"{'═'*55}")