"""
calibrate_threshold.py — Trouve le seuil optimal pour rejeter les inconnus
Structure attendue :
    known_users/     ← personnes enrôlées (déjà utilisé)
    unknown_users/   ← personnes inconnues (nouveaux visages jamais enrôlés)
        inconnu1.jpg
        inconnu2.jpg
        ...

Usage : python calibrate_threshold.py
"""

import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from scipy.spatial.distance import cosine
from face_align import align_face, imread_unicode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH   = "ghostfacenet.h5"
DB_PATH      = "face_db.pkl"
UNKNOWN_DIR  = "unknown_users"
OUTPUT_DIR   = "evaluation_results"
INPUT_SIZE   = (112, 112)
# ──────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
SUPPORTED = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def load_model(path):
    print(f"[INFO] Chargement modèle : {path}")
    return tf.keras.models.load_model(path)


def preprocess(aligned):
    face = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims((face - 127.5) / 128.0, axis=0)


def get_embedding(model, aligned):
    emb = model.predict(preprocess(aligned), verbose=0)[0]
    return emb / np.linalg.norm(emb)


def best_distance(query_emb, database):
    """Retourne la distance cosine avec la personne la plus proche."""
    best = float('inf')
    best_name = "?"
    for name, embs in database.items():
        dists = [cosine(query_emb, e) for e in embs]
        score = float(np.mean(sorted(dists)[:3]))
        if score < best:
            best = score
            best_name = name
    return best, best_name


def collect_known_distances(model, database):
    """
    Distances des personnes CONNUES vs leur propre identité.
    Ces distances doivent être PETITES → genuine scores.
    On utilise LOOCV : chaque image testée sans elle-même dans la base.
    """
    from itertools import combinations
    print("\n[INFO] Calcul distances genuines (personnes connues)...")

    genuine = []
    all_names = list(database.keys())

    for name in all_names:
        embs = database[name]
        for i, j in combinations(range(len(embs)), 2):
            d = cosine(embs[i], embs[j])
            genuine.append(d)

    print(f"       {len(genuine)} paires genuines")
    return np.array(genuine)


def collect_unknown_distances(model, database, unknown_dir):
    """
    Distances des INCONNUS vs la base.
    Ces distances doivent être GRANDES → impostor scores.
    """
    print(f"\n[INFO] Calcul distances inconnues depuis '{unknown_dir}/'...")

    # Collecter toutes les images (plat ou sous-dossiers)
    files = []
    for root, dirs, fnames in os.walk(unknown_dir):
        for f in fnames:
            if f.lower().endswith(SUPPORTED):
                files.append(os.path.join(root, f))

    if not files:
        print(f"[ERREUR] Aucune image dans {unknown_dir}/")
        return np.array([])

    print(f"       {len(files)} image(s) inconnue(s) trouvée(s)")

    unknown_dists = []
    matched_names = []

    for fpath in files:
        img = imread_unicode(fpath)
        if img is None:
            continue

        aligned, _ = align_face(img)
        emb = get_embedding(model, aligned)
        dist, matched = best_distance(emb, database)
        unknown_dists.append(dist)
        matched_names.append((os.path.basename(fpath), dist, matched))
        print(f"    {os.path.basename(fpath):30s}  dist={dist:.3f}  → reconnu comme: {matched}")

    print(f"\n       {len(unknown_dists)} inconnus traités")
    return np.array(unknown_dists), matched_names


def find_optimal_threshold(genuine_dists, unknown_dists):
    """
    Cherche le seuil T qui maximise :
      - Acceptation des genuins  (dist < T)
      - Rejet des inconnus       (dist >= T)

    Métriques calculées pour chaque T :
      TAR = True Accept Rate  = genuins acceptés / total genuins
      FAR = False Accept Rate = inconnus acceptés / total inconnus
      TRR = True Reject Rate  = inconnus rejetés / total inconnus

    Seuil optimal = maximise (TAR + TRR) / 2
                  = minimise |TAR - (1 - FAR)|
    """
    thresholds = np.arange(0.05, 0.95, 0.005)
    TARs, FARs, TRRs = [], [], []

    for t in thresholds:
        TAR = np.mean(genuine_dists  < t)   # genuins bien acceptés
        FAR = np.mean(unknown_dists  < t)   # inconnus mal acceptés
        TRR = np.mean(unknown_dists  >= t)  # inconnus bien rejetés
        TARs.append(TAR)
        FARs.append(FAR)
        TRRs.append(TRR)

    TARs = np.array(TARs)
    FARs = np.array(FARs)
    TRRs = np.array(TRRs)

    # Seuil EER (Equal Error Rate) : FAR = FRR
    FRRs    = 1 - TARs
    eer_idx = np.argmin(np.abs(FARs - FRRs))
    eer_t   = thresholds[eer_idx]
    eer     = (FARs[eer_idx] + FRRs[eer_idx]) / 2

    # Seuil F1 optimal : maximise TAR tout en gardant FAR bas
    # Score = harmonic mean de TAR et TRR
    scores = []
    for tar, trr, far in zip(TARs, TRRs, FARs):
        if tar + trr > 0:
            scores.append(2 * tar * trr / (tar + trr))
        else:
            scores.append(0)
    scores   = np.array(scores)
    best_idx = np.argmax(scores)
    best_t   = thresholds[best_idx]

    return thresholds, TARs, FARs, TRRs, eer_t, eer, best_t, scores


def plot_threshold_analysis(thresholds, TARs, FARs, TRRs,
                             genuine_dists, unknown_dists,
                             eer_t, eer, best_t, scores,
                             matched_names, output_dir):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0f1117')

    ACCENT = '#00e5ff'
    GREEN  = '#00ff88'
    RED    = '#ff4466'
    ORANGE = '#ffaa00'
    BG     = '#1a1d27'
    TEXT   = '#e0e0e0'

    def style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=ACCENT, fontsize=11, fontweight='bold')
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.grid(True, color='#2a2a3a', linewidth=0.5, linestyle='--')

    # ── 1. TAR / FAR vs seuil ─────────────────────────────
    ax = axes[0]
    ax.plot(thresholds, TARs * 100, color=GREEN,  lw=2, label='TAR (connus acceptés)')
    ax.plot(thresholds, FARs * 100, color=RED,    lw=2, label='FAR (inconnus acceptés)')
    ax.plot(thresholds, TRRs * 100, color=ORANGE, lw=1.5, linestyle='--',
            label='TRR (inconnus rejetés)')
    ax.axvline(eer_t,  color='white',  lw=1.5, linestyle=':',
               label=f'EER={eer*100:.1f}% @ {eer_t:.3f}')
    ax.axvline(best_t, color=ACCENT, lw=2, linestyle='-',
               label=f'Optimal={best_t:.3f}')
    ax.set_xlabel('Seuil (distance cosine)')
    ax.set_ylabel('%')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, facecolor=BG, labelcolor=TEXT, edgecolor='#333')
    style(ax, 'TAR / FAR vs Seuil')

    # ── 2. Distribution distances ──────────────────────────
    ax = axes[1]
    bins = np.linspace(0, 1.0, 60)
    ax.hist(genuine_dists, bins=bins, alpha=0.75, color=GREEN,
            label=f'Connus  (n={len(genuine_dists)})')
    ax.hist(unknown_dists, bins=bins, alpha=0.65, color=RED,
            label=f'Inconnus (n={len(unknown_dists)})')
    ax.axvline(best_t, color=ACCENT, lw=2, linestyle='--',
               label=f'Seuil optimal={best_t:.3f}')
    ax.axvline(eer_t,  color='white', lw=1.5, linestyle=':',
               label=f'EER={eer_t:.3f}')

    # Zone verte = zone d'acceptation
    ax.axvspan(0, best_t, alpha=0.05, color=GREEN)
    # Zone rouge = zone de rejet
    ax.axvspan(best_t, 1.0, alpha=0.05, color=RED)

    ax.set_xlabel('Distance cosine (plus petit = plus similaire)')
    ax.set_ylabel('Nombre')
    ax.legend(fontsize=7, facecolor=BG, labelcolor=TEXT, edgecolor='#333')
    style(ax, 'Distribution Connus vs Inconnus')

    # ── 3. Score F1 vs seuil ──────────────────────────────
    ax = axes[2]
    ax.plot(thresholds, scores * 100, color=ACCENT, lw=2)
    ax.axvline(best_t, color=GREEN, lw=2, linestyle='--',
               label=f'Seuil optimal = {best_t:.3f}')

    # Annoter les métriques au seuil optimal
    best_idx = np.argmax(scores)
    tar_at_best = TARs[best_idx] * 100
    far_at_best = FARs[best_idx] * 100
    trr_at_best = TRRs[best_idx] * 100

    info = (f"Au seuil {best_t:.3f} :\n"
            f"  TAR = {tar_at_best:.1f}%\n"
            f"  FAR = {far_at_best:.1f}%\n"
            f"  TRR = {trr_at_best:.1f}%")
    ax.text(0.97, 0.05, info, transform=ax.transAxes,
            color=TEXT, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='#2a2a3a', edgecolor='#555'))

    ax.set_xlabel('Seuil')
    ax.set_ylabel('Score (%)')
    ax.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, edgecolor='#333')
    style(ax, 'Score optimal (harmonic TAR+TRR)')

    fig.suptitle('Calibration du seuil — Rejet des inconnus',
                 color=TEXT, fontsize=13, fontweight='bold')

    out = os.path.join(output_dir, 'threshold_calibration.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Graphique : {out}")


def main():
    print("=" * 55)
    print("  CALIBRATION SEUIL — Rejet des inconnus")
    print("=" * 55)

    # 1. Charger modèle + base
    model    = load_model(MODEL_PATH)
    with open(DB_PATH, 'rb') as f:
        database = pickle.load(f)
    print(f"[INFO] Base : {len(database)} personnes")

    # 2. Distances genuines (depuis la base existante)
    genuine_dists = collect_known_distances(model, database)

    # 3. Distances inconnus
    result = collect_unknown_distances(model, database, UNKNOWN_DIR)
    if isinstance(result, tuple):
        unknown_dists, matched_names = result
    else:
        print("[ERREUR] Impossible de charger les inconnus")
        return

    if len(unknown_dists) == 0:
        print("[ERREUR] Aucune distance inconnue calculée")
        return

    # 4. Trouver seuil optimal
    print("\n[INFO] Recherche du seuil optimal...")
    thresholds, TARs, FARs, TRRs, eer_t, eer, best_t, scores = \
        find_optimal_threshold(genuine_dists, unknown_dists)

    best_idx    = np.argmax(scores)
    tar_best    = TARs[best_idx] * 100
    far_best    = FARs[best_idx] * 100
    trr_best    = TRRs[best_idx] * 100

    # 5. Afficher résultats
    print("\n" + "=" * 55)
    print("  RÉSULTATS")
    print("=" * 55)
    print(f"  Seuil EER      : {eer_t:.3f}  (EER={eer*100:.2f}%)")
    print(f"  Seuil optimal  : {best_t:.3f}  ← à utiliser")
    print(f"\n  Au seuil {best_t:.3f} :")
    print(f"    TAR (connus acceptés)   : {tar_best:.1f}%")
    print(f"    FAR (inconnus acceptés) : {far_best:.1f}%  ← faux positifs")
    print(f"    TRR (inconnus rejetés)  : {trr_best:.1f}%")

    print(f"\n  Inconnus les plus proches de la base :")
    sorted_matches = sorted(matched_names, key=lambda x: x[1])
    for fname, dist, matched in sorted_matches[:10]:
        danger = " ⚠ ACCEPTÉ" if dist < best_t else "   rejeté"
        print(f"    {fname:30s}  d={dist:.3f} → {matched:20s}{danger}")

    print(f"\n  → Mettre dans recognize.py et recognize_bbb.py :")
    print(f"    THRESHOLD = {best_t:.3f}")
    print("=" * 55)

    # 6. Rapport visuel
    print("\n[INFO] Génération graphique...")
    plot_threshold_analysis(
        thresholds, TARs, FARs, TRRs,
        genuine_dists, unknown_dists,
        eer_t, eer, best_t, scores,
        matched_names, OUTPUT_DIR
    )

    # 7. Sauvegarder le seuil dans un fichier config
    config_path = "threshold_config.txt"
    with open(config_path, 'w') as f:
        f.write(f"THRESHOLD = {best_t:.3f}\n")
        f.write(f"EER_THRESHOLD = {eer_t:.3f}\n")
        f.write(f"TAR_at_optimal = {tar_best:.1f}%\n")
        f.write(f"FAR_at_optimal = {far_best:.1f}%\n")
        f.write(f"TRR_at_optimal = {trr_best:.1f}%\n")
    print(f"[OK] Config sauvegardée : {config_path}")


if __name__ == "__main__":
    main()