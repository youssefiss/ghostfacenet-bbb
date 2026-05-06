"""
check_mesh.py — PC uniquement
Identifie toutes les images de known_users où FaceMesh ne détecte pas de visage.
Affiche un rapport et propose de supprimer ou déplacer les mauvaises images.

Usage :
  python check_mesh.py           # rapport seul
  python check_mesh.py --delete  # supprime automatiquement les images no-mesh
  python check_mesh.py --move    # déplace vers known_users/_rejected/
"""

import os
import cv2
import sys
import shutil
from face_align import align_face, imread_unicode

KNOWN_DIR  = "known_users"
REJECT_DIR = os.path.join(KNOWN_DIR, "_rejected")
SUPPORTED  = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

MODE = "report"
if "--delete" in sys.argv:
    MODE = "delete"
elif "--move" in sys.argv:
    MODE = "move"
    os.makedirs(REJECT_DIR, exist_ok=True)

print("=" * 60)
print("  CHECK FACEMESH — Détection visage dans known_users")
print(f"  Mode : {MODE.upper()}")
print("=" * 60)

persons = sorted([
    d for d in os.listdir(KNOWN_DIR)
    if os.path.isdir(os.path.join(KNOWN_DIR, d))
    and not d.startswith('.')
    and d != "_rejected"
])

total_imgs    = 0
total_ok      = 0
total_no_mesh = 0
no_mesh_files = []   # liste (personne, fichier, chemin)

for person_dir in persons:
    name = person_dir.capitalize()
    path = os.path.join(KNOWN_DIR, person_dir)
    imgs = sorted([f for f in os.listdir(path)
                   if f.lower().endswith(SUPPORTED)])

    n_ok = 0
    n_fail = 0
    fails = []

    for fname in imgs:
        fpath = os.path.join(path, fname)
        img = imread_unicode(fpath)
        if img is None:
            print(f"  [WARN] Impossible de lire : {fname}")
            continue

        _, ok = align_face(img)
        total_imgs += 1

        if ok:
            n_ok += 1
            total_ok += 1
            print(".", end="", flush=True)
        else:
            n_fail += 1
            total_no_mesh += 1
            fails.append((fname, fpath))
            no_mesh_files.append((name, fname, fpath))
            print("✗", end="", flush=True)

    print()

    pct_ok = 100 * n_ok / len(imgs) if imgs else 0
    status = "✓" if n_fail == 0 else "!"
    print(f"  [{status}] {name:<22} "
          f"{n_ok}/{len(imgs)} détectés  ({pct_ok:.0f}%)")

    if fails:
        for fname, fpath in fails:
            print(f"      ✗ {fname}")

            if MODE == "delete":
                os.remove(fpath)
                print(f"        → supprimé")
            elif MODE == "move":
                dest_dir = os.path.join(REJECT_DIR, person_dir)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(fpath, os.path.join(dest_dir, fname))
                print(f"        → déplacé vers _rejected/{person_dir}/")

    print()

# ── Rapport final ─────────────────────────────────────────
print("=" * 60)
print("  RAPPORT FINAL")
print("=" * 60)
print(f"  Total images      : {total_imgs}")
print(f"  FaceMesh OK       : {total_ok}  ({100*total_ok/total_imgs:.1f}%)")
print(f"  FaceMesh ECHEC    : {total_no_mesh}  ({100*total_no_mesh/total_imgs:.1f}%)")
print()

if total_no_mesh == 0:
    print("  Toutes les images sont détectées par FaceMesh !")
else:
    print(f"  {total_no_mesh} image(s) posent problème :")
    for name, fname, _ in no_mesh_files:
        print(f"    - {name}/{fname}")

    print()
    if MODE == "report":
        print("  Pour supprimer ces images :")
        print("    python check_mesh.py --delete")
        print()
        print("  Pour les déplacer dans known_users/_rejected/ :")
        print("    python check_mesh.py --move")
        print()
        print("  Ensuite recalculer le pkl :")
        print("    python precalculate_embeddings.py")
        print("    scp known_embeddings.pkl debian@BBB_IP:~/ghost2/")
    elif MODE == "delete":
        print(f"  {total_no_mesh} image(s) supprimée(s).")
        print()
        print("  Recalculer le pkl :")
        print("    python precalculate_embeddings.py")
        print("    scp known_embeddings.pkl debian@BBB_IP:~/ghost2/")
    elif MODE == "move":
        print(f"  {total_no_mesh} image(s) déplacée(s) dans {REJECT_DIR}")
        print()
        print("  Recalculer le pkl :")
        print("    python precalculate_embeddings.py")
        print("    scp known_embeddings.pkl debian@BBB_IP:~/ghost2/")

print("=" * 60)

# ── Résumé par personne ───────────────────────────────────
print()
print("  RÉSUMÉ PAR PERSONNE :")
print("  " + "-" * 50)

for person_dir in persons:
    name = person_dir.capitalize()
    path = os.path.join(KNOWN_DIR, person_dir)
    imgs = [f for f in os.listdir(path) if f.lower().endswith(SUPPORTED)]
    n_fails = sum(1 for n, f, _ in no_mesh_files if n == name)
    n_remaining = len(imgs) - (n_fails if MODE == "report" else 0)
    pct = 100 * (len(imgs) - n_fails) / len(imgs) if imgs else 0
    bar = "█" * int(pct / 5)
    warn = "⚠" if pct < 80 else " "
    print(f"  {warn} {name:<22} "
          f"{len(imgs)-n_fails}/{len(imgs)} OK  "
          f"({pct:.0f}%)  {bar}")

print("=" * 60)