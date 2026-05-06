"""
enroll.py — Construit la base d'embeddings depuis known_users/
Détection  : MediaPipe FaceDetection
Alignement : MediaPipe FaceMesh → 5 landmarks → transformation affine ArcFace
Usage      : python enroll.py
"""

import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from face_align import align_face, detect_and_align, imread_unicode

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH = "ghostfacenet.h5"
KNOWN_DIR  = "known_users"
DB_PATH    = "face_db.pkl"
INPUT_SIZE = (112, 112)
# ──────────────────────────────────────────────────────────


def load_model(path):
    print(f"[INFO] Chargement du modèle : {path}")
    model = tf.keras.models.load_model(path)
    print("[INFO] Modèle chargé.")
    return model


def preprocess(face_img):
    """face_img doit être déjà 112x112 (sorti de align_face)"""
    face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    face = (face - 127.5) / 128.0
    return np.expand_dims(face, axis=0)


def get_embedding(model, aligned_face):
    emb = model.predict(preprocess(aligned_face), verbose=0)[0]
    return emb / np.linalg.norm(emb)


def enroll_from_folder(model, folder, db_path):
    supported = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    database  = {}

    persons = sorted([
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d)) and not d.startswith('.')
    ])

    if not persons:
        print(f"[ERREUR] Aucun sous-dossier dans '{folder}/'")
        return

    print(f"\n[INFO] {len(persons)} personne(s) : {', '.join(persons)}")
    print("─" * 55)

    stats = {'aligned': 0, 'fallback': 0, 'failed': 0}

    for person_dir in persons:
        name        = person_dir.capitalize()
        person_path = os.path.join(folder, person_dir)
        images      = sorted([
            f for f in os.listdir(person_path)
            if f.lower().endswith(supported)
        ])

        if not images:
            print(f"  [WARN] {person_dir}/ vide, ignoré")
            continue

        print(f"\n  [{name}] — {len(images)} image(s)")
        database[name] = []

        for fname in images:
            img = imread_unicode(os.path.join(person_path, fname))
            if img is None:
                print(f"    ✗ {fname}  (lecture impossible)")
                stats['failed'] += 1
                continue

            # Alignement ArcFace via FaceMesh
            aligned, lm_ok = align_face(img)
            tag = "✓ aligned" if lm_ok else "~ fallback"

            if lm_ok:
                stats['aligned'] += 1
            else:
                stats['fallback'] += 1

            # Embedding + augmentation miroir
            emb = get_embedding(model, aligned)
            database[name].append(emb)

            flipped = cv2.flip(aligned, 1)
            database[name].append(get_embedding(model, flipped))

            print(f"    {tag}  {fname}  → {len(database[name])} emb")

    if not database:
        print("\n[ERREUR] Aucun embedding généré.")
        return

    with open(db_path, 'wb') as f:
        pickle.dump(database, f)

    print("\n" + "─" * 55)
    print(f"[OK] Base sauvegardée : '{db_path}'")
    print(f"     {len(database)} personne(s) enrôlée(s)")
    print(f"     Alignés FaceMesh : {stats['aligned']}")
    print(f"     Fallback (crop)  : {stats['fallback']}")
    print(f"     Échoués          : {stats['failed']}")
    for name, embs in database.items():
        print(f"       • {name} — {len(embs)} embedding(s)")


if __name__ == "__main__":
    if not os.path.exists(KNOWN_DIR):
        os.makedirs(KNOWN_DIR)
        print(f"[INFO] Dossier '{KNOWN_DIR}/' créé.")
    else:
        model = load_model(MODEL_PATH)
        enroll_from_folder(model, KNOWN_DIR, DB_PATH)