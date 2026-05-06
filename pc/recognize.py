"""
recognize.py — Reconnaissance avec double seuil
THRESHOLD_HIGH : acceptation haute confiance  (vert)
THRESHOLD_LOW  : acceptation incertaine       (orange) → demande confirmation
Au-dessus      : rejeté comme Inconnu         (rouge)
"""

import cv2
import numpy as np
import pickle
import tensorflow as tf
import mediapipe as mp
from scipy.spatial.distance import cosine
from face_align import detect_and_align
import time
import os

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH       = "ghostfacenet.h5"
DB_PATH          = "face_db.pkl"
THRESHOLD_HIGH   = 0.50    # haute confiance  → vert
THRESHOLD_LOW    = 0.605   # confiance faible → orange (à valider)
INPUT_SIZE       = (112, 112)
CAMERA_ID        = 0
DETECT_SKIP      = 2
# ──────────────────────────────────────────────────────────

GREEN  = (50,  220, 100)   # reconnu haute confiance
ORANGE = (30,  165, 255)   # reconnu faible confiance
RED    = (60,   60, 220)   # inconnu
YELLOW = (30,  200, 220)   # HUD
DARK   = (20,   20,  20)


def load_model(path):
    print(f"[INFO] Chargement modèle : {path} ...")
    model = tf.keras.models.load_model(path)
    print("[INFO] Modèle prêt.")
    return model


def load_database(path):
    if not os.path.exists(path):
        print(f"[ERREUR] Base introuvable : {path}")
        return {}
    with open(path, 'rb') as f:
        db = pickle.load(f)
    total = sum(len(v) for v in db.values())
    print(f"[INFO] Base : {len(db)} personne(s), {total} embeddings")
    print(f"       {', '.join(db.keys())}")
    return db


def preprocess(aligned_face):
    face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims((face - 127.5) / 128.0, axis=0)


def get_embedding(model, aligned_face):
    emb = model.predict(preprocess(aligned_face), verbose=0)[0]
    return emb / np.linalg.norm(emb)


def recognize(embedding, database, threshold_high, threshold_low):
    """
    Retourne (nom, confiance, distance, niveau)
    niveau : 'high'    → haute confiance  (dist < threshold_high)
             'low'     → faible confiance (dist < threshold_low)
             'unknown' → inconnu          (dist >= threshold_low)
    """
    best_name = "Inconnu"
    best_dist = float('inf')

    for name, stored_embs in database.items():
        dists = [cosine(embedding, e) for e in stored_embs]
        score = float(np.mean(sorted(dists)[:3]))
        if score < best_dist:
            best_dist = score
            best_name = name

    if best_dist < threshold_high:
        conf  = round((1.0 - best_dist) * 100, 1)
        return best_name, conf, best_dist, 'high'
    elif best_dist < threshold_low:
        conf  = round((1.0 - best_dist) * 100, 1)
        return best_name, conf, best_dist, 'low'
    else:
        return "Inconnu", 0.0, best_dist, 'unknown'


def draw_result(frame, bbox, name, confidence, dist, level):
    x, y, w, h = bbox

    if level == 'high':
        color = GREEN
        label = f"{name}  {confidence:.1f}%"
    elif level == 'low':
        color = ORANGE
        label = f"{name}?  {confidence:.1f}%"   # ? = incertain
    else:
        color = RED
        label = "Inconnu"

    # Rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Label background
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    cv2.rectangle(frame, (x, y - lh - 14), (x + lw + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, DARK, 1, cv2.LINE_AA)

    # Distance (debug)
    cv2.putText(frame, f"d={dist:.3f}", (x, y + h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def draw_hud(frame, fps, n_faces, threshold_high, threshold_low):
    lines = [
        f"FPS:{fps:.1f}  Visages:{n_faces}",
        f"Seuil haut:{threshold_high}  bas:{threshold_low}",
        "Q=Quitter  S=Screenshot  R=Reload",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, YELLOW, 1, cv2.LINE_AA)

    # Légende niveaux
    for i, (color, txt) in enumerate([
        (GREEN,  "Haute confiance"),
        (ORANGE, "Faible confiance"),
        (RED,    "Inconnu"),
    ]):
        fx = frame.shape[1] - 170
        fy = 18 + i * 20
        cv2.rectangle(frame, (fx, fy - 10), (fx + 14, fy + 2), color, -1)
        cv2.putText(frame, txt, (fx + 18, fy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)


def main():
    model    = load_model(MODEL_PATH)
    database = load_database(DB_PATH)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[ERREUR] Caméra {CAMERA_ID} inaccessible")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n[INFO] Seuil haute confiance : {THRESHOLD_HIGH}")
    print(f"[INFO] Seuil faible confiance : {THRESHOLD_LOW}")
    print(f"[INFO] Q pour quitter\n")

    frame_count  = 0
    last_results = []
    fps = 0.0
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t_now  = time.time()
        fps    = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        if frame_count % DETECT_SKIP == 0:
            faces = detect_and_align(frame, min_confidence=0.6)
            last_results = []

            for face in faces:
                aligned = face['aligned']
                bbox    = face['bbox']

                if database:
                    emb  = get_embedding(model, aligned)
                    name, conf, dist, level = recognize(
                        emb, database, THRESHOLD_HIGH, THRESHOLD_LOW
                    )
                else:
                    name, conf, dist, level = "Base vide", 0.0, 1.0, 'unknown'

                last_results.append((bbox, name, conf, dist, level))

        for (bbox, name, conf, dist, level) in last_results:
            draw_result(frame, bbox, name, conf, dist, level)

        draw_hud(frame, fps, len(last_results), THRESHOLD_HIGH, THRESHOLD_LOW)

        cv2.imshow("GhostFaceNet — Double Seuil", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot : {fname}")
        elif key == ord('r'):
            database = load_database(DB_PATH)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Terminé.")


if __name__ == "__main__":
    main()