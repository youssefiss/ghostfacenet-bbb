"""
test_image.py — Tester la reconnaissance sur une image fixe (pas webcam)
Usage : python test_image.py chemin/vers/image.jpg
        python test_image.py  (utilise la webcam pour capturer un test)
"""

import cv2
import numpy as np
import pickle
import tensorflow as tf
from scipy.spatial.distance import cosine
import sys
import os

MODEL_PATH = "ghostfacenet_w1.3_s2.h5"
DB_PATH    = "face_db.pkl"
THRESHOLD  = 0.40
INPUT_SIZE = (112, 112)


def load_model(path):
    return tf.keras.models.load_model(path)


def load_database(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def preprocess(face_img):
    face = cv2.resize(face_img, INPUT_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
    face = (face - 127.5) / 128.0
    return np.expand_dims(face, axis=0)


def get_embedding(model, face_img):
    emb = model.predict(preprocess(face_img), verbose=0)[0]
    return emb / np.linalg.norm(emb)


def recognize(embedding, database, threshold):
    best_name, best_dist = "Inconnu", float('inf')
    for name, embs in database.items():
        dists = sorted([cosine(embedding, e) for e in embs])[:3]
        score = float(np.mean(dists))
        if score < best_dist:
            best_dist, best_name = score, name
    if best_dist < threshold:
        return best_name, round((1 - best_dist) * 100, 1), best_dist
    return "Inconnu", 0.0, best_dist


def test_on_image(image_path):
    model = load_model(MODEL_PATH)
    db    = load_database(DB_PATH)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERREUR] Impossible de lire : {image_path}")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )

    if len(faces) == 0:
        print("[WARN] Aucun visage détecté. Test sur image entière.")
        h, w = img.shape[:2]
        s = min(h, w)
        roi = img[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
        faces_rois = [roi]
        bboxes = [(0, 0, w, h)]
    else:
        faces_rois, bboxes = [], []
        for (x, y, w, h) in faces:
            m = int(0.12 * w)
            x1, y1 = max(0, x-m), max(0, y-m)
            x2, y2 = min(img.shape[1], x+w+m), min(img.shape[0], y+h+m)
            faces_rois.append(img[y1:y2, x1:x2])
            bboxes.append((x, y, w, h))

    print(f"\n{'─'*50}")
    print(f"Image testée : {image_path}")
    print(f"Visages détectés : {len(faces_rois)}")
    print(f"{'─'*50}")

    for i, (roi, bbox) in enumerate(zip(faces_rois, bboxes)):
        emb = get_embedding(model, roi)
        name, conf, dist = recognize(emb, db, THRESHOLD)

        print(f"\n  Visage #{i+1}")
        print(f"    Résultat   : {name}")
        print(f"    Confiance  : {conf:.1f}%")
        print(f"    Distance   : {dist:.4f}  (seuil={THRESHOLD})")
        print(f"    Verdict    : {'✓ RECONNU' if name != 'Inconnu' else '✗ INCONNU'}")

        # Dessiner sur l'image
        x, y, w, h = bbox
        color = (50, 220, 100) if name != "Inconnu" else (60, 60, 220)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        label = f"{name}  {conf:.1f}%" if name != "Inconnu" else "Inconnu"
        cv2.putText(img, label, (x, y-8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA)

    print(f"\n{'─'*50}")

    # Afficher l'image annotée
    cv2.imshow("Résultat", img)
    out_path = "result_" + os.path.basename(image_path)
    cv2.imwrite(out_path, img)
    print(f"[INFO] Image résultat sauvegardée : {out_path}")
    print("[INFO] Appuyez sur une touche pour fermer.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def capture_and_test():
    """Capture une photo depuis la webcam et teste"""
    print("[INFO] Appuyez sur ESPACE pour capturer, Q pour annuler.")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "ESPACE=Capturer  Q=Quitter", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 200, 220), 2)
        cv2.imshow("Capture test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            tmp = "tmp_test_capture.jpg"
            cv2.imwrite(tmp, frame)
            cap.release()
            cv2.destroyAllWindows()
            test_on_image(tmp)
            return
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_on_image(sys.argv[1])
    else:
        capture_and_test()