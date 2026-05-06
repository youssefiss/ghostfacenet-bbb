"""
face_align_lite.py — Alignement facial LÉGER pour BeagleBone Black
Pas de MediaPipe FaceMesh (trop lourd pour BBB ARM Cortex-A8)
Stratégie :
  1. Détection  : Haar Cascade multi-passes (frontal + profil + paramètres relâchés)
  2. Landmarks  : Haar eye cascade + estimation géométrique calibrée
  3. Alignement : transformation affine ArcFace 112x112
  4. Fallback   : crop centré avec marge si tout échoue

Améliorations v2 :
  - Multi-passes Haar (scaleFactor 1.05 → 1.1 → détection permissive)
  - Cascade profil ajoutée
  - equalizeHist + CLAHE pour mauvais éclairages
  - minSize réduit à 40px pour petits visages
  - Fallback crop intelligent (centre de masse + ROI élargie)

Dépendances BBB : opencv-python-headless, numpy
"""

import cv2
import numpy as np
import os
import urllib.request

# ── Positions cibles ArcFace 112x112 ──────────────────────
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# ── Charger les cascades ──────────────────────────────────
def _load_cascade(fname):
    candidates = [
        fname,
        '/usr/share/opencv4/haarcascades/' + fname,
        '/usr/share/opencv/haarcascades/'  + fname,
        '/usr/local/share/opencv4/haarcascades/' + fname,
    ]
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        candidates.insert(0, cv2.data.haarcascades + fname)
    for p in candidates:
        if p and os.path.exists(p):
            return cv2.CascadeClassifier(p)
    # Télécharger si absent
    url = ("https://raw.githubusercontent.com/opencv/opencv/"
           "master/data/haarcascades/" + fname)
    try:
        urllib.request.urlretrieve(url, fname)
        return cv2.CascadeClassifier(fname)
    except Exception:
        return cv2.CascadeClassifier()

_face_front   = _load_cascade('haarcascade_frontalface_default.xml')
_face_alt     = _load_cascade('haarcascade_frontalface_alt2.xml')
_face_profile = _load_cascade('haarcascade_profileface.xml')
_eye_cascade  = _load_cascade('haarcascade_eye.xml')


# ── Prétraitement image ───────────────────────────────────
def _preprocess_gray(image_bgr):
    """
    Convertit en gris + amélioration contraste CLAHE.
    Meilleure détection sous mauvais éclairage.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE — meilleur qu'equalizeHist pour les images inégalement éclairées
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ── Détection multi-passes ────────────────────────────────
def _detect_face(gray, image_bgr):
    """
    Essaie plusieurs cascades et paramètres pour maximiser la détection.
    Retourne (x, y, w, h) du plus grand visage trouvé, ou None.
    """
    h, w = gray.shape[:2]

    # Passe 1 — frontal strict (minNeighbors=5, scaleFactor=1.05)
    faces = _face_front.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) > 0:
        return max(faces, key=lambda f: f[2] * f[3])

    # Passe 2 — frontal alt2 (meilleur sur angles ±30°)
    faces = _face_alt.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40)
    )
    if len(faces) > 0:
        return max(faces, key=lambda f: f[2] * f[3])

    # Passe 3 — paramètres plus permissifs
    faces = _face_front.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )
    if len(faces) > 0:
        return max(faces, key=lambda f: f[2] * f[3])

    # Passe 4 — profil (visage de côté)
    faces = _face_profile.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40)
    )
    if len(faces) > 0:
        return max(faces, key=lambda f: f[2] * f[3])

    # Passe 5 — profil retourné (côté gauche)
    gray_flip = cv2.flip(gray, 1)
    faces = _face_profile.detectMultiScale(
        gray_flip, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40)
    )
    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        # Reconvertir coordonnées après flip
        fx = w - fx - fw
        return (fx, fy, fw, fh)

    # Passe 6 — très permissif (dernier recours)
    faces = _face_front.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=2, minSize=(20, 20)
    )
    if len(faces) > 0:
        return max(faces, key=lambda f: f[2] * f[3])

    return None


# ── Estimation landmarks ──────────────────────────────────
def _estimate_landmarks(image_gray, fx, fy, fw, fh):
    """
    Estime 5 landmarks depuis bbox + raffine avec eye cascade.
    """
    # Estimation géométrique calibrée (proportions LFW)
    left_eye    = [fx + fw * 0.30, fy + fh * 0.37]
    right_eye   = [fx + fw * 0.70, fy + fh * 0.37]
    nose        = [fx + fw * 0.50, fy + fh * 0.55]
    mouth_left  = [fx + fw * 0.35, fy + fh * 0.75]
    mouth_right = [fx + fw * 0.65, fy + fh * 0.75]
    src_pts = np.array(
        [left_eye, right_eye, nose, mouth_left, mouth_right],
        dtype=np.float32
    )

    # Raffiner avec eye cascade
    roi_gray = image_gray[fy:fy+fh, fx:fx+fw]
    eyes = _eye_cascade.detectMultiScale(
        roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(12, 12)
    )
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        e1, e2 = eyes[0], eyes[1]
        src_pts[0] = [fx + e1[0] + e1[2]//2, fy + e1[1] + e1[3]//2]
        src_pts[1] = [fx + e2[0] + e2[2]//2, fy + e2[1] + e2[3]//2]

    return src_pts


# ── Fallback intelligent ───────────────────────────────────
def _smart_crop(image_bgr, output_size):
    """
    Crop intelligent quand aucun visage n'est détecté.
    Prend le centre de l'image avec ratio 1:1.
    """
    h, w = image_bgr.shape[:2]
    side = min(h, w)
    # Légèrement décalé vers le haut (les visages sont souvent en haut)
    x0 = (w - side) // 2
    y0 = max(0, (h - side) // 2 - int(side * 0.05))
    y0 = min(y0, h - side)
    crop = image_bgr[y0:y0+side, x0:x0+side]
    return cv2.resize(crop, output_size)


# ══════════════════════════════════════════════════════════
# API PUBLIQUE
# ══════════════════════════════════════════════════════════

def align_face(image_bgr, output_size=(112, 112)):
    """
    Détecte et aligne le visage principal dans image_bgr.
    Retourne (aligned_112x112_bgr, success_bool).

    success=True  → visage détecté et aligné correctement
    success=False → fallback utilisé (crop centré)
    """
    h, w = image_bgr.shape[:2]

    # Prétraitement
    gray = _preprocess_gray(image_bgr)

    # Détection multi-passes
    result = _detect_face(gray, image_bgr)

    if result is None:
        # Aucun visage détecté — fallback crop centré
        return _smart_crop(image_bgr, output_size), False

    fx, fy, fw, fh = result

    # Estimer landmarks
    src_pts = _estimate_landmarks(gray, fx, fy, fw, fh)

    # Transformation affine ArcFace
    scale = output_size[0] / 112.0
    dst   = ARCFACE_DST * scale
    M, _  = cv2.estimateAffinePartial2D(src_pts, dst, method=cv2.LMEDS)

    if M is None:
        # Fallback crop avec marge
        m  = int(0.12 * fw)
        x1 = max(0, fx - m);  y1 = max(0, fy - m)
        x2 = min(w, fx+fw+m); y2 = min(h, fy+fh+m)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return _smart_crop(image_bgr, output_size), False
        return cv2.resize(crop, output_size), False

    aligned = cv2.warpAffine(
        image_bgr, M, output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return aligned, True


def detect_and_align_all(frame_bgr, min_size=40):
    """
    Détecte TOUS les visages dans un frame caméra.
    Retourne liste de {'bbox': (x,y,w,h), 'aligned': img, 'lm_ok': bool}.
    Optimisé pour CPU lent (BBB).
    """
    h, w = frame_bgr.shape[:2]
    gray = _preprocess_gray(frame_bgr)

    # Détection principale
    faces_raw = _face_front.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Si rien → essayer alt2
    if len(faces_raw) == 0:
        faces_raw = _face_alt.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_size, min_size),
        )

    results = []
    for (fx, fy, fw, fh) in (faces_raw if len(faces_raw) > 0 else []):
        m  = int(0.12 * fw)
        x1 = max(0, fx - m);  y1 = max(0, fy - m)
        x2 = min(w, fx+fw+m); y2 = min(h, fy+fh+m)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        aligned, ok = align_face(roi)
        results.append({
            'bbox':    (fx, fy, fw, fh),
            'aligned': aligned,
            'lm_ok':   ok
        })

    return results
