"""
face_align.py — Alignement facial ArcFace standard via MediaPipe
Fix Windows : utilise np.frombuffer pour lire les images (évite le bug
              cv2.imread avec chemins contenant parenthèses, accents, espaces)
"""

import cv2
import numpy as np
import mediapipe as mp

# ── Positions cibles ArcFace 112x112 ──────────────────────
ARCFACE_DST = np.array([
    [38.2946, 51.6963],   # œil gauche
    [73.5318, 51.5014],   # œil droit
    [56.0252, 71.7366],   # nez
    [41.5493, 92.3655],   # bouche gauche
    [70.7299, 92.2041],   # bouche droite
], dtype=np.float32)

# ── Init MediaPipe ─────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

mp_face_det = mp.solutions.face_detection
face_det    = mp_face_det.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

MESH_LANDMARKS = {
    'left_eye':   [33,  133],
    'right_eye':  [362, 263],
    'nose':       [1],
    'mouth_left': [61],
    'mouth_right':[291],
}


def imread_unicode(path):
    """
    Lecture robuste sur Windows : gère les chemins avec
    parenthèses, espaces, accents, caractères spéciaux.
    """
    try:
        with open(path, 'rb') as f:
            buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"    [ERREUR lecture] {path} : {e}")
        return None


def _get_mesh_landmarks(image_rgb):
    h, w = image_rgb.shape[:2]
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark

    def mean_pt(indices):
        return [np.mean([lm[i].x * w for i in indices]),
                np.mean([lm[i].y * h for i in indices])]

    return np.array([
        mean_pt(MESH_LANDMARKS['left_eye']),
        mean_pt(MESH_LANDMARKS['right_eye']),
        mean_pt(MESH_LANDMARKS['nose']),
        mean_pt(MESH_LANDMARKS['mouth_left']),
        mean_pt(MESH_LANDMARKS['mouth_right']),
    ], dtype=np.float32)


def _get_detection_landmarks(image_rgb):
    h, w = image_rgb.shape[:2]
    results = face_det.process(image_rgb)
    if not results.detections:
        return None

    det = max(results.detections, key=lambda d: d.score[0])
    kp  = det.location_data.relative_keypoints
    if len(kp) < 4:
        return None

    left_eye  = [kp[0].x * w, kp[0].y * h]
    right_eye = [kp[1].x * w, kp[1].y * h]
    nose      = [kp[2].x * w, kp[2].y * h]
    mouth     = [kp[3].x * w, kp[3].y * h]

    eye_dist     = np.linalg.norm(np.array(right_eye) - np.array(left_eye))
    offset       = eye_dist * 0.18
    mouth_left   = [mouth[0] - offset, mouth[1]]
    mouth_right  = [mouth[0] + offset, mouth[1]]

    return np.array([left_eye, right_eye, nose, mouth_left, mouth_right],
                    dtype=np.float32)


def align_face(image_bgr, output_size=(112, 112)):
    """
    Aligne le visage selon le standard ArcFace.
    Retourne (aligned_bgr 112x112, landmark_ok).
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    src_pts = _get_mesh_landmarks(image_rgb)
    if src_pts is None:
        src_pts = _get_detection_landmarks(image_rgb)

    if src_pts is None:
        side = min(h, w)
        x0   = (w - side) // 2
        y0   = (h - side) // 2
        crop = image_bgr[y0:y0+side, x0:x0+side]
        return cv2.resize(crop, output_size), False

    scale = output_size[0] / 112.0
    dst   = ARCFACE_DST * scale

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst, method=cv2.LMEDS)

    if M is None:
        side = min(h, w)
        x0   = (w - side) // 2
        y0   = (h - side) // 2
        crop = image_bgr[y0:y0+side, x0:x0+side]
        return cv2.resize(crop, output_size), False

    aligned = cv2.warpAffine(image_bgr, M, output_size,
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned, True


def detect_and_align(frame_bgr, min_confidence=0.6):
    """
    Détecte tous les visages dans un frame et retourne
    liste de {bbox, aligned, lm_ok, score}.
    """
    h, w  = frame_bgr.shape[:2]
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_det.process(rgb)

    faces = []
    if not results.detections:
        return faces

    for det in results.detections:
        if det.score[0] < min_confidence:
            continue

        bb     = det.location_data.relative_bounding_box
        margin = 0.15
        x1 = int(max(0, (bb.xmin - margin)              * w))
        y1 = int(max(0, (bb.ymin - margin)              * h))
        x2 = int(min(w, (bb.xmin + bb.width  + margin)  * w))
        y2 = int(min(h, (bb.ymin + bb.height + margin)  * h))

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        aligned, lm_ok = align_face(roi)
        faces.append({
            'bbox':    (x1, y1, x2-x1, y2-y1),
            'aligned': aligned,
            'lm_ok':   lm_ok,
            'score':   float(det.score[0])
        })

    return faces


# ── Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python face_align.py image.jpg")
        sys.exit(1)

    path = sys.argv[1]
    img  = imread_unicode(path)      # ← lecture robuste Windows

    if img is None:
        print(f"Impossible de lire : {path}")
        sys.exit(1)

    aligned, ok = align_face(img)
    status = "FaceMesh ✓" if ok else "Fallback (crop centré)"
    print(f"Alignement : {status}")
    print(f"Taille sortie : {aligned.shape}")

    cv2.imshow("Original",   cv2.resize(img,     (336, 336)))
    cv2.imshow("Aligne 112", cv2.resize(aligned, (336, 336)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out = "aligned_test.jpg"
    cv2.imwrite(out, aligned)
    print(f"Sauvegardé : {out}")