"""
main.py — BeagleBone Black
Reconnaissance faciale GhostFaceNet INT8 + Flask Dashboard
Fonctionnalités : Login admin, stream MJPEG, snapshots, historique, stats

Changement vs version précédente :
  Détection visage : FaceMesh (face_align_lite.py) au lieu de Haar Cascade
  → cohérent avec known_embeddings.pkl calculé par precalculate_embeddings.py

Dépendances :
  pip3 install tflite-runtime flask mediapipe --break-system-packages
"""

import cv2
import numpy as np
from flask import Flask, Response, request, redirect, url_for, session, jsonify
import threading
import time
import pickle
import logging
import os
import signal
import sys
from datetime import datetime
from functools import wraps

# ── PIR sensor via gpiod ──────────────────────────────────
PIR_GPIO     = 47
PIR_CHIP     = "gpiochip0"
PIR_LINE     = 15
WAKE_TIMEOUT = 10

try:
    import gpiod
    GPIOD_AVAILABLE = True
except ImportError:
    GPIOD_AVAILABLE = False

_pir_chip = None
_pir_line = None

def gpio_setup():
    global _pir_chip, _pir_line
    if not GPIOD_AVAILABLE:
        return False
    try:
        _pir_chip = gpiod.Chip(PIR_CHIP)
        _pir_line = _pir_chip.get_line(PIR_LINE)
        _pir_line.request(consumer="pir_sensor", type=gpiod.LINE_REQ_DIR_IN)
        return True
    except Exception:
        _pir_chip = None
        _pir_line = None
        return False

def gpio_read():
    global _pir_line
    if _pir_line is None:
        return 0
    try:
        return _pir_line.get_value()
    except Exception:
        return 0

def gpio_cleanup():
    global _pir_line, _pir_chip
    try:
        if _pir_line:
            _pir_line.release()
        if _pir_chip:
            _pir_chip.close()
    except Exception:
        pass


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

def _init_pir():
    if not GPIOD_AVAILABLE:
        log.warning("gpiod non disponible — PIR désactivé")
        return False
    ok = gpio_setup()
    if ok:
        log.info(f"PIR prêt : gpiod {PIR_CHIP} line {PIR_LINE} (P8_15)")
    else:
        log.warning(f"GPIO {PIR_LINE} setup impossible — PIR désactivé")
    return ok

# ─── CONFIG ───────────────────────────────────────────────
TFLITE_MODEL   = "ghostfacenet_int8.tflite"
EMBEDDINGS_PKL = "known_embeddings.pkl"
THRESHOLD_HIGH = 0.60
THRESHOLD_LOW  = 0.75
FRAME_W        = 200
FRAME_H        = 180
DETECT_EVERY   = 8
JPEG_QUALITY   = 50
FLASK_PORT     = 5000
SNAPSHOT_DIR   = "snapshots"
ADMIN_USER     = "admin"
ADMIN_PASS     = "bbb2024"
SECRET_KEY     = "ghostfacenet-bbb-secret"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = SECRET_KEY

PIR_ENABLED = _init_pir()

stats = {
    'total_detections': 0,
    'recognized':       0,
    'unknown':          0,
    'snapshots':        0,
    'pir_triggers':     0,
    'start_time':       time.time(),
    'history':          [],
}
stats_lock = threading.Lock()

# ══════════════════════════════════════════════════════════
# CHARGEMENT MODÈLE
# ══════════════════════════════════════════════════════════

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    log.info("tflite_runtime chargé.")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    log.info("tensorflow.lite chargé.")

log.info("Chargement TFLite INT8...")
interpreter = Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale,  input_zero_point  = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']
log.info("Modèle TFLite prêt.")

log.info("Chargement embeddings...")
with open(EMBEDDINGS_PKL, "rb") as f:
    data = pickle.load(f)
known_embeddings = data['embeddings']
known_names      = data['names']
log.info(f"{len(known_names)} personne(s) : {', '.join(known_names)}")

# ══════════════════════════════════════════════════════════
# FACEMESH — remplace Haar Cascade
# ══════════════════════════════════════════════════════════

from face_align_lite import align_face

log.info("FaceMesh chargé via face_align_lite.py")

def detect_and_align(frame):
    """
    Détecte et aligne le visage avec FaceMesh.
    Retourne liste de (x1,y1,x2,y2,aligned_roi) ou [] si aucun visage.
    align_face() retourne (aligned_img, success_bool).
    """
    aligned, ok = align_face(frame)
    if not ok:
        return []

    # Boîte englobante approximative pour l'affichage
    h, w = frame.shape[:2]
    margin = int(min(h, w) * 0.1)
    x1 = margin
    y1 = margin
    x2 = w - margin
    y2 = h - margin

    return [(x1, y1, x2, y2, aligned)]

# ══════════════════════════════════════════════════════════
# INFÉRENCE
# ══════════════════════════════════════════════════════════

def get_embedding(roi):
    img = cv2.resize(roi, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img_int8 = (img / input_scale + input_zero_point).astype(np.int8)
    img_int8 = np.expand_dims(img_int8, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_int8)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    emb = (output.astype(np.float32) - output_zero_point) * output_scale
    emb = emb.flatten()
    return emb / (np.linalg.norm(emb) + 1e-6)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

# ══════════════════════════════════════════════════════════
# CAMÉRA
# ══════════════════════════════════════════════════════════

def open_camera():
    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            cap.set(cv2.CAP_PROP_FPS,          15)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            log.info(f"Caméra /dev/video{idx}  {FRAME_W}x{FRAME_H}")
            return cap
        cap.release()
    raise RuntimeError("Aucune caméra trouvée.")

camera        = open_camera()
lock          = threading.Lock()
latest_frame  = None
frame_counter = 0
last_results  = []

pir_state = {
    'active':        False,
    'last_trigger':  0,
    'trigger_count': 0,
}
pir_lock = threading.Lock()

def capture_thread():
    global latest_frame
    while True:
        ret, frame = camera.read()
        if ret:
            with lock:
                latest_frame = frame.copy()
        else:
            time.sleep(0.03)

threading.Thread(target=capture_thread, daemon=True).start()

def pir_thread():
    log.info("Thread PIR démarré" if PIR_ENABLED else "Thread PIR (simulation)")
    last_val = 0
    while True:
        val = gpio_read() if PIR_ENABLED else 0
        now = time.time()
        with pir_lock:
            if val == 1 and last_val == 0:
                pir_state['active']        = True
                pir_state['last_trigger']  = now
                pir_state['trigger_count'] += 1
                log.info(f"PIR déclenché #{pir_state['trigger_count']}")
                with stats_lock:
                    stats['pir_triggers'] = pir_state['trigger_count']
            if val == 1:
                pir_state['last_trigger'] = now
                pir_state['active']       = True
            if pir_state['active'] and (now - pir_state['last_trigger']) > WAKE_TIMEOUT:
                pir_state['active'] = False
                log.info("PIR timeout — veille")
        last_val = val
        time.sleep(0.1)

threading.Thread(target=pir_thread, daemon=True).start()

log.info("Attente première frame...")
while True:
    with lock:
        if latest_frame is not None:
            break
    time.sleep(0.1)
log.info("Caméra prête.")

# ══════════════════════════════════════════════════════════
# RECONNAISSANCE
# ══════════════════════════════════════════════════════════

def recognize(frame):
    global frame_counter, last_results

    frame_counter += 1

    with pir_lock:
        cam_active = pir_state['active'] or not PIR_ENABLED

    if not cam_active:
        last_results = []
        _draw_idle(frame)
        return frame

    if frame_counter % DETECT_EVERY != 0:
        _draw(frame)
        return frame

    # ── FaceMesh detection + alignment ───────────────────
    detections = detect_and_align(frame)

    new_results = []
    for (x1, y1, x2, y2, aligned_roi) in detections:
        if aligned_roi is None or aligned_roi.size == 0:
            continue
        try:
            emb = get_embedding(aligned_roi)
        except Exception as e:
            log.warning(f"Embedding : {e}")
            continue

        name = "Inconnu"; confidence = 0.0; level = 'unknown'

        if len(known_embeddings) > 0:
            sims     = [cosine_similarity(emb, e) for e in known_embeddings]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            dist     = 1.0 - best_sim
            if dist < THRESHOLD_HIGH:
                name, confidence, level = known_names[best_idx], best_sim, 'high'
            elif dist < THRESHOLD_LOW:
                name, confidence, level = known_names[best_idx] + "?", best_sim, 'low'

        new_results.append((x1, y1, x2, y2, name, confidence, level))

        with stats_lock:
            stats['total_detections'] += 1
            if level != 'unknown':
                stats['recognized'] += 1
                entry = {
                    'name': name,
                    'conf': round(confidence * 100, 1),
                    'level': level,
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'date': datetime.now().strftime('%d/%m/%Y'),
                }
                stats['history'].insert(0, entry)
                if len(stats['history']) > 50:
                    stats['history'].pop()
            else:
                stats['unknown'] += 1

    last_results = new_results
    _draw(frame)
    return frame

def _draw(frame):
    GREEN  = (50,  220, 100)
    ORANGE = (30,  165, 255)
    RED    = (60,   60, 220)
    for (x1, y1, x2, y2, name, conf, level) in last_results:
        color = GREEN if level == 'high' else ORANGE if level == 'low' else RED
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({conf:.2f})" if level != 'unknown' else "Inconnu"
        cv2.putText(frame, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if PIR_ENABLED:
        with pir_lock:
            remaining = max(0, WAKE_TIMEOUT - (time.time() - pir_state['last_trigger']))
        cv2.putText(frame, f"PIR {remaining:.0f}s",
                    (frame.shape[1]-70, frame.shape[0]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 220, 100), 1)

def _draw_idle(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    h, w = frame.shape[:2]
    cv2.putText(frame, "EN VEILLE",
                (w//2 - 55, h//2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 160, 200), 2)
    cv2.putText(frame, "PIR: aucun mouvement",
                (w//2 - 75, h//2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 100, 130), 1)

# ══════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ══════════════════════════════════════════════════════════
# FLASK STREAM
# ══════════════════════════════════════════════════════════

def generate_frames():
    while True:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.1)
            continue
        frame = recognize(frame)
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.10)

# ══════════════════════════════════════════════════════════
# HTML PAGES — identiques à l'original
# ══════════════════════════════════════════════════════════

LOGIN_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GhostFaceNet — Login</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#060a0f;--panel:#0d1421;--border:#1a3a5c;--accent:#00d4ff;--accent2:#00ff88;--danger:#ff3366;--text:#c8d8e8;--muted:#4a6080}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);font-family:'Exo 2',sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center;overflow:hidden}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,212,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,0.03) 1px,transparent 1px);background-size:40px 40px;animation:gridMove 20s linear infinite;pointer-events:none}
@keyframes gridMove{0%{background-position:0 0}100%{background-position:40px 40px}}
.orb{position:fixed;border-radius:50%;filter:blur(80px);pointer-events:none}
.orb1{width:400px;height:400px;background:rgba(0,212,255,0.06);top:-100px;right:-100px}
.orb2{width:300px;height:300px;background:rgba(0,255,136,0.04);bottom:-80px;left:-80px}
.card{position:relative;background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:48px 40px;width:100%;max-width:400px;box-shadow:0 0 60px rgba(0,212,255,0.08)}
.card::before,.card::after{content:'';position:absolute;width:20px;height:20px;border-color:var(--accent);border-style:solid}
.card::before{top:-1px;left:-1px;border-width:2px 0 0 2px;border-radius:2px 0 0 0}
.card::after{bottom:-1px;right:-1px;border-width:0 2px 2px 0;border-radius:0 0 2px 0}
.logo{text-align:center;margin-bottom:36px}
.logo-icon{font-size:2.4rem;display:block;margin-bottom:10px;filter:drop-shadow(0 0 12px var(--accent))}
.logo h1{font-family:'Share Tech Mono',monospace;font-size:1.3rem;color:var(--accent);letter-spacing:3px;text-transform:uppercase}
.logo p{font-size:0.72rem;color:var(--muted);letter-spacing:2px;margin-top:4px;text-transform:uppercase}
.field{margin-bottom:20px}
.field label{display:block;font-size:0.68rem;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;font-family:'Share Tech Mono',monospace}
.field input{width:100%;background:rgba(0,212,255,0.04);border:1px solid var(--border);border-radius:8px;padding:12px 16px;color:var(--text);font-family:'Share Tech Mono',monospace;font-size:0.9rem;outline:none;transition:border-color 0.2s,box-shadow 0.2s}
.field input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(0,212,255,0.1)}
.field input::placeholder{color:var(--muted)}
.btn{width:100%;background:linear-gradient(135deg,var(--accent),#0099cc);color:#000;border:none;border-radius:8px;padding:14px;font-family:'Exo 2',sans-serif;font-weight:800;font-size:0.9rem;letter-spacing:2px;text-transform:uppercase;cursor:pointer;margin-top:8px;transition:opacity 0.2s,transform 0.1s,box-shadow 0.2s;box-shadow:0 4px 20px rgba(0,212,255,0.3)}
.btn:hover{opacity:0.9;box-shadow:0 6px 30px rgba(0,212,255,0.5)}
.btn:active{transform:scale(0.98)}
.error{background:rgba(255,51,102,0.1);border:1px solid rgba(255,51,102,0.3);border-radius:8px;padding:10px 14px;color:var(--danger);font-size:0.8rem;margin-bottom:20px;font-family:'Share Tech Mono',monospace}
.footer-note{text-align:center;margin-top:24px;font-size:0.65rem;color:var(--muted);letter-spacing:1px;font-family:'Share Tech Mono',monospace}
</style>
</head>
<body>
<div class="orb orb1"></div><div class="orb orb2"></div>
<div class="card">
  <div class="logo">
    <span class="logo-icon">👁</span>
    <h1>GhostFaceNet</h1>
    <p>BeagleBone Black · Accès Admin</p>
  </div>
  {% if error %}<div class="error">⚠ {{ error }}</div>{% endif %}
  <form method="POST" action="/login">
    <div class="field"><label>Identifiant</label><input type="text" name="username" placeholder="admin" autocomplete="off" required></div>
    <div class="field"><label>Mot de passe</label><input type="password" name="password" placeholder="••••••••" required></div>
    <button type="submit" class="btn">Connexion →</button>
  </form>
  <div class="footer-note">Système de reconnaissance faciale embarqué</div>
</div>
</body></html>"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GhostFaceNet Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#060a0f;--panel:#0d1421;--panel2:#111c2d;--border:#1a3a5c;--accent:#00d4ff;--accent2:#00ff88;--warn:#ffaa00;--danger:#ff3366;--text:#c8d8e8;--muted:#4a6080;--mono:'Share Tech Mono',monospace}
*{margin:0;padding:0;box-sizing:border-box}html,body{height:100%}
body{background:var(--bg);color:var(--text);font-family:'Exo 2',sans-serif;display:flex;flex-direction:column;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,212,255,0.02) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,0.02) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
header{position:relative;z-index:10;background:var(--panel);border-bottom:1px solid var(--border);padding:0 24px;height:56px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.hdr-left{display:flex;align-items:center;gap:14px}
.hdr-logo{font-family:var(--mono);font-size:1rem;color:var(--accent);letter-spacing:2px}
.hdr-logo span{color:var(--accent2)}
.live-pill{display:flex;align-items:center;gap:6px;background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.2);border-radius:20px;padding:3px 10px;font-size:0.68rem;color:var(--accent2);font-family:var(--mono)}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--accent2);box-shadow:0 0 6px var(--accent2);animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.hdr-right{display:flex;align-items:center;gap:10px}
.hdr-btn{background:transparent;border:1px solid var(--border);border-radius:8px;padding:6px 14px;color:var(--muted);font-family:var(--mono);font-size:0.72rem;cursor:pointer;transition:all 0.2s;letter-spacing:1px;text-decoration:none;display:inline-flex;align-items:center;gap:6px}
.hdr-btn:hover{border-color:var(--accent);color:var(--accent)}
.hdr-btn.danger:hover{border-color:var(--danger);color:var(--danger)}
.layout{position:relative;z-index:1;flex:1;display:grid;grid-template-columns:1fr 300px;gap:0;overflow:hidden}
.video-section{display:flex;flex-direction:column;padding:20px;gap:16px;overflow:hidden}
.video-frame{position:relative;background:#000;border:1px solid var(--border);border-radius:12px;overflow:hidden;flex:1;min-height:0}
.video-frame img{width:100%;height:100%;object-fit:contain;display:block}
.corner{position:absolute;width:18px;height:18px;border-color:var(--accent);border-style:solid}
.corner.tl{top:8px;left:8px;border-width:2px 0 0 2px}.corner.tr{top:8px;right:8px;border-width:2px 2px 0 0}
.corner.bl{bottom:8px;left:8px;border-width:0 0 2px 2px}.corner.br{bottom:8px;right:8px;border-width:0 2px 2px 0}
.vid-overlay{position:absolute;bottom:12px;left:12px;display:flex;gap:8px}
.vid-badge{background:rgba(6,10,15,0.85);border:1px solid var(--border);border-radius:6px;padding:4px 10px;font-family:var(--mono);font-size:0.65rem;color:var(--muted)}
.vid-badge.hi{border-color:rgba(0,212,255,0.3);color:var(--accent)}
.quick-stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;flex-shrink:0}
.qs{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px 14px;text-align:center}
.qs-val{font-family:var(--mono);font-size:1.3rem;font-weight:600;color:var(--accent);display:block}
.qs-lbl{font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-top:3px;display:block}
.qs.g .qs-val{color:var(--accent2)}.qs.r .qs-val{color:var(--danger)}.qs.w .qs-val{color:var(--warn)}
.sidebar{background:var(--panel);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}
.sb-section{padding:16px;border-bottom:1px solid var(--border);flex-shrink:0}
.sb-title{font-family:var(--mono);font-size:0.65rem;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:6px}
.sb-title::before{content:'';display:inline-block;width:3px;height:10px;background:var(--accent);border-radius:2px}
.action-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.action-btn{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:10px 8px;color:var(--text);font-family:var(--mono);font-size:0.68rem;cursor:pointer;transition:all 0.2s;text-align:center;letter-spacing:0.5px;display:flex;flex-direction:column;align-items:center;gap:4px}
.action-btn:hover{border-color:var(--accent);color:var(--accent);background:rgba(0,212,255,0.05)}
.action-btn .ico{font-size:1.2rem}
.action-btn.snap{border-color:rgba(0,255,136,0.2)}.action-btn.snap:hover{border-color:var(--accent2);color:var(--accent2)}
.persons-list{display:flex;flex-direction:column;gap:6px;max-height:160px;overflow-y:auto}
.person-item{display:flex;align-items:center;gap:8px;padding:6px 10px;background:var(--panel2);border-radius:6px;font-size:0.75rem}
.person-avatar{width:26px;height:26px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:800;color:#000;flex-shrink:0}
.person-name{color:var(--text);flex:1}
.history-scroll{flex:1;overflow-y:auto;padding:12px 16px}
.hist-empty{text-align:center;color:var(--muted);font-size:0.75rem;font-family:var(--mono);padding:24px 0}
.hist-item{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid rgba(26,58,92,0.5)}
.hist-item:last-child{border-bottom:none}
.hist-indicator{width:4px;height:32px;border-radius:2px;flex-shrink:0}
.hi-high{background:var(--accent2)}.hi-low{background:var(--warn)}.hi-unknown{background:var(--danger)}
.hist-info{flex:1;min-width:0}
.hist-name{font-size:0.78rem;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hist-meta{font-family:var(--mono);font-size:0.62rem;color:var(--muted);margin-top:2px}
.hist-conf{font-family:var(--mono);font-size:0.7rem;color:var(--accent);flex-shrink:0}
.toast{position:fixed;bottom:24px;right:24px;background:var(--panel);border:1px solid var(--accent2);border-radius:10px;padding:12px 18px;font-family:var(--mono);font-size:0.78rem;color:var(--accent2);box-shadow:0 8px 32px rgba(0,255,136,0.2);z-index:1000;transform:translateY(80px);opacity:0;transition:all 0.3s cubic-bezier(0.34,1.56,0.64,1);pointer-events:none}
.toast.show{transform:translateY(0);opacity:1}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(6,10,15,0.85);z-index:100;align-items:center;justify-content:center;backdrop-filter:blur(4px)}
.modal-overlay.open{display:flex}
.modal{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:24px;max-width:420px;width:90%;box-shadow:0 0 60px rgba(0,212,255,0.1)}
.modal h2{font-family:var(--mono);font-size:0.9rem;color:var(--accent);letter-spacing:2px;margin-bottom:16px}
.modal img{width:100%;border-radius:8px;border:1px solid var(--border);margin-bottom:14px}
.modal-path{font-family:var(--mono);font-size:0.68rem;color:var(--muted);word-break:break-all;margin-bottom:16px}
.modal-close{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:8px 20px;color:var(--text);font-family:var(--mono);font-size:0.75rem;cursor:pointer;transition:all 0.2s;float:right}
.modal-close:hover{border-color:var(--danger);color:var(--danger)}
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
@media(max-width:700px){.layout{grid-template-columns:1fr;grid-template-rows:auto auto}.sidebar{border-left:none;border-top:1px solid var(--border)}.quick-stats{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<header>
  <div class="hdr-left">
    <div class="hdr-logo">GHOST<span>FACE</span>NET</div>
    <div class="live-pill"><div class="live-dot"></div>LIVE</div>
    <div class="live-pill" id="pir-pill" style="background:rgba(80,80,80,0.15);border-color:rgba(80,80,80,0.3);color:#4a6080">
      <div style="width:7px;height:7px;border-radius:50%;background:#4a6080;transition:all 0.3s" id="pir-dot"></div>
      <span id="pir-label">PIR</span>
    </div>
  </div>
  <div class="hdr-right">
    <span id="clock" style="font-family:var(--mono);font-size:0.72rem;color:var(--muted)"></span>
    <a href="/logout" class="hdr-btn danger">⏻ Logout</a>
  </div>
</header>
<div class="layout">
  <div class="video-section">
    <div class="video-frame">
      <img src="/video" id="stream" alt="stream" onerror="setTimeout(()=>{this.src='/video?t='+Date.now()},2000)">
      <div class="corner tl"></div><div class="corner tr"></div>
      <div class="corner bl"></div><div class="corner br"></div>
      <div class="vid-overlay">
        <div class="vid-badge hi">{{ "{}x{}".format(200,180) }}</div>
        <div class="vid-badge">GhostFaceNet INT8 + FaceMesh</div>
        <div class="vid-badge">T=0.60/0.75</div>
      </div>
    </div>
    <div class="quick-stats">
      <div class="qs"><span class="qs-val" id="qs-total">0</span><span class="qs-lbl">Détections</span></div>
      <div class="qs g"><span class="qs-val" id="qs-recog">0</span><span class="qs-lbl">Reconnus</span></div>
      <div class="qs r"><span class="qs-val" id="qs-unk">0</span><span class="qs-lbl">Inconnus</span></div>
      <div class="qs w"><span class="qs-val" id="qs-snaps">0</span><span class="qs-lbl">Snapshots</span></div>
      <div class="qs" id="qs-pir-card"><span class="qs-val" id="qs-pir">0</span><span class="qs-lbl">PIR triggers</span></div>
    </div>
  </div>
  <div class="sidebar">
    <div class="sb-section">
      <div class="sb-title">Capteur PIR — GPIO 47</div>
      <div id="pir-banner" style="border-radius:10px;padding:14px 16px;margin-bottom:10px;transition:all 0.4s ease;background:rgba(80,80,80,0.1);border:1px solid rgba(80,80,80,0.3)">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
          <div id="pir-big-dot" style="width:16px;height:16px;border-radius:50%;background:#4a6080;transition:all 0.3s;flex-shrink:0"></div>
          <span id="pir-status-text" style="font-family:var(--mono);font-size:0.82rem;color:#4a6080;font-weight:700;letter-spacing:1px;text-transform:uppercase;transition:color 0.3s">EN VEILLE</span>
        </div>
        <div style="margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span style="font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px">Durée active</span>
            <span id="pir-timer-val" style="font-family:var(--mono);font-size:0.65rem;color:var(--accent2)">—</span>
          </div>
          <div style="height:4px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden">
            <div id="pir-progress" style="height:100%;width:0%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:2px;transition:width 0.5s linear"></div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
          <div style="text-align:center;padding:8px;background:rgba(0,0,0,0.25);border-radius:7px">
            <div id="pir-count-val" style="font-family:var(--mono);font-size:1.2rem;font-weight:700;color:var(--accent)">0</div>
            <div style="font-size:0.58rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:2px">Triggers</div>
          </div>
          <div style="text-align:center;padding:8px;background:rgba(0,0,0,0.25);border-radius:7px">
            <div style="font-family:var(--mono);font-size:1.2rem;font-weight:700;color:var(--accent2)">P8_15</div>
            <div style="font-size:0.58rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:2px">GPIO 47</div>
          </div>
        </div>
      </div>
    </div>
    <div class="sb-section">
      <div class="sb-title">Actions</div>
      <div class="action-grid">
        <button class="action-btn snap" onclick="takeSnapshot()"><span class="ico">📷</span>Snapshot</button>
        <button class="action-btn" onclick="refreshStream()"><span class="ico">🔄</span>Refresh</button>
        <button class="action-btn" onclick="clearHistory()"><span class="ico">🗑</span>Effacer</button>
        <a href="/snapshots-list" class="action-btn" style="text-decoration:none"><span class="ico">🖼</span>Galerie</a>
      </div>
    </div>
    <div class="sb-title" style="padding:12px 16px 0;flex-shrink:0">
      <span style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);letter-spacing:2px;text-transform:uppercase;display:flex;align-items:center;justify-content:space-between;width:100%">
        <span style="display:flex;align-items:center;gap:6px">
          <span style="display:inline-block;width:3px;height:10px;background:var(--accent);border-radius:2px"></span>Historique
        </span>
        <button onclick="clearHistory()" style="background:transparent;border:1px solid rgba(255,255,255,0.1);border-radius:4px;padding:2px 8px;color:var(--muted);font-family:var(--mono);font-size:0.58rem;cursor:pointer;transition:all 0.2s" onmouseover="this.style.borderColor='var(--danger)';this.style.color='var(--danger)'" onmouseout="this.style.borderColor='rgba(255,255,255,0.1)';this.style.color='var(--muted)'">effacer</button>
      </span>
    </div>
    <div class="history-scroll" id="history" style="flex:1;min-height:200px;max-height:300px;overflow-y:auto;padding:8px 14px">
      <div class="hist-empty" id="hist-empty">Aucune détection</div>
    </div>
    <div class="sb-section" style="flex-shrink:0">
      <div class="sb-title" onclick="togglePersons()" style="cursor:pointer;user-select:none">
        Base ({{ n_persons }} personnes) <span id="persons-toggle" style="margin-left:auto;font-size:0.8rem">▼</span>
      </div>
      <div id="persons-list" class="persons-list" style="max-height:120px;overflow-y:auto;display:none">
        {% for name in known_names %}
        <div class="person-item">
          <div class="person-avatar">{{ name[0].upper() }}</div>
          <div class="person-name">{{ name }}</div>
        </div>
        {% endfor %}
      </div>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>
<div class="modal-overlay" id="modal">
  <div class="modal">
    <h2>📷 SNAPSHOT</h2>
    <img id="modal-img" src="" alt="">
    <div class="modal-path" id="modal-path"></div>
    <button class="modal-close" onclick="closeModal()">Fermer</button>
  </div>
</div>
<script>
let history=[],snapshotCount=0;
function updateClock(){document.getElementById('clock').textContent=new Date().toLocaleTimeString('fr-FR')}
setInterval(updateClock,1000);updateClock();
async function pollStats(){
  try{
    const d=await(await fetch('/api/stats')).json();
    document.getElementById('qs-total').textContent=d.total;
    document.getElementById('qs-recog').textContent=d.recognized;
    document.getElementById('qs-unk').textContent=d.unknown;
    document.getElementById('qs-snaps').textContent=d.snapshots;
    document.getElementById('qs-pir').textContent=d.pir_triggers||0;
    const pirPill=document.getElementById('pir-pill'),pirDot=document.getElementById('pir-dot'),pirLabel=document.getElementById('pir-label');
    const pirBigDot=document.getElementById('pir-big-dot'),pirStatus=document.getElementById('pir-status-text');
    const pirTimer=document.getElementById('pir-timer-val'),pirCount=document.getElementById('pir-count-val');
    const banner=document.getElementById('pir-banner'),progress=document.getElementById('pir-progress');
    if(d.pir_enabled){
      pirCount.textContent=d.pir_triggers||0;
      if(d.pir_active){
        pirPill.style.cssText='background:rgba(0,255,136,0.08);border-color:rgba(0,255,136,0.3);color:var(--accent2)';
        pirDot.style.cssText='background:var(--accent2);box-shadow:0 0 8px var(--accent2)';
        pirLabel.textContent='PIR ●';
        banner.style.cssText='border-radius:10px;padding:14px 16px;margin-bottom:10px;background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.3)';
        pirBigDot.style.cssText='width:16px;height:16px;border-radius:50%;background:var(--accent2);box-shadow:0 0 12px var(--accent2)';
        pirStatus.style.color='var(--accent2)';pirStatus.textContent='🔴 PRÉSENCE DÉTECTÉE';
        if(d.pir_remaining!==undefined){
          progress.style.width=Math.max(0,(d.pir_remaining/10)*100)+'%';
          pirTimer.textContent=d.pir_remaining.toFixed(1)+'s';
        }
      } else {
        pirPill.style.cssText='background:rgba(80,80,80,0.08);border-color:rgba(80,80,80,0.2);color:#4a6080';
        pirDot.style.cssText='background:#4a6080;box-shadow:none';
        pirLabel.textContent='PIR';
        banner.style.cssText='border-radius:10px;padding:14px 16px;margin-bottom:10px;background:rgba(80,80,80,0.08);border:1px solid rgba(80,80,80,0.25)';
        pirBigDot.style.cssText='width:16px;height:16px;border-radius:50%;background:#4a6080;box-shadow:none';
        pirStatus.style.color='#4a6080';pirStatus.textContent='⚪ EN VEILLE';
        pirTimer.textContent='—';progress.style.width='0%';
      }
    } else { pirPill.style.display='none'; }
    if(d.history.length!==history.length){history=d.history;renderHistory();}
  }catch(e){}
}
setInterval(pollStats,2000);pollStats();
function renderHistory(){
  const c=document.getElementById('history'),e=document.getElementById('hist-empty');
  if(history.length===0){e.style.display='block';c.querySelectorAll('.hist-item').forEach(x=>x.remove());return;}
  e.style.display='none';c.querySelectorAll('.hist-item').forEach(x=>x.remove());
  history.slice(0,30).forEach(item=>{
    const d=document.createElement('div');d.className='hist-item';
    const ic=item.level==='high'?'hi-high':item.level==='low'?'hi-low':'hi-unknown';
    d.innerHTML=`<div class="hist-indicator ${ic}"></div><div class="hist-info"><div class="hist-name">${item.name}</div><div class="hist-meta">${item.date} · ${item.time}</div></div><div class="hist-conf">${item.conf}%</div>`;
    c.appendChild(d);
  });
}
async function takeSnapshot(){
  try{const d=await(await fetch('/api/snapshot',{method:'POST'})).json();
    if(d.ok){snapshotCount++;document.getElementById('qs-snaps').textContent=snapshotCount;
      showToast('📷 Snapshot sauvegardé');
      document.getElementById('modal-img').src='/snapshot-img/'+d.filename;
      document.getElementById('modal-path').textContent=d.path;
      document.getElementById('modal').classList.add('open');
    }
  }catch(e){showToast('Erreur snapshot');}
}
function closeModal(){document.getElementById('modal').classList.remove('open')}
document.getElementById('modal').addEventListener('click',function(e){if(e.target===this)closeModal()});
function refreshStream(){document.getElementById('stream').src='/video?t='+Date.now();showToast('🔄 Stream rechargé')}
async function clearHistory(){await fetch('/api/clear-history',{method:'POST'});history=[];renderHistory();showToast('🗑 Historique effacé')}
function showToast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),3000)}
function togglePersons(){const l=document.getElementById('persons-list'),t=document.getElementById('persons-toggle');if(l.style.display==='none'){l.style.display='flex';t.textContent='▲'}else{l.style.display='none';t.textContent='▼'}}
</script>
</body></html>"""

GALLERY_HTML = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Galerie Snapshots</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#060a0f;--panel:#0d1421;--border:#1a3a5c;--accent:#00d4ff;--text:#c8d8e8;--muted:#4a6080}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Exo 2',sans-serif;min-height:100vh;padding:24px}
.header{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px}
h1{font-family:'Share Tech Mono',monospace;color:var(--accent);font-size:1rem;letter-spacing:2px}
.back{background:transparent;border:1px solid var(--border);border-radius:8px;padding:6px 14px;color:var(--muted);font-family:'Share Tech Mono',monospace;font-size:0.72rem;cursor:pointer;text-decoration:none;transition:all 0.2s}
.back:hover{border-color:var(--accent);color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;overflow:hidden;transition:border-color 0.2s}
.card:hover{border-color:var(--accent)}
.card img{width:100%;display:block;aspect-ratio:4/3;object-fit:cover}
.card-info{padding:8px 10px}
.card-name{font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:var(--muted)}
.empty{text-align:center;color:var(--muted);font-family:'Share Tech Mono',monospace;font-size:0.8rem;padding:60px 0}
</style></head><body>
<div class="header"><h1>📷 GALERIE SNAPSHOTS</h1><a href="/" class="back">← Dashboard</a></div>
{% if files %}
<div class="grid">{% for f in files %}<div class="card"><img src="/snapshot-img/{{ f }}" alt="{{ f }}" onerror="this.style.display='none'"><div class="card-info"><div class="card-name">{{ f }}</div></div></div>{% endfor %}</div>
{% else %}<div class="empty">Aucun snapshot pour l'instant.<br>Utilisez le bouton 📷 sur le dashboard.</div>{% endif %}
</body></html>"""

# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

from flask import render_template_string, send_file

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if (request.form.get('username') == ADMIN_USER and
                request.form.get('password') == ADMIN_PASS):
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        error = "Identifiant ou mot de passe incorrect"
    return render_template_string(LOGIN_HTML, error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    return render_template_string(
        DASHBOARD_HTML,
        known_names=known_names,
        n_persons=len(known_names),
    )

@app.route('/video')
@login_required
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
@login_required
def api_stats():
    with stats_lock:
        with pir_lock:
            pir_active    = pir_state['active']
            pir_count     = pir_state['trigger_count']
            pir_remaining = max(0, WAKE_TIMEOUT - (time.time() - pir_state['last_trigger']))
        return jsonify({
            'total':         stats['total_detections'],
            'recognized':    stats['recognized'],
            'unknown':       stats['unknown'],
            'snapshots':     stats['snapshots'],
            'pir_triggers':  pir_count,
            'pir_active':    pir_active,
            'pir_remaining': round(pir_remaining, 1),
            'pir_enabled':   PIR_ENABLED,
            'uptime':        int(time.time() - stats['start_time']),
            'history':       stats['history'][:30],
        })

@app.route('/api/snapshot', methods=['POST'])
@login_required
def api_snapshot():
    with lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return jsonify({'ok': False, 'error': 'Pas de frame'}), 500
    ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"snap_{ts}.jpg"
    path  = os.path.join(SNAPSHOT_DIR, fname)
    cv2.imwrite(path, frame)
    with stats_lock:
        stats['snapshots'] += 1
    log.info(f"Snapshot : {path}")
    return jsonify({'ok': True, 'filename': fname, 'path': path})

@app.route('/snapshot-img/<filename>')
@login_required
def snapshot_img(filename):
    path = os.path.join(SNAPSHOT_DIR, filename)
    if not os.path.exists(path):
        return '', 404
    return send_file(path, mimetype='image/jpeg')

@app.route('/snapshots-list')
@login_required
def snapshots_list():
    files = sorted(
        [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')],
        reverse=True
    )
    return render_template_string(GALLERY_HTML, files=files)

@app.route('/api/clear-history', methods=['POST'])
@login_required
def clear_history():
    with stats_lock:
        stats['history'] = []
    return jsonify({'ok': True})

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def shutdown(sig, frame):
    log.info("Arrêt...")
    camera.release()
    gpio_cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

if __name__ == '__main__':
    log.info(f"Flask sur http://0.0.0.0:{FLASK_PORT}")
    log.info(f"Détection : FaceMesh (face_align_lite.py)")
    log.info(f"Seuils    : high={THRESHOLD_HIGH}  low={THRESHOLD_LOW}")
    log.info(f"Login     : {ADMIN_USER} / {ADMIN_PASS}")
    app.run(host='0.0.0.0', port=FLASK_PORT,
            threaded=True, use_reloader=False, debug=False)
