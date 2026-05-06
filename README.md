# 👁 GhostFaceNet — Reconnaissance Faciale Embarquée sur BeagleBone Black

![CI/CD](https://img.shields.io/badge/CI%2FCD-passing-brightgreen?style=flat-square&logo=github-actions)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![FaceRecognition](https://img.shields.io/badge/FaceRecognition-GhostFaceNet-orange?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-BeagleBoneBlack-red?style=flat-square)
![Status](https://img.shields.io/badge/status-Active-success?style=flat-square)
![TFLite](https://img.shields.io/badge/TFLite-INT8-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy%20PC-98.7%25-brightgreen?style=flat-square)
![Accuracy BBB](https://img.shields.io/badge/Accuracy%20BBB-76.8%25-yellow?style=flat-square)

> Système de reconnaissance faciale temps réel basé sur **GhostFaceNet INT8 TFLite** déployé sur **BeagleBone Black** (ARM Cortex-A8 1GHz), avec dashboard Flask, capteur PIR et évaluation complète des performances.

---

## 📋 Table des matières

- [Description](#-description)
- [Architecture](#-architecture)
- [Matériel requis](#-matériel-requis)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Évaluation des performances](#-évaluation-des-performances)
- [Résultats](#-résultats)
- [Auteur](#-auteur)

---

## 📌 Description

Ce projet implémente un système de **reconnaissance faciale embarqué** complet sur BeagleBone Black. Il combine :

- **GhostFaceNet** quantisé en INT8 pour l'inférence légère
- **Alignement facial** via Haar Cascade multi-passes (face_align_lite)
- **Dashboard web** Flask avec stream MJPEG en temps réel
- **Capteur PIR** HC-SR501 pour l'activation par détection de mouvement
- **Pipeline d'évaluation** complet (PC + BBB) avec métriques FAR/FRR/EER

### Fonctionnalités principales

- Reconnaissance de 20 personnes en temps réel à 238ms/frame
- Dashboard web accessible depuis n'importe quel navigateur sur le réseau local
- Activation automatique par détection de mouvement (PIR)
- Snapshots, historique des reconnaissances, statistiques en direct
- Double seuil de confiance (haute / faible) pour la robustesse

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     PC (Entraînement)                    │
│                                                         │
│  known_users/          precalculate_embeddings.py       │
│  ├── Alice/    ──────► FaceMesh (MediaPipe)  ─────────► │
│  ├── Bob/              get_embedding()                  │
│  └── ...               moyenne L2-normalisée            │
│                                    │                    │
│                                    ▼                    │
│                         known_embeddings.pkl            │
└─────────────────────────────────┬───────────────────────┘
                                  │ scp
                                  ▼
┌─────────────────────────────────────────────────────────┐
│                  BeagleBone Black (Production)           │
│                                                         │
│  Caméra USB                                             │
│      │                                                  │
│      ▼                                                  │
│  face_align_lite.py (Haar Cascade multi-passes)         │
│      │                                                  │
│      ▼                                                  │
│  ghostfacenet_int8.tflite ──► embedding 512D            │
│      │                                                  │
│      ▼                                                  │
│  cosine_similarity vs known_embeddings.pkl              │
│      │                                                  │
│      ├── dist < 0.60 ──► "Alice"   (haute confiance)   │
│      ├── dist < 0.75 ──► "Alice?"  (faible confiance)  │
│      └── dist ≥ 0.75 ──► "Inconnu"                     │
│                                                         │
│  Flask Dashboard (port 5000)                            │
│  PIR HC-SR501 (GPIO 47 / P8_15)                        │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Matériel requis

| Composant | Détail |
|---|---|
| **BeagleBone Black** | Rev C, ARM Cortex-A8 1GHz, 512MB RAM |
| **Caméra USB** | Compatible V4L2, résolution min 320×240 |
| **Capteur PIR** | HC-SR501, branché sur P8_15 (GPIO 47) |
| **PC** | Pour l'entraînement et le calcul des embeddings |

### Branchement PIR

```
HC-SR501          BeagleBone Black
─────────────────────────────────
VCC    ──────────► P9_5  (5V)
GND    ──────────► P9_1  (GND)
OUT    ──────────► P8_15 (GPIO 47)
```

---

## 📁 Structure du projet

```
ghostfacenet-bbb/
│
├── README.md
├── requirements_pc.txt          # dépendances PC
├── requirements_bbb.txt         # dépendances BBB
├── .gitignore
│
├── pc/
│   ├── precalculate_embeddings.py   # calcul embeddings avec FaceMesh
│   ├── evaluate.py                  # évaluation précision sur PC
│   └── check_mesh.py                # vérification images dataset
│
├── bbb/
│   ├── main.py                  # application principale Flask
│   ├── face_align_lite.py       # alignement facial léger (Haar)
│   └── evaluate_bbb.py          # évaluation précision sur BBB
│
├── models/
│   └── .gitkeep                 # ghostfacenet_int8.tflite (non versionné)
│
└── known_users/
    └── .gitkeep                 # dataset visages (non versionné)
```

> **Note** : Le modèle `.tflite` et le dataset `known_users/` ne sont pas inclus dans le dépôt pour des raisons de taille et de confidentialité.

---

## ⚙️ Installation

### Sur le PC

```bash
# Cloner le dépôt
git clone https://github.com/TON_USERNAME/ghostfacenet-bbb.git
cd ghostfacenet-bbb

# Installer les dépendances
pip install -r requirements_pc.txt

# Préparer le dataset
# Créer known_users/<nom_personne>/ et y placer les photos
mkdir -p known_users/alice known_users/bob

# Calculer les embeddings
cd pc/
python precalculate_embeddings.py

# Vérifier la qualité des images
python check_mesh.py

# Évaluer les performances
python evaluate.py
```

### Sur le BeagleBone Black

```bash
# Installer les dépendances
pip3 install tflite-runtime flask opencv-python-headless numpy --break-system-packages

# Copier les fichiers depuis le PC
scp pc/known_embeddings.pkl debian@BBB_IP:~/ghost2/
scp bbb/main.py              debian@BBB_IP:~/ghost2/
scp bbb/face_align_lite.py   debian@BBB_IP:~/ghost2/
scp bbb/evaluate_bbb.py      debian@BBB_IP:~/ghost2/
scp -r known_users/          debian@BBB_IP:~/ghost2/

# Se connecter au BBB
ssh debian@BBB_IP
cd ~/ghost2

# Lancer l'application
python3 main.py
```

---

## 🚀 Utilisation

### Lancer le système

```bash
# Sur le BBB
python3 main.py
```

```
2026-04-06 10:00:00 [INFO] tflite_runtime chargé.
2026-04-06 10:00:01 [INFO] Modèle TFLite prêt.
2026-04-06 10:00:02 [INFO] 20 personne(s) chargées
2026-04-06 10:00:03 [INFO] PIR prêt : gpiod gpiochip0 line 15 (P8_15)
2026-04-06 10:00:03 [INFO] Flask sur http://0.0.0.0:5000
```

### Accéder au dashboard

Ouvre un navigateur sur le même réseau :

```
http://BBB_IP:5000
```

Identifiants par défaut :
```
Utilisateur : admin
Mot de passe : bbb2024
```

### Dashboard

Le dashboard affiche en temps réel :
- **Stream MJPEG** avec boîtes de détection colorées
  - 🟢 Vert : reconnaissance haute confiance
  - 🟠 Orange : reconnaissance faible confiance
  - 🔴 Rouge : inconnu
- **Statistiques** : détections, reconnus, inconnus, snapshots, triggers PIR
- **Historique** des 30 dernières reconnaissances
- **État PIR** avec timer de veille
- **Galerie** des snapshots

### Évaluation sur BBB

```bash
# Copier known_users sur BBB si pas déjà fait
scp -r known_users/ debian@BBB_IP:~/ghost2/

# Lancer l'évaluation
ssh debian@BBB_IP
cd ~/ghost2
python3 evaluate_bbb.py

# Récupérer les résultats
scp debian@BBB_IP:~/ghost2/evaluation_results.txt .
```

---

## 📊 Évaluation des performances

### Méthodologie

L'évaluation utilise la méthode **Repeated Random Split** (5 répétitions, 80% train / 20% test) sur le dataset `known_users`. Cette méthode simule exactement le comportement de `main.py` en production.

### Métriques mesurées

| Métrique | Description |
|---|---|
| **Accuracy** | % de photos correctement identifiées |
| **Précision** | % de reconnaissances qui sont correctes |
| **Rappel** | % de vrais visages correctement reconnus |
| **FAR** | Taux d'acceptation de faux visages |
| **FRR** | Taux de rejet de vrais visages |
| **EER** | Point d'équilibre FAR = FRR |

---

## 📈 Résultats

### PC (evaluate.py) — Pipeline FaceMesh complet

| Métrique | Valeur |
|---|---|
| **Accuracy** | 98.7% |
| **Précision** | 100.0% |
| **Rappel** | 98.9% |
| **F1-score** | 99.5% |
| **FAR** | 0.67% |
| **FRR** | 1.35% |
| **EER** | 1.09% @ seuil 0.710 |
| **AUC-ROC** | 0.9986 |

### BBB (evaluate_bbb.py) — Pipeline face_align_lite

| Métrique | Valeur |
|---|---|
| **Accuracy** | 76.8% |
| **Précision** | 99.1% |
| **Rappel** | 76.8% |
| **F1-score** | 86.5% |
| **FAR** | 1.49% |
| **FRR** | 22.75% |
| **EER** | 14.37% @ seuil 0.886 |
| **Inférence** | 238 ms/image |

### Comparaison PC vs BBB

| | PC | BBB |
|---|---|---|
| Détection visage | MediaPipe FaceMesh | Haar Cascade multi-passes |
| Seuil HIGH | 0.69 | 0.60 |
| Seuil LOW | — | 0.75 |
| Accuracy | **98.7%** | **76.8%** |
| Mauvais noms | 0 | 15 |
| Temps inférence | ~50ms | ~238ms |

### Analyse de l'écart PC / BBB

L'écart de ~22% s'explique par :
1. **Alignement moins précis** : Haar Cascade estime 5 landmarks géométriquement vs 468 points FaceMesh
2. **141 images no-mesh** : visages non détectés → fallback image entière → embedding dégradé
3. **Seuils non optimaux** : les distances cosinus sont plus grandes avec Haar qu'avec FaceMesh

### Accuracy par personne (BBB)

```
Grant gustin        91.7% ███████████████████
Ala                 93.3% ███████████████████
Melissa fumero      89.5% ###################
Kiernen shipka      87.1% ###################
Ellen page          85.2% #################
...
Bobby morley        61.2% ############
Barbara palvin      63.3% ############
Gal gadot           51.2% ##########
```

---

## 🔬 Technologies utilisées

| Technologie | Rôle |
|---|---|
| **GhostFaceNet** | Modèle de reconnaissance faciale (512D embeddings) |
| **TFLite INT8** | Quantisation pour déploiement embarqué |
| **OpenCV** | Traitement d'image, Haar Cascade, streaming |
| **MediaPipe** | Alignement facial précis (PC uniquement) |
| **Flask** | Serveur web dashboard |
| **NumPy** | Calculs vectoriels (similarité cosinus) |
| **gpiod** | Lecture GPIO PIR sur BBB |
| **scikit-learn** | Métriques d'évaluation (ROC, confusion matrix) |
| **matplotlib** | Génération rapport visuel |

---

## 📝 Configuration

Les paramètres principaux sont dans `bbb/main.py` :

```python
TFLITE_MODEL   = "ghostfacenet_int8.tflite"
EMBEDDINGS_PKL = "known_embeddings.pkl"
THRESHOLD_HIGH = 0.60    # haute confiance
THRESHOLD_LOW  = 0.75    # faible confiance
FRAME_W        = 200     # largeur frame caméra
FRAME_H        = 180     # hauteur frame caméra
DETECT_EVERY   = 8       # 1 détection toutes les 8 frames
WAKE_TIMEOUT   = 10      # secondes d'activation après PIR
FLASK_PORT     = 5000
ADMIN_USER     = "admin"
ADMIN_PASS     = "bbb2024"
```

---

## 👤 Auteur

**Youssef Issaoui**
Projet de Fin d'Études (PFE)

---

## 📄 Licence

Ce projet est développé dans le cadre d'un PFE académique.
