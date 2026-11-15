"""
Live leaf-disease detection for the dashboard.

Loads the trained MobileNetV2 model in a background thread (so the dashboard
starts instantly) and classifies camera frames sent by the browser.

Two guards against confident nonsense:
  1. Leaf check -- if too few pixels are leaf-coloured (green through
     yellow-brown), the frame is reported as "no leaf" and not classified.
     Needed because the model happily gives 98% to random noise.
  2. Smoothing -- probabilities are averaged over the last few frames, so the
     answer doesn't flicker between classes.
"""

import io
import os
import threading
import time
from collections import deque

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CANDIDATES = [
    ("model/phytocast_model_v2.keras", "model/class_names_v2.txt", "v2 · 87,867 images"),
    ("model/phytocast_model.keras", "model/class_names.txt", "v1 · 54,305 images"),
]

IMG_SIZE = 224
CONFIDENCE = 0.70      # below this, answer "not sure"
LEAF_MIN_RATIO = 0.20  # below this share of leaf-coloured pixels, "no leaf"
SMOOTH_FRAMES = 4

_model = None
_names = []
_status = "loading"
_info = ""
_lock = threading.Lock()
_recent = deque(maxlen=SMOOTH_FRAMES)


def _load():
    global _model, _names, _status, _info
    try:
        import tensorflow as tf
        for model_path, names_path, info in MODEL_CANDIDATES:
            mp, np_ = os.path.join(ROOT, model_path), os.path.join(ROOT, names_path)
            if os.path.exists(mp) and os.path.exists(np_):
                model = tf.keras.models.load_model(mp)
                model(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), "float32"), training=False)  # warm up
                with open(np_) as f:
                    _names = f.read().splitlines()
                _model, _info, _status = model, info, "ready"
                return
        _status = "unavailable: no model file in model/"
    except Exception as e:  # TensorFlow missing or model incompatible
        _status = f"unavailable: {type(e).__name__}: {e}"


def start():
    threading.Thread(target=_load, daemon=True).start()


def status():
    return {"status": _status, "info": _info, "classes": len(_names)}


def pretty(label):
    crop, _, cond = label.partition("___")
    crop = crop.replace("_", " ").strip()
    cond = cond.replace("_", " ").strip() or "Unknown"
    cond = "Healthy" if cond.lower() == "healthy" else cond[0].upper() + cond[1:]
    return crop, cond


def _leaf_ratio(img):
    hsv = np.asarray(img.convert("HSV"), dtype=np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # hue ~20°-95° (yellow-brown to green), reasonably saturated, not dark
    mask = (h >= 14) & (h <= 67) & (s >= 60) & (v >= 40)
    return float(mask.mean())


def classify(jpeg_bytes):
    if _status != "ready":
        return {"verdict": "unavailable", "status": _status}

    t0 = time.time()
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    img = img.resize((IMG_SIZE, IMG_SIZE))

    leaf = _leaf_ratio(img)
    if leaf < LEAF_MIN_RATIO:
        _recent.clear()
        return {"verdict": "no_leaf", "leaf": round(leaf, 2), "ms": round((time.time() - t0) * 1000)}

    x = np.asarray(img, dtype="float32")[None] / 127.5 - 1.0   # MobileNetV2 preprocessing
    with _lock:
        probs = np.asarray(_model(x, training=False))[0]
        _recent.append(probs)
        avg = np.mean(_recent, axis=0)

    order = np.argsort(avg)[::-1][:3]
    top = []
    for i in order:
        crop, cond = pretty(_names[i])
        top.append({"label": _names[i], "crop": crop, "condition": cond, "p": round(float(avg[i]), 3)})

    best = top[0]
    verdict = "unsure" if best["p"] < CONFIDENCE else ("healthy" if best["condition"] == "Healthy" else "disease")
    return {"verdict": verdict, "top": top, "leaf": round(leaf, 2),
            "ms": round((time.time() - t0) * 1000)}
