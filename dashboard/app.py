"""
PhytoCast Dashboard (M6)
------------------------
Live view of the ESP32: leaf-spot risk, actuator states, sensor trends, an
activity log of everything the system does, and an assistant grounded in
the live data.

RUN:
    pip install -r requirements.txt
    python app.py
Open http://localhost:5000

The ESP32 (firmware/phytocast/phytocast.ino) POSTs to http://<this-laptop-IP>:5000/update
every ~3 s. With no ESP32, run send_test_data.py in a second terminal.

Live camera detection needs TensorFlow (pip install tensorflow pillow) and the
model in ../model/. Without it the rest of the dashboard still works.
"""

import os
# Must be set before numpy/TensorFlow load -- otherwise OpenBLAS can fail to
# allocate memory on this laptop.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from flask import Flask, request, jsonify, Response
import time
import requests
import detector

app = Flask(__name__)
detector.start()   # loads the model in the background

camera = {"last": None, "logged_label": None, "logged_ts": 0}

# ---- plant profile (mirrors the firmware's PLANT PROFILE block) ----
PLANT = {
    "name": "Syngonium",
    "latin": "Syngonium podophyllum",
    "comfort": "18–27 °C",
    "heat": 30,
    "disease_band": "20–30 °C at ≥85% RH",
    "disease_humidity": 85,
}

# Soil calibration from the real pot (raw ADC: higher = drier)
SOIL_WET_RAW = 1150
SOIL_DRY_RAW = 2200
SOIL_THRESHOLD_RAW = 1950


def soil_pct(raw):
    if raw is None:
        return None
    pct = (SOIL_DRY_RAW - raw) / (SOIL_DRY_RAW - SOIL_WET_RAW) * 100
    return round(max(0, min(100, pct)))


FIELDS = ("plant", "temp", "humidity", "risk", "soil_raw", "soil_state", "pumping",
          "shade", "bright", "fan", "fan_reason", "tank_cm", "tank_low", "rain", "motion")

state = {
    "plant": PLANT["name"], "temp": None, "humidity": None, "risk": "LOW",
    "soil_raw": None, "soil_state": "ok", "pumping": False,
    "shade": False, "bright": False, "fan": False, "fan_reason": "",
    "tank_cm": None, "tank_low": False, "rain": False, "motion": False,
    "last_update": None, "last_update_ts": 0,
}

HISTORY = []          # recent points for the sparklines
HISTORY_LIMIT = 60
EVENTS = []           # newest first
EVENTS_LIMIT = 40

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

FAN_REASON = {"heat": "heat above 30 °C", "humid": "humid, still air",
              "risk": "high leaf-spot risk"}


def add_event(kind, text):
    EVENTS.insert(0, {"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text})
    del EVENTS[EVENTS_LIMIT:]


def record_transitions(prev, cur):
    if not prev["last_update_ts"]:
        add_event("system", "ESP32 connected — monitoring started")

    if cur["pumping"] and not prev["pumping"]:
        add_event("water", f"Watering pulse — soil read {cur['soil_raw']} (dry)")
    if cur["soil_state"] == "rain-skip" and prev["soil_state"] != "rain-skip":
        add_event("water", "Soil dry, but it's raining — watering skipped")

    if cur["shade"] != prev["shade"]:
        if cur["shade"]:
            why = "bright light" if cur["bright"] else "heat"
            add_event("climate", f"Shade deployed — {why}")
        else:
            add_event("climate", "Shade retracted")

    if cur["fan"] != prev["fan"]:
        if cur["fan"]:
            add_event("climate", f"Fan on — {FAN_REASON.get(cur['fan_reason'], 'climate')}")
        else:
            add_event("climate", "Fan off")

    if cur["risk"] != prev["risk"]:
        kind = "alert" if cur["risk"] == "HIGH" else "climate"
        add_event(kind, f"Leaf-spot risk {prev['risk']} → {cur['risk']}")

    if cur["motion"] and not prev["motion"]:
        add_event("alert", "Motion near the plant — deterrent sounded")
    if cur["tank_low"] and not prev["tank_low"]:
        add_event("alert", "Reservoir low — refill soon")
    if cur["rain"] and not prev["rain"]:
        add_event("water", "Rain detected")


@app.route("/update", methods=["POST"])
def update():
    data = request.get_json(force=True, silent=True) or {}
    prev = dict(state)
    for key in FIELDS:
        if key in data:
            state[key] = data[key]
    state["last_update"] = time.strftime("%H:%M:%S")
    state["last_update_ts"] = time.time()

    record_transitions(prev, state)

    HISTORY.append({"temp": state["temp"], "humidity": state["humidity"],
                    "soil": soil_pct(state["soil_raw"])})
    del HISTORY[:-HISTORY_LIMIT]
    return jsonify({"ok": True})


@app.route("/state")
def get_state():
    age = time.time() - state["last_update_ts"] if state["last_update_ts"] else None
    return jsonify({
        **state,
        "soil_pct": soil_pct(state["soil_raw"]),
        "soil_threshold_pct": soil_pct(SOIL_THRESHOLD_RAW),
        "age": age,
        "online": age is not None and age < 10,
        "plant_profile": PLANT,
        "trend": {k: [p[k] for p in HISTORY if p[k] is not None]
                  for k in ("temp", "humidity", "soil")},
        "events": EVENTS[:25],
        "model": detector.status(),
    })


@app.route("/detect", methods=["POST"])
def detect():
    result = detector.classify(request.get_data())
    if result["verdict"] in ("disease", "healthy"):
        best = result["top"][0]
        camera["last"] = {**best, "t": time.strftime("%H:%M:%S")}
        # log each new confident finding, and repeat a disease at most every 30 s
        if best["label"] != camera["logged_label"] or (
                result["verdict"] == "disease" and time.time() - camera["logged_ts"] > 30):
            pct = round(best["p"] * 100)
            add_event("detect", f"Camera: {best['crop']} — {best['condition']} ({pct}%)")
            camera["logged_label"], camera["logged_ts"] = best["label"], time.time()
    return jsonify(result)


def fallback_answer(question):
    """Rule-based answer from live data — always works, no internet or Ollama needed."""
    q = question.lower()
    s = state
    t, h, sp = s["temp"], s["humidity"], soil_pct(s["soil_raw"])

    if any(w in q for w in ("camera", "detect", "leaf", "photo", "see")):
        last = camera["last"]
        if not last:
            return "The camera hasn't made a confident detection yet. Hold a single leaf close inside the square."
        return (f"At {last['t']} the camera saw {last['crop']} — {last['condition']} "
                f"({round(last['p'] * 100)}% confident). The model covers 38 crop classes and is trained on lab photos, "
                "so treat it as a check, not a diagnosis.")

    if s["temp"] is None:
        return "No readings yet — the ESP32 hasn't sent any data."

    if any(w in q for w in ("water", "pump", "irrigat", "soil", "dry")):
        st = s["soil_state"]
        if st == "watering":
            return f"Watering now: soil moisture is about {sp}%, below the dry threshold, so it's giving a short 3-second pulse."
        if st == "soaking":
            return f"Soil is still reading dry ({sp}%), but it just watered — it waits 60 s for water to soak in before checking again, to avoid root rot."
        if st == "rain-skip":
            return "Soil is dry, but rain is detected, so watering is skipped to avoid waterlogging the roots."
        return f"Soil moisture is about {sp}%, above the dry threshold — no watering needed."

    if any(w in q for w in ("risk", "disease", "spot", "sick", "fung", "bacteri")):
        return (f"Leaf-spot risk is {s['risk']}. For a Syngonium it rises when it stays {PLANT['disease_band']}; "
                f"right now it's {t} °C and {h}% humidity.")

    if "fan" in q:
        if s["fan"]:
            return f"The fan is on because of {FAN_REASON.get(s['fan_reason'], 'the climate conditions')} — moving air cools the plant and dries leaf surfaces."
        return f"The fan is off: {t} °C and {h}% humidity don't need extra airflow."

    if any(w in q for w in ("shade", "light", "sun", "servo")):
        if s["shade"]:
            why = "the light is bright" if s["bright"] else "it's above 30 °C"
            return f"The shade is deployed because {why}. Variegated Syngonium leaves scorch in direct sun."
        return "The shade is retracted — light and temperature are comfortable."

    if any(w in q for w in ("tank", "reservoir", "refill")):
        if s["tank_low"]:
            return "The reservoir is low — refill it soon or watering will run dry."
        return f"The reservoir is fine ({s['tank_cm']} cm from the sensor to the water)."

    if "rain" in q:
        return "Rain is detected." if s["rain"] else "No rain right now."

    if any(w in q for w in ("motion", "animal", "pest", "bird", "buzz")):
        return "Motion was just detected and the deterrent sounded." if s["motion"] else "No motion near the plant right now."

    if any(w in q for w in ("temp", "hot", "cold", "heat", "humid")):
        return f"It's {t} °C and {h}% humidity. A Syngonium is comfortable at {PLANT['comfort']} and stressed above {PLANT['heat']} °C."

    return (f"{t} °C, {h}% humidity, soil {sp}%, leaf-spot risk {s['risk']}. "
            "Ask about watering, disease risk, the fan, the shade, the reservoir, rain, or motion.")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Ask me something about the plant or the system."})

    prompt = (
        "You are PhytoCast, a smart-farm assistant monitoring a potted Syngonium podophyllum "
        f"(comfortable {PLANT['comfort']}, heat stress above {PLANT['heat']} °C, leaf-spot risk "
        f"when {PLANT['disease_band']}). Answer in one or two plain sentences using only this "
        f"live data:\n{ {k: state[k] for k in FIELDS} }\nQuestion: {question}\nAnswer:"
    )
    try:
        r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=6)
        r.raise_for_status()
        answer = r.json().get("response", "").strip()
        if answer:
            return jsonify({"answer": answer, "source": "ollama"})
    except Exception:
        pass
    return jsonify({"answer": fallback_answer(question), "source": "rules"})


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PhytoCast</title>
<style>
  :root{
    --bg:#0a0f0c; --surface:#111814; --surface-2:#161f19; --line:#1e2a22; --line-2:#2a392f;
    --ink:#e9efe9; --ink-2:#9aab9e; --muted:#66786b;
    --accent:#57b078; --water:#3987e5;
    --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
    --mono:"Cascadia Mono","SF Mono",ui-monospace,Consolas,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--ink)}
  body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;line-height:1.45;
       -webkit-font-smoothing:antialiased;padding:24px 20px 48px}
  .wrap{max-width:1180px;margin:0 auto;display:grid;gap:14px}

  /* header */
  header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;
         padding-bottom:16px;border-bottom:1px solid var(--line);margin-bottom:6px}
  .brand{display:flex;align-items:center;gap:12px}
  .brand h1{margin:0;font-size:17px;font-weight:650;letter-spacing:-.01em}
  .plant{font-size:12.5px;color:var(--ink-2)}
  .plant i{color:var(--muted)}
  .hdr-right{display:flex;align-items:center;gap:14px}
  .clock{font-family:var(--mono);font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
  .pill{display:inline-flex;align-items:center;gap:8px;font-size:12px;color:var(--ink-2);
        border:1px solid var(--line-2);border-radius:999px;padding:5px 12px 5px 10px}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--muted);flex:none}
  .pill.live{color:var(--ink)} .pill.live .dot{background:var(--accent);animation:beat 2s infinite}
  .pill.stale .dot{background:var(--warning)}
  @keyframes beat{0%{box-shadow:0 0 0 0 rgba(87,176,120,.5)}70%{box-shadow:0 0 0 6px rgba(87,176,120,0)}100%{box-shadow:0 0 0 0 rgba(87,176,120,0)}}

  .card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px}
  .eyebrow{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}

  /* top row */
  .top{display:grid;grid-template-columns:1fr 1.7fr;gap:14px}
  .risk-level{display:flex;align-items:center;gap:12px;margin:10px 0 6px}
  .risk-level .glyph{width:38px;height:38px;border-radius:10px;display:grid;place-items:center;flex:none}
  .risk-level .word{font-size:30px;font-weight:700;letter-spacing:-.02em}
  .risk-desc{color:var(--ink-2);font-size:13px;min-height:38px}
  .scale{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin:14px 0 12px}
  .scale span{height:4px;border-radius:2px;background:var(--line-2)}
  .profile{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:12px}
  .lvl-LOW .glyph{background:rgba(12,163,12,.14);color:var(--good)}
  .lvl-MEDIUM .glyph{background:rgba(250,178,25,.14);color:var(--warning)}
  .lvl-HIGH .glyph{background:rgba(208,59,59,.16);color:var(--critical)}
  .lvl-LOW .scale span:nth-child(1){background:var(--good)}
  .lvl-MEDIUM .scale span:nth-child(-n+2){background:var(--warning)}
  .lvl-HIGH .scale span{background:var(--critical)}

  .card.col{display:flex;flex-direction:column}
  .acts{display:grid;grid-template-columns:repeat(4,1fr);margin-top:12px;border:1px solid var(--line);border-radius:10px;overflow:hidden;flex:1}
  .thread .hint{color:var(--muted);font-size:13px;margin:auto 0}
  .act{padding:14px;border-right:1px solid var(--line);min-height:112px;display:flex;flex-direction:column;gap:6px}
  .act:last-child{border-right:none}
  .act .name{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink-2)}
  .act .name svg{color:var(--muted)}
  .act .st{font-size:16px;font-weight:650;display:flex;align-items:center;gap:8px}
  .act .why{font-size:12px;color:var(--muted)}
  .act.on .name svg{color:var(--accent)} .act.on .st .dot{background:var(--accent)}
  .act.water.on .name svg{color:var(--water)} .act.water.on .st .dot{background:var(--water)}
  .act.alert.on .name svg{color:var(--warning)} .act.alert.on .st .dot{background:var(--warning)}

  /* metrics */
  .metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
  .m-head{display:flex;justify-content:space-between;align-items:baseline}
  .val{display:flex;align-items:baseline;gap:4px;margin-top:8px}
  .val b{font-size:28px;font-weight:650;letter-spacing:-.02em}
  .val small{color:var(--ink-2);font-size:13px}
  .foot{margin-top:8px;font-size:12px;color:var(--muted);display:flex;justify-content:space-between;gap:8px}
  .tag{font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;border:1px solid var(--line-2);color:var(--ink-2)}
  .tag.warn{border-color:rgba(250,178,25,.45);color:var(--warning)}
  .tag.ok{border-color:rgba(87,176,120,.4);color:var(--accent)}
  svg.spark{display:block;width:100%;height:42px;margin-top:10px;overflow:visible;cursor:crosshair}
  .meter{position:relative;height:6px;border-radius:3px;background:#12263d;margin-top:18px}
  .meter .fill{height:100%;border-radius:3px;background:var(--water);transition:width .6s ease}
  .meter .tick{position:absolute;top:-4px;width:2px;height:14px;background:var(--ink-2);border-radius:1px}
  .state-big{font-size:20px;font-weight:650;margin-top:10px;display:flex;align-items:center;gap:10px}

  /* bottom */
  .bottom{display:grid;grid-template-columns:1.35fr 1fr;gap:14px}
  .log{list-style:none;margin:12px 0 0;padding:0;max-height:330px;overflow-y:auto}
  .log li{display:grid;grid-template-columns:64px 3px 1fr;gap:12px;align-items:center;
          padding:9px 0;border-top:1px solid var(--line);font-size:13px}
  .log li:first-child{border-top:none}
  .log time{font-family:var(--mono);font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
  .log .bar{height:16px;border-radius:2px;background:var(--line-2)}
  .log .water{background:var(--water)} .log .climate{background:var(--accent)}
  .log .alert{background:var(--warning)} .log .system{background:var(--ink-2)}
  .log .detect{background:#9085e9}

  /* live detection */
  .cam{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:22px;margin-top:12px}
  .stage{position:relative;aspect-ratio:4/3;background:#070b09;border:1px solid var(--line);border-radius:10px;overflow:hidden}
  .stage video{width:100%;height:100%;object-fit:cover;display:block}
  .stage .guide{position:absolute;inset:0;margin:auto;height:72%;aspect-ratio:1;border:1.5px solid rgba(233,239,233,.55);
                border-radius:10px;box-shadow:0 0 0 999px rgba(7,11,9,.35);pointer-events:none;display:none}
  .stage.on .guide{display:block}
  .stage .idle{position:absolute;inset:0;display:grid;place-items:center;text-align:center;color:var(--muted);font-size:13px;padding:24px}
  .stage.on .idle{display:none}
  .stage .badge{position:absolute;left:12px;top:12px;display:none;align-items:center;gap:7px;font-size:11px;font-weight:600;
                letter-spacing:.06em;text-transform:uppercase;background:rgba(7,11,9,.75);border:1px solid var(--line-2);
                border-radius:999px;padding:4px 10px 4px 8px}
  .stage.on .badge{display:inline-flex}
  .stage .badge .dot{background:var(--critical);animation:blink 1.4s infinite}
  @keyframes blink{50%{opacity:.25}}
  .result .verdict{font-size:22px;font-weight:700;letter-spacing:-.01em;margin:8px 0 2px;min-height:30px}
  .result .crop{color:var(--ink-2);font-size:13px;min-height:19px}
  .preds{list-style:none;margin:18px 0 0;padding:0;display:grid;gap:12px}
  .preds li{display:grid;grid-template-columns:1fr auto;gap:4px 12px;font-size:12.5px}
  .preds .track{grid-column:1 / -1;height:5px;border-radius:3px;background:var(--line-2);overflow:hidden}
  .preds .track i{display:block;height:100%;background:var(--ink-2);border-radius:3px;transition:width .4s ease}
  .preds li:first-child .track i{background:var(--accent)}
  .preds .pct{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink-2)}
  .cam-controls{display:flex;gap:8px;margin-top:18px;flex-wrap:wrap}
  .btn{font:inherit;font-weight:650;font-size:13px;border-radius:8px;padding:9px 16px;cursor:pointer;border:1px solid var(--line-2);
       background:transparent;color:var(--ink)}
  .btn.primary{background:var(--accent);border-color:var(--accent);color:#07120b}
  select.btn{font-weight:500;color:var(--ink-2);background:var(--bg);max-width:220px}
  .cam-note{font-size:12px;color:var(--muted);margin-top:14px;line-height:1.5}
  @media (max-width:920px){ .cam{grid-template-columns:1fr} }
  .empty{color:var(--muted);font-size:13px;padding:24px 0}

  .thread{display:flex;flex-direction:column;gap:8px;margin:12px 0;max-height:240px;min-height:120px;overflow-y:auto}
  .msg{max-width:88%;padding:9px 12px;border-radius:10px;font-size:13px}
  .msg.q{align-self:flex-end;background:var(--accent);color:#07120b;font-weight:550}
  .msg.a{align-self:flex-start;background:var(--surface-2);border:1px solid var(--line)}
  .msg.pending{color:var(--muted)}
  .chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
  .chip{font:inherit;font-size:12px;color:var(--ink-2);background:transparent;border:1px solid var(--line-2);
        border-radius:999px;padding:5px 11px;cursor:pointer}
  .chip:hover{color:var(--ink);border-color:var(--muted)}
  .askbar{display:flex;gap:8px}
  .askbar input{flex:1;background:var(--bg);color:var(--ink);border:1px solid var(--line-2);border-radius:8px;
                padding:10px 12px;font:inherit}
  .askbar button{background:var(--accent);color:#07120b;border:none;border-radius:8px;padding:0 16px;font:inherit;font-weight:650;cursor:pointer}
  input:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

  #tip{position:fixed;pointer-events:none;background:#0d1410;border:1px solid var(--line-2);border-radius:6px;
       padding:4px 8px;font-family:var(--mono);font-size:11.5px;color:var(--ink);opacity:0;transition:opacity .1s;z-index:10}
  footer{font-size:11.5px;color:var(--muted);display:flex;justify-content:space-between;padding-top:4px}

  @media (max-width:920px){
    .top,.bottom{grid-template-columns:1fr} .metrics{grid-template-columns:repeat(2,1fr)}
    .acts{grid-template-columns:repeat(2,1fr)} .act:nth-child(2){border-right:none}
    .act:nth-child(-n+2){border-bottom:1px solid var(--line)}
  }
  @media (max-width:560px){ .metrics{grid-template-columns:1fr} body{padding:16px} }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <svg width="30" height="30" viewBox="0 0 30 30" fill="none" aria-hidden="true">
        <rect x=".5" y=".5" width="29" height="29" rx="8" stroke="#2a392f"/>
        <path d="M15 23c-4.6-1.9-6.4-5.6-5.4-11 4.8-.9 8.6.9 9.5 5.6.6 2.8-.4 4.6-4.1 5.4z" fill="#57b078"/>
        <path d="M10 12.5c3.6-.6 6.4 1 8 4.6" stroke="#0a0f0c" stroke-width="1.1" fill="none"/>
      </svg>
      <div>
        <h1>PhytoCast</h1>
        <div class="plant" id="plant">Syngonium <i>· Syngonium podophyllum</i></div>
      </div>
    </div>
    <div class="hdr-right">
      <span class="clock" id="clock">--:--:--</span>
      <span class="pill" id="conn"><span class="dot"></span><span id="connText">Waiting for ESP32</span></span>
    </div>
  </header>

  <section class="top">
    <div class="card lvl-LOW" id="riskCard">
      <div class="eyebrow">Leaf-spot risk</div>
      <div class="risk-level">
        <div class="glyph" id="riskGlyph"></div>
        <div class="word" id="riskWord">LOW</div>
      </div>
      <div class="risk-desc" id="riskDesc">Waiting for readings.</div>
      <div class="scale"><span></span><span></span><span></span></div>
      <div class="profile" id="profile">Profile · comfort 18–27 °C · risk when 20–30 °C at ≥85% RH</div>
    </div>

    <div class="card col">
      <div class="eyebrow">What the system is doing</div>
      <div class="acts">
        <div class="act water" id="actPump">
          <div class="name"><svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 1.5c2.4 3 3.8 5 3.8 7a3.8 3.8 0 01-7.6 0c0-2 1.4-4 3.8-7z" stroke="currentColor" stroke-width="1.3"/></svg>Irrigation</div>
          <div class="st"><span class="dot"></span><span>--</span></div>
          <div class="why">--</div>
        </div>
        <div class="act" id="actShade">
          <div class="name"><svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M1.5 7.5a5.5 5.5 0 0111 0h-11z" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/><path d="M7 7.5v5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>Shade</div>
          <div class="st"><span class="dot"></span><span>--</span></div>
          <div class="why">--</div>
        </div>
        <div class="act" id="actFan">
          <div class="name"><svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="1.3" stroke="currentColor" stroke-width="1.2"/><path d="M7 5.7C6 3 7.2 1.5 8.6 1.9c1.2.4 1 2.4-1.6 3.8M8.3 7.4c2.8.4 3.6 2.1 2.8 3.2-.8 1-2.6.2-3.1-2.8M5.8 7.6c-1.8 2.2-3.6 2-4 .7-.4-1.2 1.4-2.3 4-1.3" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/></svg>Fan</div>
          <div class="st"><span class="dot"></span><span>--</span></div>
          <div class="why">--</div>
        </div>
        <div class="act alert" id="actDeter">
          <div class="name"><svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 5.2h2.2L7.5 2.5v9L4.2 8.8H2z" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/><path d="M9.8 5a2.8 2.8 0 010 4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>Deterrent</div>
          <div class="st"><span class="dot"></span><span>--</span></div>
          <div class="why">--</div>
        </div>
      </div>
    </div>
  </section>

  <section class="card">
    <div class="m-head"><span class="eyebrow">Live detection</span><span class="tag" id="modelTag">Model loading</span></div>
    <div class="cam">
      <div class="stage" id="stage">
        <video id="video" autoplay playsinline muted></video>
        <div class="guide"></div>
        <span class="badge"><span class="dot"></span>Live</span>
        <div class="idle">Camera off. Start it, then hold a single leaf inside the square.</div>
      </div>
      <div class="result">
        <div class="eyebrow">Diagnosis</div>
        <div class="verdict" id="verdict">—</div>
        <div class="crop" id="verdictSub">Waiting for camera</div>
        <ul class="preds" id="preds"></ul>
        <div class="cam-controls">
          <button class="btn primary" id="camBtn">Start camera</button>
          <select class="btn" id="camSel" aria-label="Camera"></select>
        </div>
        <div class="cam-note" id="camNote">MobileNetV2 trained on lab photos of 38 crop classes. Works best on one leaf, close up, against a plain background. Frames with too little leaf in them are skipped rather than guessed.</div>
      </div>
    </div>
  </section>

  <section class="metrics">
    <div class="card">
      <div class="m-head"><span class="eyebrow">Temperature</span><span class="tag" id="tempTag">--</span></div>
      <div class="val"><b id="temp">--</b><small>°C</small></div>
      <svg class="spark" id="sparkTemp" viewBox="0 0 300 42" preserveAspectRatio="none"></svg>
      <div class="foot"><span>Comfort 18–27 °C</span><span>line: 30 °C heat limit</span></div>
    </div>
    <div class="card">
      <div class="m-head"><span class="eyebrow">Humidity</span><span class="tag" id="humTag">--</span></div>
      <div class="val"><b id="hum">--</b><small>%</small></div>
      <svg class="spark" id="sparkHum" viewBox="0 0 300 42" preserveAspectRatio="none"></svg>
      <div class="foot"><span>Leaf-spot risk ≥85%</span><span>line: 85%</span></div>
    </div>
    <div class="card">
      <div class="m-head"><span class="eyebrow">Soil moisture</span><span class="tag" id="soilTag">--</span></div>
      <div class="val"><b id="soil">--</b><small>%</small></div>
      <svg class="spark" id="sparkSoil" viewBox="0 0 300 42" preserveAspectRatio="none"></svg>
      <div class="foot"><span id="soilRaw">raw --</span><span id="soilThr">waters below --%</span></div>
    </div>
    <div class="card">
      <div class="m-head"><span class="eyebrow">Reservoir</span><span class="tag" id="tankTag">--</span></div>
      <div class="val"><b id="tank">--</b><small>cm to water</small></div>
      <div class="foot" style="margin-top:14px"><span>Distance from sensor to water surface — grows as it empties</span></div>
    </div>
    <div class="card">
      <div class="eyebrow">Rain</div>
      <div class="state-big" id="rain"><span class="dot"></span><span>--</span></div>
      <div class="foot"><span>Rain pauses irrigation to prevent waterlogging</span></div>
    </div>
    <div class="card">
      <div class="eyebrow">Motion</div>
      <div class="state-big" id="motion"><span class="dot"></span><span>--</span></div>
      <div class="foot"><span>Triggers the buzzer deterrent</span></div>
    </div>
  </section>

  <section class="bottom">
    <div class="card">
      <div class="eyebrow">Activity</div>
      <ul class="log" id="log"></ul>
    </div>
    <div class="card">
      <div class="eyebrow">Ask PhytoCast</div>
      <div class="thread" id="thread"><div class="hint" id="threadHint">Answers are built from the live readings above — try a suggestion below.</div></div>
      <div class="chips">
        <button class="chip">Why did it water?</button>
        <button class="chip">Is the plant at risk?</button>
        <button class="chip">Why is the fan on?</button>
        <button class="chip">Why is the shade out?</button>
      </div>
      <div class="askbar"><input id="q" placeholder="Ask about the plant or the system" autocomplete="off"><button id="askBtn">Ask</button></div>
    </div>
  </section>

  <footer><span>ESP32 → Flask · updates every ~3 s</span><span id="lastUpd">No data yet</span></footer>
</div>
<div id="tip"></div>

<script>
const $ = id => document.getElementById(id);
const RISK_DESC = {
  LOW: "Conditions don't favour leaf spot. Nothing to act on.",
  MEDIUM: "Warm, very humid air has started to build up. Airflow is being watched.",
  HIGH: "Warm, humid, still air has persisted — prime conditions for bacterial and fungal leaf spot. Fan running."
};
const FAN_WHY = {heat: "Above 30 °C", humid: "Humid, still air", risk: "High leaf-spot risk"};
const GLYPH = {
  ok: '<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M4 9.4l3.2 3.1L14 5.6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  warn: '<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M9 2.2l6.8 12.1H2.2L9 2.2z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M9 7.2v3" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="9" cy="12.4" r=".9" fill="currentColor"/></svg>'
};

function setAct(id, on, text, why){
  const el = $(id);
  el.classList.toggle("on", on);
  el.querySelector(".st span:last-child").textContent = text;
  el.querySelector(".why").textContent = why;
}
function setState(id, on, text){
  const el = $(id);
  el.querySelector(".dot").style.background = on ? "var(--accent)" : "var(--muted)";
  el.querySelector("span:last-child").textContent = text;
}
function tag(id, text, cls){ const el = $(id); el.textContent = text; el.className = "tag " + (cls || ""); }

function spark(svg, vals, lo, hi, ref, unit){
  svg._vals = vals; svg._unit = unit;
  const W = 300, H = 42, pad = 4;
  const y = v => H - pad - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * (H - 2 * pad);
  let out = `<line x1="0" x2="${W}" y1="${y(ref)}" y2="${y(ref)}" stroke="#2a392f" stroke-width="1"/>`;
  if (vals.length > 1){
    const step = W / (vals.length - 1);
    const d = vals.map((v, i) => (i ? "L" : "M") + (i * step).toFixed(1) + "," + y(v).toFixed(1)).join(" ");
    out += `<path d="${d}" fill="none" stroke="#9aab9e" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>`;
    out += `<circle cx="${W}" cy="${y(vals[vals.length-1])}" r="3.5" fill="#57b078" stroke="#111814" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
  }
  svg.innerHTML = out;
}
document.querySelectorAll("svg.spark").forEach(svg => {
  const tip = $("tip");
  svg.addEventListener("mousemove", e => {
    const vals = svg._vals || []; if (!vals.length) return;
    const r = svg.getBoundingClientRect();
    const i = Math.round((e.clientX - r.left) / r.width * (vals.length - 1));
    const v = vals[Math.max(0, Math.min(vals.length - 1, i))];
    tip.textContent = v + svg._unit + "  ·  " + (vals.length - 1 - i) * 3 + "s ago";
    tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY - 28) + "px"; tip.style.opacity = 1;
  });
  svg.addEventListener("mouseleave", () => $("tip").style.opacity = 0);
});

function fmt(v, d){ return (v === null || v === undefined) ? "--" : Number(v).toFixed(d); }

async function refresh(){
  let s;
  try { s = await (await fetch("/state")).json(); } catch(e){ return; }

  modelTag(s.model);
  const pill = $("conn");
  pill.className = "pill " + (s.online ? "live" : s.age !== null ? "stale" : "");
  $("connText").textContent = s.online ? "Live" : s.age !== null ? "Offline · last seen " + Math.round(s.age) + "s ago" : "Waiting for ESP32";
  $("lastUpd").textContent = s.last_update ? "Last update " + s.last_update : "No data yet";

  const lvl = s.risk || "LOW";
  $("riskCard").className = "card lvl-" + lvl;
  $("riskWord").textContent = lvl;
  const g = lvl === "LOW" ? "ok" : "warn";
  if ($("riskGlyph")._g !== g){ $("riskGlyph").innerHTML = GLYPH[g]; $("riskGlyph")._g = g; }
  $("riskDesc").textContent = s.temp === null ? "Waiting for readings." : RISK_DESC[lvl];
  const p = s.plant_profile;
  $("plant").innerHTML = p.name + " <i>· " + p.latin + "</i>";
  $("profile").textContent = "Profile · comfort " + p.comfort + " · risk when " + p.disease_band;

  const st = s.soil_state;
  setAct("actPump", !!s.pumping,
    s.pumping ? "Watering" : st === "soaking" ? "Soaking in" : st === "rain-skip" ? "Paused" : "Idle",
    s.pumping ? "3-second pulse" : st === "soaking" ? "Waits 60 s before re-checking" : st === "rain-skip" ? "Raining — skipped" : "Soil moist enough");
  setAct("actShade", !!s.shade, s.shade ? "Deployed" : "Retracted",
    s.shade ? (s.bright ? "Bright light" : "Above 30 °C") : "Light is comfortable");
  setAct("actFan", !!s.fan, s.fan ? "On" : "Off", s.fan ? (FAN_WHY[s.fan_reason] || "Climate") : "Airflow not needed");
  setAct("actDeter", !!s.motion, s.motion ? "Sounding" : "Armed", s.motion ? "Motion near plant" : "Watching for motion");

  $("temp").textContent = fmt(s.temp, 1);
  $("hum").textContent = fmt(s.humidity, 0);
  $("soil").textContent = fmt(s.soil_pct, 0);
  $("soilRaw").textContent = "raw " + (s.soil_raw ?? "--");
  $("soilThr").textContent = "waters below " + s.soil_threshold_pct + "%";

  if (s.temp !== null){
    tag("tempTag", s.temp > 30 ? "Heat stress" : s.temp < 15 ? "Too cold" : s.temp > 27 ? "Warm" : "Comfortable",
        s.temp > 30 || s.temp < 15 ? "warn" : s.temp > 27 ? "" : "ok");
    tag("humTag", s.humidity >= 85 ? "Leaf-spot range" : s.humidity >= 80 ? "High" : "Normal",
        s.humidity >= 85 ? "warn" : s.humidity >= 80 ? "" : "ok");
  }
  if (s.soil_pct !== null) tag("soilTag", st === "ok" ? "Moist" : "Dry", st === "ok" ? "ok" : "warn");

  $("tank").textContent = (s.tank_cm === null || s.tank_cm < 0) ? "--" : s.tank_cm;
  if (s.tank_cm !== null) tag("tankTag", s.tank_low ? "Refill soon" : "OK", s.tank_low ? "warn" : "ok");
  setState("rain", !!s.rain, s.rain ? "Raining" : "Dry");
  setState("motion", !!s.motion, s.motion ? "Detected" : "Clear");

  spark($("sparkTemp"), s.trend.temp, 20, 36, 30, " °C");
  spark($("sparkHum"), s.trend.humidity, 50, 100, 85, "%");
  spark($("sparkSoil"), s.trend.soil, 0, 100, s.soil_threshold_pct, "%");

  const log = $("log");
  const key = s.events.map(e => e.t + e.text).join("|");
  if (log._key !== key){
    log._key = key;
    log.innerHTML = s.events.length
      ? s.events.map(e => `<li><time>${e.t}</time><span class="bar ${e.kind}"></span><span></span></li>`).join("")
      : `<li class="empty" style="display:block;border:none">Nothing yet — every action the system takes will appear here.</li>`;
    log.querySelectorAll("li span:last-child").forEach((el, i) => el.textContent = s.events[i].text);
  }
}

function addMsg(text, cls){
  const hint = $("threadHint"); if (hint) hint.remove();
  const t = $("thread"), d = document.createElement("div");
  d.className = "msg " + cls; d.textContent = text; t.appendChild(d); t.scrollTop = t.scrollHeight; return d;
}
async function ask(q){
  q = (q ?? $("q").value).trim(); if (!q) return;
  addMsg(q, "q"); $("q").value = "";
  const pending = addMsg("Thinking…", "a pending");
  try {
    const r = await fetch("/ask", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({question: q})});
    pending.textContent = (await r.json()).answer;
  } catch(e){ pending.textContent = "Couldn't reach the server."; }
  pending.classList.remove("pending");
}
$("askBtn").addEventListener("click", () => ask());
$("q").addEventListener("keydown", e => { if (e.key === "Enter") ask(); });
document.querySelectorAll(".chip").forEach(c => c.addEventListener("click", () => ask(c.textContent)));

/* ---------- live detection ---------- */
let stream = null, camTimer = null, busy = false;
const grab = document.createElement("canvas");
grab.width = grab.height = 224;

function showResult(r){
  const v = $("verdict"), sub = $("verdictSub"), list = $("preds");
  if (r.verdict === "no_leaf"){
    v.textContent = "No leaf in view"; v.style.color = "var(--ink-2)";
    sub.textContent = "Hold a single leaf close, inside the square"; list.innerHTML = ""; return;
  }
  if (r.verdict === "unavailable"){
    v.textContent = "Model unavailable"; v.style.color = "var(--ink-2)"; sub.textContent = r.status; return;
  }
  const b = r.top[0];
  v.textContent = r.verdict === "unsure" ? "Not sure" : b.condition;
  v.style.color = r.verdict === "disease" ? "var(--warning)" : r.verdict === "healthy" ? "var(--accent)" : "var(--ink-2)";
  sub.textContent = r.verdict === "unsure"
    ? "Below 70% confidence — try a closer, clearer leaf"
    : b.crop + " · " + Math.round(b.p * 100) + "% confident";
  list.innerHTML = r.top.map(() => `<li><span></span><span class="pct"></span><span class="track"><i></i></span></li>`).join("");
  list.querySelectorAll("li").forEach((li, i) => {
    const p = r.top[i];
    li.children[0].textContent = p.crop + " — " + p.condition;
    li.children[1].textContent = (p.p * 100).toFixed(1) + "%";
    li.querySelector("i").style.width = (p.p * 100).toFixed(1) + "%";
  });
}

async function sendFrame(){
  const video = $("video");
  if (!stream || busy || !video.videoWidth) return;
  busy = true;
  const s = Math.min(video.videoWidth, video.videoHeight) * 0.72;   // matches the guide square
  const sx = (video.videoWidth - s) / 2, sy = (video.videoHeight - s) / 2;
  grab.getContext("2d").drawImage(video, sx, sy, s, s, 0, 0, 224, 224);
  const blob = await new Promise(res => grab.toBlob(res, "image/jpeg", 0.9));
  try {
    const r = await fetch("/detect", {method: "POST", headers: {"Content-Type": "image/jpeg"}, body: blob});
    showResult(await r.json());
  } catch(e){}
  busy = false;
}

async function listCams(){
  const sel = $("camSel"), current = sel.value;
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
  sel.innerHTML = "";
  cams.forEach((c, i) => { const o = document.createElement("option"); o.value = c.deviceId; o.textContent = c.label || "Camera " + (i + 1); sel.appendChild(o); });
  if (current) sel.value = current;
  sel.style.display = cams.length > 1 ? "" : "none";
}

async function startCam(){
  if (!navigator.mediaDevices){ $("camNote").textContent = "Camera needs the page opened at http://localhost:5000 on this laptop."; return; }
  const id = $("camSel").value;
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: id ? {deviceId: {exact: id}} : {width: {ideal: 1280}}});
  } catch(e){ $("camNote").textContent = "Couldn't open the camera: " + e.message; return; }
  $("video").srcObject = stream;
  $("stage").classList.add("on");
  $("camBtn").textContent = "Stop camera"; $("camBtn").classList.remove("primary");
  $("verdictSub").textContent = "Looking…";
  await listCams();
  camTimer = setInterval(sendFrame, 900);
}

function stopCam(){
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; clearInterval(camTimer);
  $("stage").classList.remove("on");
  $("camBtn").textContent = "Start camera"; $("camBtn").classList.add("primary");
  $("verdict").textContent = "—"; $("verdict").style.color = ""; $("verdictSub").textContent = "Camera off"; $("preds").innerHTML = "";
}

$("camBtn").addEventListener("click", () => stream ? stopCam() : startCam());
$("camSel").addEventListener("change", () => { if (stream){ stopCam(); startCam(); } });
if (navigator.mediaDevices) listCams();

function modelTag(m){
  if (m.status === "ready") tag("modelTag", "Model ready · " + m.info, "ok");
  else if (m.status === "loading") tag("modelTag", "Model loading", "");
  else tag("modelTag", "Model unavailable", "warn");
}

setInterval(() => $("clock").textContent = new Date().toLocaleTimeString([], {hour12: false}), 1000);
refresh(); setInterval(refresh, 2000);
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return Response(PAGE, mimetype="text/html")


if __name__ == "__main__":
    print("PhytoCast dashboard -> http://localhost:5000")
    print("ESP32 should POST to http://<this-laptop-IP>:5000/update")
    app.run(host="0.0.0.0", port=5000, debug=False)
