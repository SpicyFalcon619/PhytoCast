"""
Simulated ESP32 for testing the dashboard without hardware. Sends the same
JSON the real firmware sends, every 3 seconds, and runs the same decision
logic (Syngonium profile). Run alongside app.py:

    python send_test_data.py
"""

import random
import time
import requests

URL = "http://localhost:5000/update"
SOIL_DRY = 1950

soil = 1500.0
tank = 6.0
temp = 28.0
hum = 72.0
fav = 0
pumping_left = 0
soak_left = 0
bright = False

print("Sending simulated readings to the dashboard. Ctrl+C to stop.")

while True:
    temp = max(22, min(34, temp + random.uniform(-0.4, 0.45)))
    hum = max(55, min(95, hum + random.uniform(-2.0, 2.2)))
    if random.random() < 0.08:
        bright = not bright
    rain = random.random() < 0.05
    motion = random.random() < 0.08

    # soil dries slowly; a watering pulse wets it
    if pumping_left > 0:
        soil -= random.uniform(120, 180)
        pumping_left -= 1
        if pumping_left == 0:
            soak_left = 20                      # 60 s at 3 s per tick
    else:
        soil += random.uniform(4, 12)
    soil = max(1150, min(2250, soil))
    soak_left = max(0, soak_left - 1)

    soil_dry = soil > SOIL_DRY
    if pumping_left == 0 and soil_dry and not rain and soak_left == 0:
        pumping_left = 1
    pumping = pumping_left > 0
    soil_state = "watering" if pumping else ("rain-skip" if rain else "soaking") if soil_dry else "ok"

    tank = min(22, tank + random.uniform(0, 0.08) + (0.6 if pumping else 0))
    if tank > 18 and random.random() < 0.1:
        tank = 5.0                               # someone refilled it

    favourable = 20 <= temp <= 30 and hum >= 85
    fav = fav + 1 if favourable else max(0, fav - 1)
    risk = "LOW" if fav == 0 else "MEDIUM" if fav < 5 else "HIGH"

    too_hot = temp > 30
    humid_fan = hum >= 80 and temp >= 20
    fan = too_hot or humid_fan or risk == "HIGH"
    fan_reason = "heat" if too_hot else "humid" if humid_fan else "risk" if fan else ""

    payload = {
        "plant": "Syngonium", "temp": round(temp, 1), "humidity": round(hum),
        "risk": risk, "soil_raw": int(soil), "soil_state": soil_state, "pumping": pumping,
        "shade": bright or too_hot, "bright": bright, "fan": fan, "fan_reason": fan_reason,
        "tank_cm": round(tank), "tank_low": tank > 15, "rain": rain, "motion": motion,
    }
    try:
        requests.post(URL, json=payload, timeout=2)
        print("sent:", payload)
    except Exception as e:
        print("Can't reach the dashboard — is app.py running?", e)
    time.sleep(3)
