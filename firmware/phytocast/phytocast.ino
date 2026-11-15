/*
  PhytoCast — Combined Sketch, tuned for Syngonium podophyllum (arrowhead plant)
  ------------------------------------------------------------------------------
  All thresholds live in the PLANT PROFILE block below. Swapping to another
  plant means editing only that block -- the logic stays the same.

  Why these numbers (Syngonium care basics):
    - Comfortable at ~18-27 C; heat stress above ~30 C; cold stress below ~15 C.
    - Wants bright INDIRECT light. Direct sun scorches leaves, and variegated
      cultivars (like this one) burn more easily -> shade on bright light.
    - Most common killer is root rot from soggy soil -> water in short pulses,
      let it soak in before re-checking, and never water while it's raining.
    - Bacterial / fungal leaf spot is favoured by warm (~20-30 C), very humid,
      still air -> airflow (fan) helps; sustained conditions raise risk.

  LED left out -- still needs a current-limiting resistor.

  Power: DHT22/OLED/ultrasonic/rain/soil/LDR -> 3V3 rail.
  PIR/relay coil/servo -> VIN rail. Pump + fan -> relay channels, powered by
  the external 5V adapter.
*/

#include "DHT.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <Wire.h>

// ================= DASHBOARD (M6) =================
// ESP32 only joins 2.4 GHz WiFi. Laptop must be on the same network.
// Laptop IP: run `ipconfig` on the laptop, use its IPv4 address.
#include "secrets.h"   // WIFI_SSID, WIFI_PASS, DASH_URL (copy secrets.example.h)
const unsigned long POST_EVERY_MS = 3000;
// ===================================================

// ================= PLANT PROFILE: Syngonium podophyllum =================
const char *PLANT_NAME = "Syngonium";
const float TEMP_COLD = 15.0;        // below: cold stress, no fan
const float TEMP_HOT = 30.0;         // above: heat stress -> shade + fan
const float DISEASE_TEMP_MIN = 20.0; // leaf-spot favourable band
const float DISEASE_TEMP_MAX = 30.0;
const float DISEASE_HUMIDITY =
    85.0;                           // sustained humidity above this raises risk
const float AIRFLOW_HUM_ON = 80.0;  // warm + above this -> fan for airflow
const float AIRFLOW_HUM_OFF = 75.0; // fan releases only below this (no flicker)
const int SOIL_DRY = 1950;          // calibrated in this pot (dry ~2150-2230)
const unsigned long PUMP_PULSE_MS = 3000; // short pulse, avoids flooding
const unsigned long SOAK_MS = 60000; // let water percolate before re-checking
// =======================================================================

// Reservoir alert -- distance from ultrasonic to water surface.
const long TANK_LOW_CM = 15;               // TUNE to the real reservoir
const unsigned long PIR_WARMUP_MS = 60000; // PIR ignored while it settles

#define DHT_PIN 4
#define DHT_TYPE DHT22
#define ULTRA_TRIG 5
#define ULTRA_ECHO 18
#define RAIN_PIN 27
#define PIR_PIN 19
#define SOIL_PIN 34
#define RELAY_PIN 23 // channel 1 -- pump
#define FAN_PIN 32   // channel 2 -- fan
#define LDR_PIN 35
#define SERVO_PIN 13
#define BUZZER_PIN 17

DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_SSD1306 display(128, 64, &Wire, -1);
Servo coverServo;

int favourableCount = 0;
unsigned long pumpStartedAt = 0;
unsigned long lastWateredAt = 0;
bool pumping = false;
bool everWatered = false;
bool humidFan = false;
unsigned long lastPost = 0;

void postToDashboard(const char *json) {
  static bool wasConnected = false;
  bool connected = WiFi.status() == WL_CONNECTED;
  if (connected != wasConnected) {
    wasConnected = connected;
    Serial.print(connected ? "[WiFi connected, ESP32 IP: " : "[WiFi lost");
    if (connected) Serial.print(WiFi.localIP());
    Serial.println("]");
  }
  if (!connected)
    return;
  HTTPClient http;
  http.setConnectTimeout(
      800); // don't stall the plant logic if the laptop is off
  http.setTimeout(800);
  http.begin(DASH_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(String(json));
  http.end();
  if (code <= 0)
    Serial.println("[dashboard unreachable]");
}

void drawBMOFace(int x, int y) {
  display.fillCircle(x, y, 1, SSD1306_WHITE);
  display.fillCircle(x + 8, y, 1, SSD1306_WHITE);
  display.drawCircleHelper(x + 4, y + 5, 9, 0xC, SSD1306_WHITE);
}

long readDistanceOnce() {
  digitalWrite(ULTRA_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRA_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRA_TRIG, LOW);
  long duration = pulseIn(ULTRA_ECHO, HIGH, 25000);
  if (duration == 0)
    return -1;
  return duration * 0.034 / 2;
}

// Median of 3 readings -- throws out a single stray echo (e.g. a random 190
// cm).
long readDistanceCM() {
  long a = readDistanceOnce();
  delay(30);
  long b = readDistanceOnce();
  delay(30);
  long c = readDistanceOnce();
  if ((a <= b && b <= c) || (c <= b && b <= a))
    return b;
  if ((b <= a && a <= c) || (c <= a && a <= b))
    return a;
  return c;
}

void setup() {
  Serial.begin(115200);
  dht.begin();

  Wire.begin(21, 22);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED not found - check wiring/address (try 0x3D)");
  }
  display.setTextColor(SSD1306_WHITE);

  pinMode(ULTRA_TRIG, OUTPUT);
  pinMode(ULTRA_ECHO, INPUT);
  pinMode(RAIN_PIN, INPUT);
  pinMode(PIR_PIN, INPUT_PULLDOWN); // reads 0 if the PIR wire is ever loose
  pinMode(LDR_PIN, INPUT);

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH); // active-low: HIGH = off
  pinMode(FAN_PIN, OUTPUT);
  digitalWrite(FAN_PIN, HIGH);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  coverServo.attach(SERVO_PIN);
  coverServo.write(0);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000)
    delay(250);
  wl_status_t ws = WiFi.status();
  if (ws == WL_CONNECTED) {
    Serial.print("WiFi connected, ESP32 IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.print("WiFi not connected (");
    Serial.print(ws == WL_NO_SSID_AVAIL ? "network not found -- off, hidden, or 5 GHz"
               : ws == WL_CONNECT_FAILED ? "wrong password or WPA3 security"
                                         : "timed out");
    Serial.println(") -- running offline, will keep retrying");
  }
}

void loop() {
  unsigned long now = millis();

  // ---------- read everything ----------
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();
  bool dhtOk = !isnan(temp) && !isnan(hum);
  int soil = analogRead(SOIL_PIN);
  bool bright = digitalRead(LDR_PIN) == LOW; // confirmed: LOW = bright
  bool raining = digitalRead(RAIN_PIN) == LOW;
  bool pirReady = now > PIR_WARMUP_MS;
  bool motion = pirReady && digitalRead(PIR_PIN) == HIGH;
  long tankCM = readDistanceCM();

  // ---------- M1: leaf-spot risk forecast ----------
  bool favourable = dhtOk && temp >= DISEASE_TEMP_MIN &&
                    temp <= DISEASE_TEMP_MAX && hum >= DISEASE_HUMIDITY;
  if (favourable)
    favourableCount++;
  else if (favourableCount > 0)
    favourableCount--;
  String risk = favourableCount == 0  ? "LOW"
                : favourableCount < 5 ? "MEDIUM"
                                      : "HIGH";

  // ---------- M2: irrigation -- short pulses, soak, never in rain ----------
  bool soilDry = soil > SOIL_DRY;
  bool soaked = !everWatered || (now - lastWateredAt) > SOAK_MS;
  if (!pumping && soilDry && !raining && soaked) {
    pumping = true;
    pumpStartedAt = now;
  }
  if (pumping && (now - pumpStartedAt) >= PUMP_PULSE_MS) {
    pumping = false;
    lastWateredAt = now;
    everWatered = true;
  }
  digitalWrite(RELAY_PIN, pumping ? LOW : HIGH);

  // ---------- M3: climate response ----------
  bool tooHot = dhtOk && temp > TEMP_HOT;
  bool tooCold = dhtOk && temp < TEMP_COLD;
  if (dhtOk && hum >= AIRFLOW_HUM_ON && temp >= DISEASE_TEMP_MIN)
    humidFan = true;
  else if (!dhtOk || hum < AIRFLOW_HUM_OFF || temp < DISEASE_TEMP_MIN)
    humidFan = false;

  bool shade = bright || tooHot; // protect variegated leaves
  bool fanOn = !tooCold && (tooHot || humidFan || risk == "HIGH");
  coverServo.write(shade ? 90 : 0);
  digitalWrite(FAN_PIN, fanOn ? LOW : HIGH);

  // ---------- M4 + M5: tank alert, pest deterrent ----------
  bool tankLow = (tankCM > TANK_LOW_CM || tankCM < 0);
  digitalWrite(BUZZER_PIN, (motion || tankLow) ? HIGH : LOW);

  // ---------- Serial ----------
  Serial.print(PLANT_NAME);
  Serial.print(" | T:");
  Serial.print(temp, 1);
  Serial.print(" H:");
  Serial.print(hum, 0);
  Serial.print(" Risk:");
  Serial.print(risk);
  Serial.print(" | Soil:");
  Serial.print(soil);
  Serial.print(pumping   ? "(WATERING)"
               : soilDry ? (raining ? "(dry, rain-skip)" : "(dry, soaking)")
                         : "(ok)");
  Serial.print(" | Shade:");
  Serial.print(shade);
  Serial.print(" Fan:");
  Serial.print(fanOn);
  Serial.print(tooHot           ? "(heat)"
               : humidFan       ? "(humid)"
               : risk == "HIGH" ? "(risk)"
                                : "");
  Serial.print(" | Tank:");
  Serial.print(tankCM);
  Serial.print("cm");
  Serial.print(tankLow ? "(LOW)" : "");
  Serial.print(" Rain:");
  Serial.print(raining);
  Serial.print(" Motion:");
  Serial.println(pirReady ? (motion ? "1" : "0") : "warming");

  // ---------- Dashboard ----------
  if (now - lastPost >= POST_EVERY_MS) {
    lastPost = now;
    const char *soilState = pumping   ? "watering"
                            : soilDry ? (raining ? "rain-skip" : "soaking")
                                      : "ok";
    const char *fanReason = !fanOn     ? ""
                            : tooHot   ? "heat"
                            : humidFan ? "humid"
                                       : "risk";
    char t[12], h[12];
    if (dhtOk) {
      snprintf(t, sizeof(t), "%.1f", temp);
      snprintf(h, sizeof(h), "%.0f", hum);
    } else {
      strcpy(t, "null");
      strcpy(h, "null");
    }
#define JB(x) ((x) ? "true" : "false")
    char json[480];
    snprintf(json, sizeof(json),
             "{\"plant\":\"%s\",\"temp\":%s,\"humidity\":%s,\"risk\":\"%s\","
             "\"soil_raw\":%d,"
             "\"soil_state\":\"%s\",\"pumping\":%s,\"shade\":%s,\"bright\":%s,"
             "\"fan\":%s,"
             "\"fan_reason\":\"%s\",\"tank_cm\":%ld,\"tank_low\":%s,\"rain\":%"
             "s,\"motion\":%s}",
             PLANT_NAME, t, h, risk.c_str(), soil, soilState, JB(pumping),
             JB(shade), JB(bright), JB(fanOn), fanReason, tankCM, JB(tankLow),
             JB(raining), JB(motion));
    postToDashboard(json);
  }

  // ---------- OLED ----------
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print(PLANT_NAME);
  display.setCursor(0, 11);
  display.print("Risk: ");
  display.println(risk);
  display.setCursor(0, 22);
  display.print("T:");
  display.print(temp, 1);
  display.print(" H:");
  display.print(hum, 0);
  display.println("%");
  display.setCursor(0, 33);
  display.print("Soil:");
  display.println(pumping ? "WATERING" : soilDry ? "DRY" : "OK");
  display.setCursor(0, 44);
  display.print("Shade:");
  display.print(shade ? "ON " : "off");
  display.print(" Fan:");
  display.println(fanOn ? "ON" : "off");
  display.setCursor(0, 55);
  display.print("Tank:");
  display.print(tankCM);
  display.print("cm ");
  display.print(motion ? "MOTION" : "");
  drawBMOFace(104, 8);
  display.display();

  delay(1000);
}
