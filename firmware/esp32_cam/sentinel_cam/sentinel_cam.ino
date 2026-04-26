/*
 * ╔══════════════════════════════════════════════════════════════╗
 * ║  PROJECT SENTINEL V2 — ESP32-CAM Firmware                   ║
 * ║  Board: AI Thinker ESP32-CAM                                 ║
 * ║                                                              ║
 * ║  Functions:                                                  ║
 * ║  1. Streams MJPEG video on  → http://[IP]:81/stream         ║
 * ║  2. Opens gate on command   → http://[IP]/access_granted    ║
 * ║  3. Denies entry on command → http://[IP]/access_denied     ║
 * ║                                                              ║
 * ║  WIRING (GPIO 12 → Relay IN):                               ║
 * ║    GPIO 12 ─── Relay IN                                      ║
 * ║    3.3V    ─── Relay VCC                                     ║
 * ║    GND     ─── Relay GND                                     ║
 * ║                                                              ║
 * ║  HOW TO FLASH:                                               ║
 * ║  1. Board: "AI Thinker ESP32-CAM" in Arduino IDE            ║
 * ║  2. Partition: "Huge APP (3MB No OTA/1MB SPIFFS)"           ║
 * ║  3. Put GPIO 0 to GND before flashing, remove after         ║
 * ║  4. Open Serial Monitor (115200 baud) to see the IP         ║
 * ╚══════════════════════════════════════════════════════════════╝
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ── Wi-Fi Credentials ─────────────────────────────────────────────────────────
// CHANGE THESE TO YOUR NETWORK
const char* WIFI_SSID = "AndroidAP8469";
const char* WIFI_PASSWORD = "12345678";

// ── Gate Indicator Pin ───────────────────────────────────────────────────────
// Using the onboard WHITE FLASH LED (GPIO 4) — no extra wiring needed!
// In a real deployment, change this to GPIO 12 and connect a relay.
// HIGH = LED ON (gate open indicator), LOW = LED OFF (gate closed)
#define GATE_PIN 4
#define GATE_OPEN_MS 5000  // How long to keep the LED on (milliseconds)

// ── Camera Pin Config (AI Thinker ESP32-CAM) ─────────────────────────────────
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

// ── HTTP Servers ──────────────────────────────────────────────────────────────
// Port 80 for commands, Port 81 for video stream
WebServer commandServer(80);
WebServer streamServer(81);

bool gateOpen = false;
unsigned long gateOpenedAt = 0;

// ── Camera Setup ──────────────────────────────────────────────────────────────
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Use SVGA (800x600) for InsightFace — good balance of quality vs speed
  // If stream is too slow, change to VGA (640x480):
  //   config.frame_size = FRAMESIZE_VGA;
  config.frame_size = FRAMESIZE_SVGA;
  config.jpeg_quality = 12;  // 0=best, 63=worst. 12 is production quality.
  config.fb_count = 2;       // 2 frame buffers for smoother streaming

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Camera init failed: 0x%x\n", err);
    return false;
  }

  // Flip the image if the camera is mounted upside down
  sensor_t* s = esp_camera_sensor_get();
  s->set_vflip(s, 0);    // set to 1 if image is upside down
  s->set_hmirror(s, 1);  // Mirror horizontally (standard for face cams)

  return true;
}

// ── MJPEG Stream Handler ──────────────────────────────────────────────────────
void handleStream() {
  WiFiClient client = streamServer.client();

  // Send MJPEG multipart header
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  client.print(response);

  while (client.connected()) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("[WARN] Camera frame capture failed");
      break;
    }

    // Send MJPEG frame boundary and headers
    client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
    client.write(fb->buf, fb->len);
    client.print("\r\n");

    esp_camera_fb_return(fb);

    // Yield to prevent watchdog reset
    delay(10);
  }
}

// ── Gate Command Handlers ─────────────────────────────────────────────────────
void handleAccessGranted() {
  Serial.println("[GATE] ACCESS GRANTED — Flash LED ON (gate open demo)!");
  digitalWrite(GATE_PIN, HIGH);  // Turn on flash LED
  gateOpen = true;
  gateOpenedAt = millis();
  commandServer.send(200, "application/json", "{\"status\":\"gate_opened\"}");
}

void handleAccessDenied() {
  Serial.println("[GATE] ACCESS DENIED — Gate stays closed.");
  // Optionally flash the onboard LED as a denied indicator
  for (int i = 0; i < 3; i++) {
    digitalWrite(4, HIGH);  // Flash LED (GPIO 4 = flash LED on AI Thinker)
    delay(100);
    digitalWrite(4, LOW);
    delay(100);
  }
  commandServer.send(200, "application/json", "{\"status\":\"access_denied\"}");
}

void handleHealth() {
  String json = "{\"status\":\"online\",\"gate\":\"";
  json += gateOpen ? "open" : "closed";
  json += "\"}";
  commandServer.send(200, "application/json", json);
}

// ── Arduino Setup ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("\n[SENTINEL] Booting ESP32-CAM...");

  // Gate relay pin
  pinMode(GATE_PIN, OUTPUT);
  digitalWrite(GATE_PIN, LOW);  // Ensure gate is closed at boot

  // Flash LED pin
  pinMode(4, OUTPUT);
  digitalWrite(4, LOW);

  // Initialize camera
  if (!initCamera()) {
    Serial.println("[FATAL] Camera failed. Halting.");
    while (true) delay(1000);
  }
  Serial.println("[OK] Camera initialized.");

  // Connect to Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WIFI] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("[WIFI] Connected!");
  Serial.print("[WIFI] ESP32-CAM IP Address: ");
  Serial.println(WiFi.localIP());

  // Register command routes (port 80)
  commandServer.on("/access_granted", HTTP_GET, handleAccessGranted);
  commandServer.on("/access_denied", HTTP_GET, handleAccessDenied);
  commandServer.on("/health", HTTP_GET, handleHealth);

  // Register stream route (port 81)
  streamServer.on("/stream", HTTP_GET, handleStream);

  commandServer.begin();
  streamServer.begin();

  Serial.println("════════════════════════════════════════");
  Serial.println("  SENTINEL ESP32-CAM ONLINE");
  Serial.printf("  Video Stream : http://%s:81/stream\n", WiFi.localIP().toString().c_str());
  Serial.printf("  Gate Open    : http://%s/access_granted\n", WiFi.localIP().toString().c_str());
  Serial.printf("  Gate Deny    : http://%s/access_denied\n", WiFi.localIP().toString().c_str());
  Serial.printf("  Health Check : http://%s/health\n", WiFi.localIP().toString().c_str());
  Serial.println("════════════════════════════════════════");

  // Boot confirmation: 2 quick LED flashes
  for (int i = 0; i < 2; i++) {
    digitalWrite(4, HIGH);
    delay(200);
    digitalWrite(4, LOW);
    delay(200);
  }
}

// ── Arduino Loop ──────────────────────────────────────────────────────────────
void loop() {
  commandServer.handleClient();
  streamServer.handleClient();

  // Auto-close gate after GATE_OPEN_MS milliseconds
  if (gateOpen && (millis() - gateOpenedAt > GATE_OPEN_MS)) {
    Serial.println("[GATE] Closing — LED OFF.");
    digitalWrite(GATE_PIN, LOW);
    gateOpen = false;
  }
}
