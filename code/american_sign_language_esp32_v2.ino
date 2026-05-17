#define CAMERA_MODEL_AI_THINKER

#include <Arduino.h>
#include <WiFi.h>

#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include <esp_heap_caps.h>

#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/generated.h"   // sometimes called schema_generated.h depending on lib
// If above gives error, replace with:  #include "tensorflow/lite/schema/schema_generated.h"

#include "p_det_model.h"      // contains ASL_1_lite_tflite and ASL_1_lite_tflite_len
#include "model_settings.h"   // kNumRows, kNumCols, kCategoryCount, extern kCategoryLabels

#include "camera_pins.h"
#include "downsample.h"       // dstImage, img192x192, downsampleImage(), upsample()

#include "soc/soc.h"              // Disable brownout problems
#include "soc/rtc_cntl_reg.h"     // Disable brownout problems

// ------------- WiFi -------------
const char* ssid     = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";

// ------------- TFLite globals -------------
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model*   model          = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor            = nullptr;

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 39 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif

constexpr int kTensorArenaSize = 81 * 1024 + scratchBufSize;
static uint8_t* tensor_arena = nullptr;

// ------------- Camera / HTTP globals -------------
httpd_handle_t server = NULL;

// Reuse your category labels from model_settings.cpp
extern const char* kCategoryLabels[kCategoryCount];

// ------------- Camera init (same settings as original TFT code) -------------
void init_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;  // like original
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.frame_size   = FRAMESIZE_96X96;   // like original
  config.jpeg_quality = 12;
  config.fb_count     = 2;

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    delay(2000);
    ESP.restart();
  }
}

// ------------- TFLite setup (same ops as original) -------------
void setup_tflite() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(ASL_1_lite_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model schema %d not equal to supported %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (tensor_arena == nullptr) {
    tensor_arena = (uint8_t*) heap_caps_malloc(
      kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  }

  if (tensor_arena == nullptr) {
    Serial.printf("Couldn't allocate tensor_arena of %d bytes\n", kTensorArenaSize);
    return;
  }

  static tflite::MicroMutableOpResolver<9> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  input_tensor = interpreter->input(0);
}

// ------------- Inference pipeline (one-shot) -------------
bool run_inference_once(String &top1, String &top2, uint32_t &infer_ms) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed for inference");
    return false;
  }

  // Use exactly your original path: 96x96 RGB565 -> downsample -> dstImage (28x28)
  downsampleImage((uint16_t*)fb->buf, fb->width, fb->height);

  // We don't need fb for input anymore
  esp_camera_fb_return(fb);

  // Fill TFLite input
  int8_t* image_data = input_tensor->data.int8;

  for (int i = 0; i < kNumRows; i++) {
    for (int j = 0; j < kNumCols; j++) {
      uint16_t pixel = ((uint16_t*)dstImage)[i * kNumCols + j];

      uint8_t hb = pixel & 0xFF;
      uint8_t lb = pixel >> 8;
      uint8_t r  = (lb & 0x1F) << 3;
      uint8_t g  = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
      uint8_t b  = (hb & 0xF8);

      int8_t grey_pixel = ((305 * r + 600 * g + 119 * b) >> 10) - 128;
      image_data[i * kNumCols + j] = grey_pixel;
    }
  }

  int64_t t0 = esp_timer_get_time();
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return false;
  }
  int64_t t1 = esp_timer_get_time();
  infer_ms = (uint32_t)((t1 - t0) / 1000);

  TfLiteTensor* output = interpreter->output(0);

  int idx  = 0;
  int idx2 = 0;
  int8_t max_conf  = output->data.uint8[0];
  int8_t cur_conf;
  float  max_tmp = -10000.0f;

  for (int i = 0; i < kCategoryCount; i++) {
    float tmp = output->data.f[i];      // model seems to also provide float
    cur_conf  = output->data.uint8[i];  // and uint8

    if (max_conf < cur_conf) {
      idx2     = idx;
      idx      = i;
      max_conf = cur_conf;
    }
    if (tmp > max_tmp) {
      max_tmp = tmp;
    }
  }

  top1 = String(kCategoryLabels[idx]);
  top2 = String(kCategoryLabels[idx2]);

  Serial.print("Top1: ");
  Serial.print(top1);
  Serial.print("  Top2: ");
  Serial.print(top2);
  Serial.print("  conf: ");
  Serial.println((int)max_conf);

  return true;
}

// ------------- HTTP handlers -------------

// Root: simple HTML page
static esp_err_t root_handler(httpd_req_t *req) {
  const char* html = R"rawlit(
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ESP32 ASL Sign Detection</title>
  <style>
    body { font-family: Arial, sans-serif; text-align: center; background: #111; color: #eee; }
    #box { display: inline-block; border: 2px solid #444; padding: 10px; background: #222; }
    #preview { width: 384px; height: 384px; background: #000; }
    button { margin-top: 15px; padding: 10px 20px; font-size: 16px; }
    #status { margin-top: 10px; font-weight: bold; }
    #result { margin-top: 5px; font-size: 20px; }
  </style>
</head>
<body>
  <h2>ASL Detection (ESP32-CAM)</h2>
  <div id="box">
    <img id="preview" src="/frame.jpg" alt="camera view">
    <br>
    <button id="btn">Detect Sign</button>
    <div id="status">Idle</div>
    <div id="result">Top1: -, Top2: -</div>
  </div>

<script>
const img    = document.getElementById('preview');
const btn    = document.getElementById('btn');
const status = document.getElementById('status');
const result = document.getElementById('result');

// live view: just refresh /frame.jpg
setInterval(() => {
  img.src = '/frame.jpg?t=' + Date.now();
}, 200);

btn.addEventListener('click', async () => {
  status.textContent = 'Calculating...';
  try {
    const resp = await fetch('/run');
    const data = await resp.json();
    status.textContent = 'Done in ' + data.ms + ' ms';
    result.textContent = 'Top1: ' + data.top1 + ', Top2: ' + data.top2;
  } catch (e) {
    console.error(e);
    status.textContent = 'Error during inference';
  }
});
</script>
</body>
</html>
  )rawlit";

  httpd_resp_set_type(req, "text/html");
  httpd_resp_send(req, html, strlen(html));
  return ESP_OK;
}

// /frame.jpg : show what TFT would have shown (upsampled 192x192)
static esp_err_t frame_handler(httpd_req_t *req) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  // Use your exact processing: 96x96 -> upsample to 192x192 using your function
  // downsampleImage updates dstImage, but upsample uses original fb->buf
  downsampleImage((uint16_t*)fb->buf, fb->width, fb->height);
  upsample((uint16_t*)fb->buf);   // fills img192x192 (192x192 RGB565)

  uint8_t* jpg_buf = nullptr;
  size_t   jpg_len = 0;

  bool ok = fmt2jpg(
      (uint8_t*)img192x192,
      192 * 192 * 2,
      192,
      192,
      PIXFORMAT_RGB565,
      90,
      &jpg_buf,
      &jpg_len
  );

  esp_camera_fb_return(fb);

  if (!ok || !jpg_buf) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
  httpd_resp_send(req, (const char*)jpg_buf, jpg_len);

  free(jpg_buf);
  return ESP_OK;
}

// /run : capture + inference, return JSON
static esp_err_t run_handler(httpd_req_t *req) {
  String top1, top2;
  uint32_t ms = 0;

  bool ok = run_inference_once(top1, top2, ms);
  if (!ok) {
    httpd_resp_set_status(req, "500 Internal Server Error");
    const char* msg = "{\"error\":\"inference failed\"}";
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, msg, strlen(msg));
    return ESP_FAIL;
  }

  String json = "{";
  json += "\"top1\":\"" + top1 + "\",";
  json += "\"top2\":\"" + top2 + "\",";
  json += "\"ms\":" + String(ms);
  json += "}";

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_send(req, json.c_str(), json.length());
  return ESP_OK;
}

void start_webserver() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t root_uri = {
    .uri      = "/",
    .method   = HTTP_GET,
    .handler  = root_handler,
    .user_ctx = NULL
  };

  httpd_uri_t frame_uri = {
    .uri      = "/frame.jpg",
    .method   = HTTP_GET,
    .handler  = frame_handler,
    .user_ctx = NULL
  };

  httpd_uri_t run_uri = {
    .uri      = "/run",
    .method   = HTTP_GET,
    .handler  = run_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&server, &config) == ESP_OK) {
    httpd_register_uri_handler(server, &root_uri);
    httpd_register_uri_handler(server, &frame_uri);
    httpd_register_uri_handler(server, &run_uri);
    Serial.println("HTTP server started");
  } else {
    Serial.println("Failed to start HTTP server");
  }
}

// ------------- setup / loop -------------

void setup() {
  // disable brownout
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  delay(100);

  init_camera();
  setup_tflite();

  // allocate image buffers exactly like original
  dstImage   = (uint16_t*) malloc(DST_WIDTH * DST_HEIGHT * 2);
  img192x192 = (uint16_t*) malloc(192 * 192 * 2);

  if (!dstImage || !img192x192) {
    Serial.println("Image buffer allocation failed");
  }

  // WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected, IP: ");
  Serial.println(WiFi.localIP());

  start_webserver();
}

void loop() {
  delay(1000); // everything is event-driven by HTTP
}