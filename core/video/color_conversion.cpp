#include "core/video/color_conversion.hpp"

namespace core::video {
// YUYV (Y0 U0 Y1 V0) -> RGB F32 (BT.601)
core::Mat convert_yuyv_to_rgb_f32(const std::uint8_t* src, int w, int h) {
  core::Mat out_rgb(h, w, 3);
  const int n = w * h;
  auto clamp = [](float v) {
    return v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
  };
  const std::uint8_t* s = src;
  float* d = out_rgb.data();
  for (int i = 0; i < n; i += 2) {
    int y0 = s[0];
    int u = s[1] - 128;
    int y1 = s[2];
    int v = s[3] - 128;
    auto y2rgb = [&](int y, float* o) {
      int c = y - 16;
      float r = (298 * c + 409 * v + 128) >> 8;
      float g = (298 * c - 100 * u - 208 * v + 128) >> 8;
      float b = (298 * c + 516 * u + 128) >> 8;
      o[0] = clamp(r) / 255.0f;
      o[1] = clamp(g) / 255.0f;
      o[2] = clamp(b) / 255.0f;
    };
    y2rgb(y0, d + 0);
    y2rgb(y1, d + 3);
    s += 4;
    d += 6;
  }
  return out_rgb;
}
};  // namespace core::video
