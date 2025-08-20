#include "core/video/color_conversion.hpp"

#include <algorithm>

namespace core::video {

// YUYV 4:2:2 -> RGB F32
core::Mat convert_yuyv_to_rgb_f32(const std::uint8_t* src, int w, int h,
                                  const ColorSpaceGains& gains) {
  core::Mat out_rgb(h, w, 3);
  const int n = w * h;
  const std::uint8_t* s = src;
  float* d = out_rgb.data();
  for (int i = 0; i < n; i += 2) {
    int y0 = s[0];
    int u = s[1] - 128;
    int y1 = s[2];
    int v = s[3] - 128;
    auto y2rgb = [&](int y, float* o) {
      float c = y - 16.0f;
      float r = gains.kY * c + gains.kRV * v;
      float g = gains.kY * c + gains.kGV * v + gains.kGU * u;
      float b = gains.kY * c + gains.kBU * u;
      o[0] = std::clamp(r / 255.0f, 0.0f, 1.0f);
      o[1] = std::clamp(g / 255.0f, 0.0f, 1.0f);
      o[2] = std::clamp(b / 255.0f, 0.0f, 1.0f);
    };
    y2rgb(y0, d + 0);
    y2rgb(y1, d + 3);
    s += 4;
    d += 6;
  }
  return out_rgb;
}
};  // namespace core::video
