#include "core/mat/utils/color_conversion.hpp"

#include <algorithm>

#include "absl/log/log.h"

namespace core::video {

// YUYV 4:2:2 -> RGB F32
void convert_yuyv_to_rgb_f32_inplace(const std::uint8_t* src, int w, int h,
                                     const ColorSpaceGains& gains,
                                     core::Mat& output) {
  // Ensure output Mat has correct dimensions
  if (output.rows() != static_cast<size_t>(h) ||
      output.cols() != static_cast<size_t>(w) || output.channels() != 3) {
    output = core::Mat(core::MatShape::make_3d(h, w, 3));
  }

  const int n = w * h;
  const std::uint8_t* s = src;
  float* d = output.data();

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
}

core::Mat convert_yuyv_to_rgb_f32(const std::uint8_t* src, int w, int h,
                                  const ColorSpaceGains& gains) {
  core::Mat out_rgb(core::MatShape::make_3d(h, w, 3));
  convert_yuyv_to_rgb_f32_inplace(src, w, h, gains, out_rgb);
  return out_rgb;
}

core::Mat convert_rgb_to_grayscale(const core::Mat& rgb_image) {
  if (rgb_image.channels() != 3) {
    LOG(FATAL) << "Expected RGB image with 3 channels, got "
               << rgb_image.channels();
  }

  core::Mat gray(core::MatShape::make_3d(rgb_image.rows(), rgb_image.cols(), 1),
                 std::nullopt, core::MatLayout::NHWC);

  const size_t pixel_count = rgb_image.rows() * rgb_image.cols();

  for (size_t r = 0; r < rgb_image.rows(); ++r) {
    for (size_t c = 0; c < rgb_image.cols(); ++c) {
      const float r_val = rgb_image(r, c, 0);
      const float g_val = rgb_image(r, c, 1);
      const float b_val = rgb_image(r, c, 2);

      const float gray_val = 0.299f * r_val + 0.587f * g_val + 0.114f * b_val;

      gray(r, c, 0) = gray_val;
    }
  }

  return gray;
}

};  // namespace core::video
