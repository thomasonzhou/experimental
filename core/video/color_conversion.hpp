#pragma once

#include <cstdint>

#include "core/mat/mat.hpp"

namespace core::video {
struct ColorSpaceGains {
  const float kY, kRV, kGV, kBU, kGU;
};

// D = 1 for 8 bit quantization, D = 4 for 10 bit
constexpr ColorSpaceGains compute_color_space_gains(const float Kr,
                                                    const float Kb) noexcept {
  const float kY = 255.0f / 219.0f;  // luma
  const float kC = 255.0f / 224.0f;  // chroma

  const float Kg = 1.0f - Kr - Kb;

  return {
      .kY = kY,
      .kRV = kC * 2.0f * (1.0f - Kr),
      .kGV = -kC * 2.0f * Kr * (1.0f - Kr) / Kg,
      .kBU = kC * 2.0f * (1.0f - Kb),
      .kGU = -kC * 2.0f * Kb * (1.0f - Kb) / Kg,
  };
}

// https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I%21%21PDF-E.pdf
inline constexpr ColorSpaceGains kBT601 =
    compute_color_space_gains(0.299f, 0.114f);
// https://www.itu.int/dms_pubrec/itu-r/rec/bt/r-rec-bt.709-6-201506-i%21%21pdf-e.pdf
inline constexpr ColorSpaceGains kBT709 =
    compute_color_space_gains(0.2126f, 0.0722f);

core::Mat convert_yuyv_to_rgb_f32(const std::uint8_t* src, int w, int h,
                                  const ColorSpaceGains& cs);

};  // namespace core::video
