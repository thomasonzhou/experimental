#include <cstdint>

#include "core/mat/mat.hpp"

namespace core::video {
core::Mat convert_yuyv_to_rgb_f32(const std::uint8_t* src, int w, int h);

};  // namespace core::video
