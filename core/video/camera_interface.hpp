#pragma once

#include <functional>
#include <optional>
#include <string>

#include "core/mat/mat.hpp"

namespace core::video {

// Provides a abstract API for different camera backends (V4L2, RealSense, etc.)

class CameraInterface {
 public:
  virtual ~CameraInterface() = default;

  /**
   * Try to capture a frame and return as RGB float32 data.
   * @return RGB Mat if frame available, nullopt if no frame ready
   */
  virtual std::optional<std::reference_wrapper<const core::Mat>>
  try_grab_rgb() = 0;
  virtual int width() const noexcept = 0;
  virtual int height() const noexcept = 0;
  virtual bool streaming() const noexcept = 0;
  virtual std::string device() const noexcept = 0;
};

}  // namespace core::video
