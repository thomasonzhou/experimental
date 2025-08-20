#pragma once

#include <linux/videodev2.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/mat/mat.hpp"
#include "core/video/v4l2/v4l2_utils.hpp"

namespace core::video::v4l2 {

struct Config {
  std::string device{"/dev/video0"};
  int width{640};
  int height{480};
  int fps{30};
  unsigned int pixfmt{V4L2_PIX_FMT_YUYV};
};

class Camera {
 public:
  explicit Camera(const Config& cfg);
  ~Camera();

  std::optional<std::reference_wrapper<const core::Mat>> try_grab_rgb();

  Config config() const { return cfg_; }

  int width() const { return w_; }
  int height() const { return h_; }
  bool streaming() const { return streaming_; }

 private:
  void init_device();
  void start_streaming();
  void stop_streaming();
  void cleanup();

  Config cfg_;
  int fd_{-1};
  int w_{0}, h_{0};
  bool streaming_{false};
  std::vector<MappedBuffer> bufs_;
  mutable core::Mat rgb_buffer_;
};

}  // namespace core::video::v4l2
