#include "core/video/v4l2/v4l2_camera.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "core/video/color_conversion.hpp"

namespace core::video::v4l2 {

Camera::Camera(const Config& cfg) : cfg_(cfg) {
  init_device();
  start_streaming();
}

Camera::~Camera() { cleanup(); }

void Camera::init_device() {
  fd_ = ::open(cfg_.device.c_str(), O_RDWR | O_NONBLOCK);
  if (fd_ < 0) {
    throw std::runtime_error("open(" + cfg_.device +
                             ") failed: " + std::string(std::strerror(errno)));
  }

  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = cfg_.width;
  fmt.fmt.pix.height = cfg_.height;
  fmt.fmt.pix.pixelformat = cfg_.pixfmt;
  fmt.fmt.pix.field = V4L2_FIELD_ANY;

  if (xioctl(fd_, VIDIOC_S_FMT, &fmt) == -1) {
    throw std::runtime_error("VIDIOC_S_FMT failed");
  }

  w_ = (int)fmt.fmt.pix.width;
  h_ = (int)fmt.fmt.pix.height;

  v4l2_requestbuffers req{};
  req.count = 4;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (xioctl(fd_, VIDIOC_REQBUFS, &req) == -1 || req.count < 2) {
    throw std::runtime_error("VIDIOC_REQBUFS failed");
  }

  bufs_.resize(req.count);
  for (std::size_t i = 0; i < bufs_.size(); ++i) {
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = (uint32_t)i;

    if (xioctl(fd_, VIDIOC_QUERYBUF, &b) == -1) {
      throw std::runtime_error("VIDIOC_QUERYBUF failed");
    }

    void* m = ::mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
                     b.m.offset);
    if (m == MAP_FAILED) {
      throw std::runtime_error("mmap failed");
    }

    bufs_[i].data = m;
    bufs_[i].size = b.length;
  }
}

void Camera::start_streaming() {
  for (std::size_t i = 0; i < bufs_.size(); ++i) {
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = (uint32_t)i;

    if (xioctl(fd_, VIDIOC_QBUF, &b) == -1) {
      throw std::runtime_error("VIDIOC_QBUF failed");
    }
  }

  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(fd_, VIDIOC_STREAMON, &type) == -1) {
    throw std::runtime_error("VIDIOC_STREAMON failed");
  }

  streaming_ = true;
}

void Camera::stop_streaming() {
  if (streaming_ && fd_ >= 0) {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    (void)xioctl(fd_, VIDIOC_STREAMOFF, &type);
    streaming_ = false;
  }
}

void Camera::cleanup() {
  stop_streaming();

  for (auto& b : bufs_) {
    if (b.data && b.size) {
      munmap(b.data, b.size);
    }
  }
  bufs_.clear();

  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

std::optional<std::reference_wrapper<const core::Mat>> Camera::try_grab_rgb() {
  if (!streaming_) return std::nullopt;

  struct pollfd pfd{fd_, POLLIN, 0};
  int pr = ::poll(&pfd, 1, 0);
  if (pr <= 0) return std::nullopt;

  v4l2_buffer buf;
  std::memset(&buf, 0, sizeof(buf));
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;

  if (xioctl(fd_, VIDIOC_DQBUF, &buf) == -1) {
    return std::nullopt;
  }

  const std::uint8_t* yuyv =
      static_cast<const std::uint8_t*>(bufs_[buf.index].data);

  core::video::convert_yuyv_to_rgb_f32_inplace(
      yuyv, w_, h_, core::video::kBT709, rgb_buffer_);

  if (xioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
    return std::nullopt;
  }

  return std::cref(rgb_buffer_);
}

}  // namespace core::video::v4l2
