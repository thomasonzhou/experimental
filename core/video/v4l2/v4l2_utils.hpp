#pragma once

#include <sys/ioctl.h>

#include <cerrno>

namespace core::video::v4l2 {

static int xioctl(int fd, unsigned long req, void* arg) noexcept {
  int r;
  do {
    r = ioctl(fd, req, arg);
  } while (r == -1 && errno == EINTR);
  return r;
}

struct MappedBuffer {
  void* data;
  std::size_t size;
  MappedBuffer() : data(nullptr), size(0) {}
};

}  // namespace core::video::v4l2
