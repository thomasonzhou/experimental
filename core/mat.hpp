#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

namespace core {
class Mat {
 public:
  Mat(const std::string& filename);

  Mat(const int height, const int width, const int channels,
      const unsigned char value);

  unsigned char* data() { return img_ptr.get(); }
  const unsigned char* data() const { return img_ptr.get(); }

  const int calculate_index(int row, int col, int channel) const noexcept {
    return row * width_ * channels_ + col * channels_ + channel;
  }

  Mat clone() const {
    Mat copy(height_, width_, channels_, 0);
    std::copy(img_ptr.get(), img_ptr.get() + size(), copy.img_ptr.get());
    return copy;
  }

  // accessors for pixel values
  unsigned char& operator()(const int row, const int col, const int channel) {
    return img_ptr.get()[calculate_index(row, col, channel)];
  }
  const unsigned char& operator()(const int row, const int col,
                                  const int channel) const {
    return img_ptr.get()[calculate_index(row, col, channel)];
  }

  unsigned char& operator()(const int row, const int col) {
    if (channels_ != 1) {
      throw std::runtime_error(
          "This operator is only valid for single-channel images.");
    }
    return img_ptr.get()[calculate_index(row, col, 0)];
  }
  const unsigned char& operator()(const int row, const int col) const {
    if (channels_ != 1) {
      throw std::runtime_error(
          "This operator is only valid for single-channel images.");
    }
    return img_ptr.get()[calculate_index(row, col, 0)];
  }

  // equality
  const bool operator==(const Mat& other) const {
    if (height_ != other.height_ || width_ != other.width_ ||
        channels_ != other.channels_) {
      return false;
    }
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
        for (int c = 0; c < channels_; ++c) {
          if (this->operator()(h, w, c) != other(h, w, c)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  const bool operator!=(const Mat& other) const { return !(*this == other); }

  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }
  int size() const { return width_ * height_ * channels_; }

 private:
  std::unique_ptr<unsigned char, decltype(&stbi_image_free)> img_ptr;
  int height_, width_, channels_;
};

Mat imread(const std::string& filename);

};  // namespace core
