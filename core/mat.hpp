#pragma once

#include <memory>
#include <string>

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

namespace core {
class Mat {
 public:
  Mat(const std::string filename);

  Mat(const int height, const int width, const int channels,
      const unsigned char value);
  Mat(const int batch, const int height, const int width, const int channels,
      const unsigned char value);

  unsigned char* data() { return img_ptr.get(); }
  const unsigned char* data() const { return img_ptr.get(); }

  const int calculate_index(int batch, int row, int col,
                            int channel) const noexcept {
    return batch * height_ * width_ * channels_ + row * width_ * channels_ +
           col * channels_ + channel;
  }

  unsigned char& operator()(int batch, int row, int col, int channel) {
    return img_ptr.get()[calculate_index(batch, row, col, channel)];
  }
  const unsigned char& operator()(int batch, int row, int col,
                                  int channel) const {
    return img_ptr.get()[calculate_index(batch, row, col, channel)];
  }

  const bool operator==(const Mat& other) const {
    if (batch_size_ != other.batch_size_ || height_ != other.height_ ||
        width_ != other.width_ || channels_ != other.channels_) {
      return false;
    }
    for (int b = 0; b < batch_size_; ++b) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          for (int c = 0; c < channels_; ++c) {
            if (this->operator()(b, h, w, c) != other(b, h, w, c)) {
              return false;
            }
          }
        }
      }
    }
    return true;
  }

  int batch_size() const { return batch_size_; }
  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }
  int size() const { return batch_size_ * width_ * height_ * channels_; }

 private:
  std::unique_ptr<unsigned char, decltype(&stbi_image_free)> img_ptr;
  int batch_size_, height_, width_, channels_;
};
};  // namespace core
