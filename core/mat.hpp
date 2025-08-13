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

  Mat(const int height, const int width, const int channels, const float value);

  float* data() { return data_ptr.get(); }
  const float* data() const { return data_ptr.get(); }

  const int calculate_index(int row, int col, int channel) const noexcept {
    return row * width_ * channels_ + col * channels_ + channel;
  }

  Mat clone() const {
    Mat copy(height_, width_, channels_, 0.0f);
    std::copy(data_ptr.get(), data_ptr.get() + size(), copy.data_ptr.get());
    return copy;
  }

  // accessors for pixel values
  float& operator()(const int row, const int col, const int channel) {
    return data_ptr.get()[calculate_index(row, col, channel)];
  }
  const float& operator()(const int row, const int col,
                          const int channel) const {
    return data_ptr.get()[calculate_index(row, col, channel)];
  }

  float& operator()(const int row, const int col) {
    if (channels_ != 1) {
      throw std::runtime_error(
          "This operator is only valid for single-channel matrices.");
    }
    return data_ptr.get()[calculate_index(row, col, 0)];
  }
  const float& operator()(const int row, const int col) const {
    if (channels_ != 1) {
      throw std::runtime_error(
          "This operator is only valid for single-channel matrices.");
    }
    return data_ptr.get()[calculate_index(row, col, 0)];
  }

  // operators
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

  const Mat operator+(const float value) const {
    Mat result(height_, width_, channels_, 0.0f);
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
        for (int c = 0; c < channels_; ++c) {
          result(h, w, c) = this->operator()(h, w, c) + value;
        }
      }
    }
    return result;
  }

  const Mat operator+(const Mat& other) const {
    if (height_ != other.height_ || width_ != other.width_ ||
        channels_ != other.channels_) {
      throw std::runtime_error("Incompatible matrix dimensions");
    }
    Mat result(height_, width_, channels_, 0.0f);
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
        for (int c = 0; c < channels_; ++c) {
          result(h, w, c) = this->operator()(h, w, c) + other(h, w, c);
        }
      }
    }
    return result;
  }

  const Mat operator*(float scalar) const {
    Mat result(height_, width_, channels_, 0.0f);
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
        for (int c = 0; c < channels_; ++c) {
          result(h, w, c) = this->operator()(h, w, c) * scalar;
        }
      }
    }
    return result;
  }

  const Mat operator*(double scalar) const {
    return (*this) * static_cast<float>(scalar);
  }

  const Mat operator-(const Mat& other) const {
    if (height_ != other.height_ || width_ != other.width_ ||
        channels_ != other.channels_) {
      throw std::runtime_error("Incompatible matrix dimensions");
    }
    return *this + (other * -1.0f);
  }

  const Mat operator-() const { return *this * -1.0f; }

  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }
  int size() const { return width_ * height_ * channels_; }

 private:
  std::unique_ptr<float[]> data_ptr;
  int height_, width_, channels_;
};

static inline Mat imread(const std::string& filename) {
  Mat img(filename);
  if (!img.data()) {
    throw std::runtime_error("Failed to read image");
  }
  return img;
}

static inline Mat ones(const int height, const int width, const int channels) {
  Mat mat(height, width, channels, 1.0f);
  return mat;
}
static inline Mat ones(const int height, const int width) {
  return ones(height, width, 1);
}

static inline Mat zeros(const int height, const int width, const int channels) {
  Mat mat(height, width, channels, 0.0f);
  return mat;
}
static inline Mat zeros(const int height, const int width) {
  return zeros(height, width, 1);
}

static inline Mat operator*(float scalar, const Mat& mat) {
  return mat * scalar;
}

static inline Mat operator*(double scalar, const Mat& mat) {
  return mat * scalar;
}

};  // namespace core
