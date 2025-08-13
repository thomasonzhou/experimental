#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include "core/mat.pb.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

namespace core {
class Mat {
 public:
  Mat() noexcept;
  Mat(const std::string& filename);

  Mat(const int rows, const int cols, const int channels, const float value);

  Mat(const Mat& other)
      : data_ptr_(std::make_unique<float[]>(other.size())),
        rows_(other.rows_),
        cols_(other.cols_),
        channels_(other.channels_) {
    std::copy(other.data_ptr_.get(), other.data_ptr_.get() + size(),
              data_ptr_.get());
  }

  Mat(Mat&& other) noexcept = default;
  Mat& operator=(Mat&& other) noexcept = default;

  Mat& operator=(const Mat& other) {
    if (this != &other) {
      rows_ = other.rows_;
      cols_ = other.cols_;
      channels_ = other.channels_;
      data_ptr_ = std::make_unique<float[]>(other.size());
      std::copy(other.data_ptr_.get(), other.data_ptr_.get() + size(),
                data_ptr_.get());
    }
    return *this;
  }

  ~Mat() = default;

  ::core::v1::Mat to_proto() const {
    ::core::v1::Mat proto;
    proto.set_rows(rows_);
    proto.set_cols(cols_);
    proto.set_channels(channels_);

    auto bytes = std::as_bytes(std::span<const float>(data_ptr_.get(), size()));
    proto.set_data(bytes.data(), static_cast<int>(bytes.size()));
    return proto;
  }

  void from_proto(const ::core::v1::Mat& proto) {
    rows_ = proto.rows();
    cols_ = proto.cols();
    channels_ = proto.channels();

    size_t total_size = size();
    data_ptr_ = std::make_unique<float[]>(total_size);

    const std::string& byte_data = proto.data();
    if (byte_data.size() != total_size * sizeof(float)) {
      throw std::runtime_error("Proto data size mismatch");
    }
    std::memcpy(data_ptr_.get(), byte_data.data(), byte_data.size());
  }

  float* data() { return data_ptr_.get(); }
  const float* data() const { return data_ptr_.get(); }

  size_t calculate_index(int row, int col, int channel) const noexcept {
    return static_cast<size_t>(row) * cols_ * channels_ +
           static_cast<size_t>(col) * channels_ + static_cast<size_t>(channel);
  }

  Mat clone() const {
    Mat copy(rows_, cols_, channels_, 0.0f);
    std::copy(data_ptr_.get(), data_ptr_.get() + size(), copy.data_ptr_.get());
    return copy;
  }

  bool oob(int row, int col, int channel) const noexcept {
    return row < 0 || row >= rows_ || col < 0 || col >= cols_ || channel < 0 ||
           channel >= channels_;
  }

  // accessors for pixel values
  float& operator()(const int row, const int col, const int channel) {
    if (oob(row, col, channel)) {
      throw std::runtime_error("Out of bounds access");
    }
    return data_ptr_.get()[calculate_index(row, col, channel)];
  }
  const float& operator()(const int row, const int col,
                          const int channel) const {
    if (oob(row, col, channel)) {
      throw std::runtime_error("Out of bounds access");
    }
    return data_ptr_.get()[calculate_index(row, col, channel)];
  }

  float& operator()(const int row, const int col) {
    if (channels_ != 1) {
      throw std::runtime_error(
          "This operator is only valid for single-channel matrices.");
    }
    return this->operator()(row, col, 0);
  }
  const float& operator()(const int row, const int col) const {
    if (channels_ != 1) {
      throw std::runtime_error(
          "This operator is only valid for single-channel matrices.");
    }
    return this->operator()(row, col, 0);
  }

  // operators
  bool operator==(const Mat& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_ ||
        channels_ != other.channels_) {
      return false;
    }
    for (int h = 0; h < rows_; ++h) {
      for (int w = 0; w < cols_; ++w) {
        for (int c = 0; c < channels_; ++c) {
          if (this->operator()(h, w, c) != other(h, w, c)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  bool operator!=(const Mat& other) const { return !(*this == other); }

  Mat operator+(const float value) const {
    Mat result(rows_, cols_, channels_, 0.0f);
    for (int h = 0; h < rows_; ++h) {
      for (int w = 0; w < cols_; ++w) {
        for (int c = 0; c < channels_; ++c) {
          result(h, w, c) = this->operator()(h, w, c) + value;
        }
      }
    }
    return result;
  }

  Mat operator+(const Mat& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_ ||
        channels_ != other.channels_) {
      throw std::runtime_error("Incompatible matrix dimensions");
    }
    Mat result(rows_, cols_, channels_, 0.0f);
    for (int h = 0; h < rows_; ++h) {
      for (int w = 0; w < cols_; ++w) {
        for (int c = 0; c < channels_; ++c) {
          result(h, w, c) = this->operator()(h, w, c) + other(h, w, c);
        }
      }
    }
    return result;
  }

  Mat operator*(float scalar) const {
    Mat result(rows_, cols_, channels_, 0.0f);
    for (int h = 0; h < rows_; ++h) {
      for (int w = 0; w < cols_; ++w) {
        for (int c = 0; c < channels_; ++c) {
          result(h, w, c) = this->operator()(h, w, c) * scalar;
        }
      }
    }
    return result;
  }

  Mat operator*(double scalar) const {
    return (*this) * static_cast<float>(scalar);
  }

  Mat operator-(const Mat& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_ ||
        channels_ != other.channels_) {
      throw std::runtime_error("Incompatible matrix dimensions");
    }
    return *this + (other * -1.0f);
  }

  Mat operator-() const { return *this * -1.0f; }

  int rows() const noexcept { return rows_; }
  int cols() const noexcept { return cols_; }
  int channels() const noexcept { return channels_; }
  size_t size() const noexcept {
    return static_cast<size_t>(rows_) * cols_ * channels_;
  }

 private:
  std::unique_ptr<float[]> data_ptr_;
  int rows_, cols_, channels_;
};

static inline Mat imread(const std::string& filename) {
  Mat img(filename);
  if (!img.data()) {
    throw std::runtime_error("Failed to read image");
  }
  return img;
}

static inline bool imwrite(const std::string& filename, const Mat& mat) {
  if (filename.empty() || mat.size() == 0) {
    return false;
  }

  std::unique_ptr<unsigned char[]> scaled =
      std::make_unique<unsigned char[]>(sizeof(unsigned char) * mat.size());
  for (size_t i = 0; i < mat.size(); ++i) {
    scaled[i] = static_cast<unsigned char>(mat.data()[i] * 255.0f);
  }

  int result = stbi_write_png(filename.c_str(), mat.cols(), mat.rows(),
                              mat.channels(), scaled.get(), 0);
  if (result == 0) {
    return false;
  }
  return true;
}

static inline Mat ones(const int rows, const int cols, const int channels) {
  Mat mat(rows, cols, channels, 1.0f);
  return mat;
}
static inline Mat ones(const int rows, const int cols) {
  return ones(rows, cols, 1);
}

static inline Mat zeros(const int rows, const int cols, const int channels) {
  Mat mat(rows, cols, channels, 0.0f);
  return mat;
}
static inline Mat zeros(const int rows, const int cols) {
  return zeros(rows, cols, 1);
}

static inline Mat operator*(float scalar, const Mat& mat) {
  return mat * scalar;
}

static inline Mat operator*(double scalar, const Mat& mat) {
  return mat * scalar;
}

};  // namespace core
