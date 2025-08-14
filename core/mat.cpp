#include "mat.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace core {

Mat::Mat() noexcept : data_ptr_(nullptr), rows_(0), cols_(0), channels_(0) {};

Mat::Mat(const size_t rows, const size_t cols, const size_t channels,
         const std::optional<float> value) noexcept
    : data_ptr_(std::make_unique<float[]>(rows * cols * channels)),
      rows_(rows),
      cols_(cols),
      channels_(channels) {
  if (value) {
    std::fill(data_ptr_.get(), data_ptr_.get() + size(), *value);
  }
}

Mat::Mat(const Mat& other)
    : data_ptr_(std::make_unique<float[]>(other.size())),
      rows_(other.rows_),
      cols_(other.cols_),
      channels_(other.channels_) {
  std::copy(other.data(), other.data() + size(), data());
}

Mat& Mat::operator=(const Mat& other) {
  if (this != &other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    channels_ = other.channels_;
    data_ptr_ = std::make_unique<float[]>(other.size());
    std::copy(other.data(), other.data() + size(), data());
  }
  return *this;
}

Mat Mat::clone() const noexcept {
  Mat copy(rows_, cols_, channels_);
  std::copy(data(), data() + size(), copy.data());
  return copy;
}

bool Mat::operator==(const Mat& other) const {
  if (rows_ != other.rows_ || cols_ != other.cols_ ||
      channels_ != other.channels_) {
    return false;
  }

  for (size_t i = 0; i < size(); ++i) {
    if (!approx_equal(data()[i], other.data()[i])) {
      return false;
    }
  }
  return true;
}

// why support these?
Mat Mat::operator+(const Mat& other) const noexcept {
  Mat result(rows_, cols_, channels_);
  for (size_t i = 0; i < size(); ++i) {
    result.data()[i] = data()[i] + other.data()[i];
  }
  return result;
}

Mat Mat::operator-(const Mat& other) const noexcept {
  Mat result(rows_, cols_, channels_);
  for (size_t i = 0; i < size(); ++i) {
    result.data()[i] = data()[i] - other.data()[i];
  }
  return result;
}

Mat Mat::operator*(const float scalar) const noexcept {
  Mat result(rows_, cols_, channels_);
  for (size_t i = 0; i < size(); ++i) {
    result.data()[i] = data()[i] * scalar;
  }
  return result;
}

Mat Mat::operator/(const float scalar) const noexcept {
  Mat result(rows_, cols_, channels_);
  for (size_t i = 0; i < size(); ++i) {
    result.data()[i] = data()[i] / scalar;
  }
  return result;
}

Mat Mat::operator+(const float scalar) const noexcept {
  Mat result(rows_, cols_, channels_);
  for (size_t i = 0; i < size(); ++i) {
    result.data()[i] = data()[i] + scalar;
  }
  return result;
}

Mat Mat::operator-(const float scalar) const noexcept {
  Mat result(rows_, cols_, channels_);
  for (size_t i = 0; i < size(); ++i) {
    result.data()[i] = data()[i] - scalar;
  }
  return result;
}

Mat operator*(const float scalar, const Mat& mat) noexcept {
  return mat * scalar;
}

};  // namespace core
