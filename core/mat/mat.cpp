#include "mat.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace core {

Mat::Mat() noexcept : data_ptr_(nullptr), rows_(0), cols_(0), channels_(0) {};

Mat::Mat(const size_t rows, const size_t cols, const size_t channels,
         const std::optional<float> value,
         const std::optional<MatLayout> layout) noexcept
    : data_ptr_(std::make_unique<float[]>(rows * cols * channels)),
      rows_(rows),
      cols_(cols),
      channels_(channels),
      layout_(layout.value_or(MatLayout::HWC)) {
  if (value) {
    std::fill(data_ptr_.get(), data_ptr_.get() + size(), *value);
  }
  index_fn_ = (layout_ == MatLayout::CHW) ? chw_index : hwc_index;
}

Mat::Mat(const Mat& other)
    : data_ptr_(std::make_unique<float[]>(other.size())),
      rows_(other.rows_),
      cols_(other.cols_),
      channels_(other.channels_),
      layout_(other.layout_) {
  std::copy(other.data(), other.data() + size(), data());
}

Mat& Mat::operator=(const Mat& other) {
  if (this != &other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    channels_ = other.channels_;
    layout_ = other.layout_;
    data_ptr_ = std::make_unique<float[]>(other.size());
    std::copy(other.data(), other.data() + size(), data());
  }
  return *this;
}

Mat Mat::clone() const noexcept {
  Mat copy(rows_, cols_, channels_, std::nullopt, layout_);
  std::copy(data(), data() + size(), copy.data());
  return copy;
}

Mat Mat::to_layout(MatLayout target_layout) const noexcept {
  if (layout_ == target_layout) {
    return clone();
  }

  Mat result(rows_, cols_, channels_, std::nullopt, target_layout);
  for (size_t r = 0; r < rows_; ++r) {
    for (size_t c = 0; c < cols_; ++c) {
      for (size_t ch = 0; ch < channels_; ++ch) {
        result(r, c, ch) = (*this)(r, c, ch);
      }
    }
  }
  return result;
}

bool Mat::operator==(const Mat& other) const {
  if (rows_ != other.rows_ || cols_ != other.cols_ ||
      channels_ != other.channels_) {
    return false;
  }
  // consider different layout to be equal if values are the same
  if (layout_ == other.layout_) {
    for (size_t i = 0; i < size(); ++i) {
      if (!approx_equal(data()[i], other.data()[i])) {
        return false;
      }
    }
  } else {
    for (size_t r = 0; r < rows_; ++r) {
      for (size_t c = 0; c < cols_; ++c) {
        for (size_t ch = 0; ch < channels_; ++ch) {
          if (!approx_equal((*this)(r, c, ch), other(r, c, ch))) {
            return false;
          }
        }
      }
    }
  }
  return true;
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
