#include "mat.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace core {

Mat::Mat() noexcept : data_ptr_(nullptr), shape_{0, 0, 0} {};

Mat::Mat(const MatShape shape, const std::optional<float> value,
         const std::optional<MatLayout> layout) noexcept
    : data_ptr_(
          std::make_unique<float[]>(shape.rows * shape.cols * shape.channels)),
      shape_(shape),
      layout_(layout.value_or(MatLayout::HWC)) {
  if (value) {
    std::fill(data_ptr_.get(), data_ptr_.get() + size(), *value);
  }
  index_fn_ = (layout_ == MatLayout::CHW) ? chw_index : hwc_index;
}

Mat::Mat(const size_t rows, const size_t cols, const size_t channels,
         const std::optional<float> value,
         const std::optional<MatLayout> layout) noexcept
    : data_ptr_(std::make_unique<float[]>(rows * cols * channels)),
      shape_{rows, cols, channels},
      layout_(layout.value_or(MatLayout::HWC)) {
  if (value) {
    std::fill(data_ptr_.get(), data_ptr_.get() + size(), *value);
  }
  index_fn_ = (layout_ == MatLayout::CHW) ? chw_index : hwc_index;
}

Mat::Mat(const Mat& other)
    : data_ptr_(std::make_unique<float[]>(other.size())),
      shape_(other.shape_),
      layout_(other.layout_) {
  std::copy(other.data(), other.data() + size(), data());
}

Mat& Mat::operator=(const Mat& other) {
  if (this != &other) {
    shape_ = other.shape_;
    layout_ = other.layout_;
    data_ptr_ = std::make_unique<float[]>(other.size());
    std::copy(other.data(), other.data() + size(), data());
  }
  return *this;
}

Mat Mat::clone() const noexcept {
  Mat copy(shape_, std::nullopt, layout_);
  std::copy(data(), data() + size(), copy.data());
  return copy;
}

Mat Mat::to_layout(MatLayout target_layout) const noexcept {
  if (layout_ == target_layout) {
    return clone();
  }

  Mat result(shape_, std::nullopt, target_layout);
  for (size_t r = 0; r < shape_.rows; ++r) {
    for (size_t c = 0; c < shape_.cols; ++c) {
      for (size_t ch = 0; ch < shape_.channels; ++ch) {
        result(r, c, ch) = (*this)(r, c, ch);
      }
    }
  }
  return result;
}

bool Mat::operator==(const Mat& other) const {
  if (shape_.rows != other.shape_.rows || shape_.cols != other.shape_.cols ||
      shape_.channels != other.shape_.channels) {
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
    for (size_t r = 0; r < shape_.rows; ++r) {
      for (size_t c = 0; c < shape_.cols; ++c) {
        for (size_t ch = 0; ch < shape_.channels; ++ch) {
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
  Mat result(shape_);
  for (size_t i = 0; i < size(); ++i) {
    result.data()[i] = data()[i] * scalar;
  }
  return result;
}

Mat Mat::operator/(const float scalar) const noexcept {
  Mat result(shape_);
  for (size_t i = 0; i < size(); ++i) {
    result.data()[i] = data()[i] / scalar;
  }
  return result;
}

Mat Mat::operator+(const float scalar) const noexcept {
  Mat result(shape_);
  for (size_t i = 0; i < size(); ++i) {
    result.data()[i] = data()[i] + scalar;
  }
  return result;
}

Mat Mat::operator-(const float scalar) const noexcept {
  Mat result(shape_);
  for (size_t i = 0; i < size(); ++i) {
    result.data()[i] = data()[i] - scalar;
  }
  return result;
}

Mat operator*(const float scalar, const Mat& mat) noexcept {
  return mat * scalar;
}

};  // namespace core
