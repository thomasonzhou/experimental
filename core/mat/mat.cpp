#include "mat.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace core {

Mat::Mat() noexcept : data_ptr_(nullptr), shape_{0, 0, 0, 1} {};

Mat::Mat(const MatShape shape, const std::optional<float> value,
         const std::optional<MatLayout> layout) noexcept
    : data_ptr_(std::make_unique<float[]>(shape.size())),
      shape_(shape),
      layout_(layout.value_or(shape.ndim() == 4 ? MatLayout::NCHW
                                                : MatLayout::NHWC)) {
  if (value) {
    std::fill(data_ptr_.get(), data_ptr_.get() + size(), *value);
  }

  // Set indexing function based on layout (works for both 3D and 4D)
  index_fn_ = (layout_ == MatLayout::NHWC) ? nhwc_index : nchw_index;
}

Mat::Mat(const std::vector<size_t>& dims, const std::optional<float> value,
         const std::optional<MatLayout> layout) noexcept {
  switch (dims.size()) {
    case 2:
      shape_ = {1, dims[0], dims[1], 1};
      break;
    case 3:
      shape_ = {1, dims[0], dims[1], dims[2]};
      break;
    case 4:
      shape_ = {dims[0], dims[1], dims[2], dims[3]};
      break;
    default:
      shape_ = {1, 1, 1, 1};
  }
  layout_ = MatLayout::NHWC;
  index_fn_ = (layout_ == MatLayout::NHWC) ? nhwc_index : nchw_index;

  data_ptr_ = std::make_unique<float[]>(shape_.size());
  if (value) {
    std::fill(data_ptr_.get(), data_ptr_.get() + size(), *value);
  }
}

Mat::Mat(const Mat& other)
    : data_ptr_(std::make_unique<float[]>(other.size())),
      shape_(other.shape_),
      layout_(other.layout_),
      index_fn_(other.index_fn_) {
  std::copy(other.data(), other.data() + size(), data());
}

Mat& Mat::operator=(const Mat& other) {
  if (this != &other) {
    shape_ = other.shape_;
    layout_ = other.layout_;
    index_fn_ = other.index_fn_;
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
  for (size_t b = 0; b < shape_.batch_size; ++b) {
    for (size_t c = 0; c < shape_.channels; ++c) {
      for (size_t r = 0; r < shape_.rows; ++r) {
        for (size_t col = 0; col < shape_.cols; ++col) {
          result(b, c, r, col) = (*this)(b, c, r, col);
        }
      }
    }
  }
  return result;
}

bool Mat::operator==(const Mat& other) const {
  if (shape_.batch_size != other.shape_.batch_size ||
      shape_.rows != other.shape_.rows || shape_.cols != other.shape_.cols ||
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
    for (size_t b = 0; b < shape_.batch_size; ++b) {
      for (size_t c = 0; c < shape_.channels; ++c) {
        for (size_t r = 0; r < shape_.rows; ++r) {
          for (size_t col = 0; col < shape_.cols; ++col) {
            if (!approx_equal((*this)(b, c, r, col), other(b, c, r, col))) {
              return false;
            }
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
