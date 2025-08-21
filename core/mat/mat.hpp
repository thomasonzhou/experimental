#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <expected>
#include <memory>
#include <optional>
#include <string>

namespace core {

enum class MatError {
  InvalidDimensions,
  InvalidFilename,
  ImageLoadFailed,
  OutOfBounds,
  IncompatibleDimensions,
  ProtoDataMismatch,
  InvalidChannelsForOperation,
  WriteImageFailed,
};

enum class MatLayout { HWC, CHW };

struct MatShape {
  size_t rows;
  size_t cols;
  size_t channels;
};

static constexpr bool approx_equal(const float a, const float b,
                                   const float epsilon = 1e-6f) {
  return std::fabs(a - b) < epsilon;
}

class Mat {
 public:
  Mat() noexcept;
  Mat(const MatShape shape, const std::optional<float> value = std::nullopt,
      const std::optional<MatLayout> layout = std::nullopt) noexcept;
  Mat(const size_t rows, const size_t cols, const size_t channels,
      const std::optional<float> value = std::nullopt,
      const std::optional<MatLayout> layout = std::nullopt) noexcept;

  // rule of five
  ~Mat() = default;
  Mat(const Mat& other);
  Mat(Mat&& other) noexcept = default;
  Mat& operator=(const Mat& other);
  Mat& operator=(Mat&& other) noexcept = default;

  [[nodiscard]] Mat clone() const noexcept;

  [[nodiscard]] constexpr size_t rows() const noexcept { return shape_.rows; }
  [[nodiscard]] constexpr size_t cols() const noexcept { return shape_.cols; }
  [[nodiscard]] constexpr size_t channels() const noexcept {
    return shape_.channels;
  }
  [[nodiscard]] constexpr size_t size() const noexcept {
    return shape_.rows * shape_.cols * shape_.channels;
  }
  [[nodiscard]] constexpr MatShape shape() const noexcept { return shape_; }
  [[nodiscard]] constexpr MatLayout layout() const noexcept { return layout_; }

  [[nodiscard]] float* data() noexcept { return data_ptr_.get(); }
  [[nodiscard]] const float* data() const noexcept { return data_ptr_.get(); }

  [[nodiscard]] constexpr bool oob(size_t row, size_t col,
                                   size_t channel) const noexcept {
    return row >= shape_.rows || col >= shape_.cols ||
           channel >= shape_.channels;
  }

  [[nodiscard]] size_t calculate_index(size_t row, size_t col,
                                       size_t channel) const noexcept {
    return index_fn_(row, col, channel, shape_);
  }

  [[nodiscard]] Mat to_layout(MatLayout target_layout) const noexcept;

  // unsafe direct access (no bounds checking)
  float& operator()(const size_t row, const size_t col) noexcept {
    return data_ptr_.get()[calculate_index(row, col, 0)];
  }
  float operator()(const size_t row, const size_t col) const noexcept {
    return data_ptr_.get()[calculate_index(row, col, 0)];
  }
  float& operator()(const size_t row, const size_t col,
                    const size_t channel) noexcept {
    return data_ptr_.get()[calculate_index(row, col, channel)];
  }
  float operator()(const size_t row, const size_t col,
                   const size_t channel) const noexcept {
    return data_ptr_.get()[calculate_index(row, col, channel)];
  }

  [[nodiscard]] std::expected<std::reference_wrapper<float>, MatError> at(
      const size_t row, const size_t col, const size_t channel) {
    if (oob(row, col, channel)) {
      return std::unexpected(MatError::OutOfBounds);
    }
    return std::ref(data_ptr_.get()[calculate_index(row, col, channel)]);
  }
  [[nodiscard]] std::expected<float, MatError> at(const size_t row,
                                                  const size_t col,
                                                  const size_t channel) const {
    if (oob(row, col, channel)) {
      return std::unexpected(MatError::OutOfBounds);
    }
    return data_ptr_.get()[calculate_index(row, col, channel)];
  }
  [[nodiscard]] std::expected<std::reference_wrapper<float>, MatError> at(
      const size_t row, const size_t col) {
    if (shape_.channels != 1) {
      return std::unexpected(MatError::InvalidChannelsForOperation);
    }
    return at(row, col, 0);
  }
  [[nodiscard]] std::expected<float, MatError> at(const size_t row,
                                                  const size_t col) const {
    if (shape_.channels != 1) {
      return std::unexpected(MatError::InvalidChannelsForOperation);
    }
    return at(row, col, 0);
  }

  // comparison operators
  [[nodiscard]] bool operator==(const Mat& other) const;
  [[nodiscard]] bool operator!=(const Mat& other) const {
    return !(*this == other);
  }

  // unchecked arithmetic operations
  [[nodiscard]] Mat operator+(const Mat& other) const noexcept;
  [[nodiscard]] Mat operator-(const Mat& other) const noexcept;

  [[nodiscard]] Mat operator*(const float scalar) const noexcept;
  [[nodiscard]] Mat operator/(const float scalar) const noexcept;
  [[nodiscard]] Mat operator+(const float scalar) const noexcept;
  [[nodiscard]] Mat operator-(const float scalar) const noexcept;
  [[nodiscard]] Mat operator-() const noexcept {
    return operator*(static_cast<float>(-1));
  }

  [[nodiscard]] Mat operator*(const double scalar) const noexcept {
    return operator*(static_cast<float>(scalar));
  }
  [[nodiscard]] Mat operator/(const double scalar) const noexcept {
    return operator/(static_cast<float>(scalar));
  }
  [[nodiscard]] Mat operator+(const double scalar) const noexcept {
    return operator+(static_cast<float>(scalar));
  }
  [[nodiscard]] Mat operator-(const double scalar) const noexcept {
    return operator-(static_cast<float>(scalar));
  }

  // DON'T CROSS THIS LINE (•̀ᴗ•́)و ̑̑
 private:
  std::unique_ptr<float[]> data_ptr_;
  MatShape shape_;
  MatLayout layout_;

  static size_t hwc_index(size_t row, size_t col, size_t channel,
                          const MatShape& shape) noexcept {
    return row * shape.cols * shape.channels + col * shape.channels + channel;
  }
  static size_t chw_index(size_t row, size_t col, size_t channel,
                          const MatShape& shape) noexcept {
    return channel * shape.rows * shape.cols + row * shape.cols + col;
  }

  using IndexFunction = size_t (*)(size_t row, size_t col, size_t channel,
                                   const MatShape& shape) noexcept;
  IndexFunction index_fn_ = &Mat::hwc_index;
};

[[nodiscard]] Mat operator*(const float scalar, const Mat& mat) noexcept;

template <typename T>
[[nodiscard]] inline Mat operator*(const T scalar, const Mat& mat) noexcept {
  return operator*(static_cast<float>(scalar), mat);
}

};  // namespace core
