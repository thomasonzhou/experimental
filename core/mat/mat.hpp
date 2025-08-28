#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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

enum class MatLayout { NCHW, NHWC };

struct MatShape {
  size_t batch_size = 1;
  size_t rows;
  size_t cols;
  size_t channels;

  [[nodiscard]] constexpr size_t ndim() const noexcept {
    return (batch_size > 1) ? 4 : 3;
  }

  [[nodiscard]] constexpr size_t size() const noexcept {
    return batch_size * rows * cols * channels;
  }

  static constexpr MatShape make_3d(size_t rows, size_t cols, size_t channels) {
    return {1, rows, cols, channels};
  }

  static constexpr MatShape make_4d(size_t batch_size, size_t rows, size_t cols,
                                    size_t channels) {
    return {batch_size, rows, cols, channels};
  }
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

  Mat(const std::vector<size_t>& dims,
      const std::optional<float> value = std::nullopt,
      const std::optional<MatLayout> layout = std::nullopt) noexcept;

  // rule of five
  ~Mat() = default;
  Mat(const Mat& other);
  Mat(Mat&& other) noexcept = default;
  Mat& operator=(const Mat& other);
  Mat& operator=(Mat&& other) noexcept = default;

  [[nodiscard]] Mat clone() const noexcept;

  [[nodiscard]] constexpr size_t batch_size() const noexcept {
    return shape_.batch_size;
  }
  [[nodiscard]] constexpr size_t rows() const noexcept { return shape_.rows; }
  [[nodiscard]] constexpr size_t cols() const noexcept { return shape_.cols; }
  [[nodiscard]] constexpr size_t channels() const noexcept {
    return shape_.channels;
  }
  [[nodiscard]] constexpr size_t ndim() const noexcept { return shape_.ndim(); }
  [[nodiscard]] constexpr size_t size() const noexcept { return shape_.size(); }
  [[nodiscard]] constexpr MatShape shape() const noexcept { return shape_; }
  [[nodiscard]] std::vector<size_t> shape_vec() const noexcept {
    if (ndim() == 4) {
      return {shape_.batch_size, shape_.channels, shape_.rows, shape_.cols};
    } else {
      return {shape_.rows, shape_.cols, shape_.channels};
    }
  }
  [[nodiscard]] constexpr MatLayout layout() const noexcept { return layout_; }

  [[nodiscard]] float* data() noexcept { return data_ptr_.get(); }
  [[nodiscard]] const float* data() const noexcept { return data_ptr_.get(); }

  [[nodiscard]] constexpr bool oob(size_t batch, size_t channel, size_t row,
                                   size_t col) const noexcept {
    return batch >= shape_.batch_size || channel >= shape_.channels ||
           row >= shape_.rows || col >= shape_.cols;
  }

  [[nodiscard]] size_t calculate_index(size_t row, size_t col,
                                       size_t channel) const noexcept {
    return index_fn_(0, channel, row, col, shape_);  // batch=0 for 3D access
  }

  [[nodiscard]] size_t calculate_index(size_t batch, size_t channel, size_t row,
                                       size_t col) const noexcept {
    return index_fn_(batch, channel, row, col, shape_);
  }

  [[nodiscard]] Mat to_layout(MatLayout target_layout) const noexcept;

  // 2D unsafe direct access (no bounds checking)
  float& operator()(const size_t row, const size_t col) noexcept {
    return data_ptr_.get()[calculate_index(row, col, 0)];
  }
  float operator()(const size_t row, const size_t col) const noexcept {
    return data_ptr_.get()[calculate_index(row, col, 0)];
  }

  // 3D unsafe direct access (no bounds checking)
  float& operator()(const size_t row, const size_t col,
                    const size_t channel) noexcept {
    return data_ptr_.get()[calculate_index(row, col, channel)];
  }
  float operator()(const size_t row, const size_t col,
                   const size_t channel) const noexcept {
    return data_ptr_.get()[calculate_index(row, col, channel)];
  }

  // 4D unsafe direct access (no bounds checking)
  float& operator()(const size_t batch, const size_t channel, const size_t row,
                    const size_t col) noexcept {
    return data_ptr_.get()[calculate_index(batch, channel, row, col)];
  }
  float operator()(const size_t batch, const size_t channel, const size_t row,
                   const size_t col) const noexcept {
    return data_ptr_.get()[calculate_index(batch, channel, row, col)];
  }

  // safe accessors
  [[nodiscard]] std::expected<std::reference_wrapper<float>, MatError> at(
      const size_t batch, const size_t channel, const size_t row,
      const size_t col) {
    if (oob(batch, channel, row, col)) {
      return std::unexpected(MatError::OutOfBounds);
    }
    return std::ref(data_ptr_.get()[calculate_index(batch, channel, row, col)]);
  }
  [[nodiscard]] std::expected<float, MatError> at(const size_t batch,
                                                  const size_t channel,
                                                  const size_t row,
                                                  const size_t col) const {
    if (oob(batch, channel, row, col)) {
      return std::unexpected(MatError::OutOfBounds);
    }
    return data_ptr_.get()[calculate_index(batch, channel, row, col)];
  }
  [[nodiscard]] std::expected<std::reference_wrapper<float>, MatError> at(
      const size_t row, const size_t col, const size_t channel) {
    if (oob(0, channel, row, col)) {
      return std::unexpected(MatError::OutOfBounds);
    }
    return std::ref(data_ptr_.get()[calculate_index(row, col, channel)]);
  }
  [[nodiscard]] std::expected<float, MatError> at(const size_t row,
                                                  const size_t col,
                                                  const size_t channel) const {
    if (oob(0, channel, row, col)) {
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

  static size_t nchw_index(size_t batch, size_t channel, size_t row, size_t col,
                           const MatShape& shape) noexcept {
    return batch * shape.rows * shape.cols * shape.channels +
           channel * shape.rows * shape.cols + row * shape.cols + col;
  }
  static size_t nhwc_index(size_t batch, size_t channel, size_t row, size_t col,
                           const MatShape& shape) noexcept {
    return batch * shape.rows * shape.cols * shape.channels +
           row * shape.cols * shape.channels + col * shape.channels + channel;
  }

  using IndexFunction = size_t (*)(size_t batch, size_t channel, size_t row,
                                   size_t col, const MatShape& shape) noexcept;

  IndexFunction index_fn_ = &Mat::nhwc_index;
};

[[nodiscard]] Mat operator*(const float scalar, const Mat& mat) noexcept;

template <typename T>
[[nodiscard]] inline Mat operator*(const T scalar, const Mat& mat) noexcept {
  return operator*(static_cast<float>(scalar), mat);
}

};  // namespace core
