#pragma once

// utility functions for Mat class

#include <expected>
#include <optional>
#include <string>

#include "core/mat/mat.hpp"
#include "core/mat/mat.pb.h"

namespace core {
[[nodiscard]] ::core::v1::MatLayout to_proto(const MatLayout layout) noexcept;
[[nodiscard]] ::core::v1::Mat to_proto(const Mat &mat) noexcept;
[[nodiscard]] MatLayout from_proto(const ::core::v1::MatLayout layout) noexcept;
[[nodiscard]] std::expected<Mat, MatError> from_proto(
    const ::core::v1::Mat &proto);

[[nodiscard]] std::expected<Mat, MatError> imread(
    const std::string &filename,
    const std::optional<MatLayout> layout = std::nullopt);
[[nodiscard]] std::expected<void, MatError> imwrite(const std::string &filename,
                                                    const Mat &mat);
[[nodiscard]] Mat ones(const MatShape shape) noexcept;
[[nodiscard]] Mat zeros(const MatShape shape) noexcept;

// Convenience helpers for 3D shapes
[[nodiscard]] inline Mat ones(const size_t rows, const size_t cols,
                              const size_t channels = 1) noexcept {
  return ones(MatShape::make_3d(rows, cols, channels));
}
[[nodiscard]] inline Mat zeros(const size_t rows, const size_t cols,
                               const size_t channels = 1) noexcept {
  return zeros(MatShape::make_3d(rows, cols, channels));
}

[[nodiscard]] Mat gaussian(
    const MatShape shape, const std::optional<float> sigma_rows = std::nullopt,
    const std::optional<float> sigma_cols = std::nullopt) noexcept;

// Convenience helper for 3D gaussian
[[nodiscard]] inline Mat gaussian(
    const size_t rows, const size_t cols, const size_t channels = 1,
    const std::optional<float> sigma_rows = std::nullopt,
    const std::optional<float> sigma_cols = std::nullopt) noexcept {
  return gaussian(MatShape::make_3d(rows, cols, channels), sigma_rows,
                  sigma_cols);
}
};  // namespace core
