#pragma once

// utility functions for Mat class

#include <expected>
#include <optional>
#include <string>

#include "core/mat/mat.hpp"
#include "core/mat/mat.pb.h"

namespace core {
[[nodiscard]] ::core::v1::Mat to_proto(const Mat &mat) noexcept;
[[nodiscard]] std::expected<Mat, MatError> from_proto(
    const ::core::v1::Mat &proto);

[[nodiscard]] std::expected<Mat, MatError> imread(const std::string &filename);
[[nodiscard]] std::expected<void, MatError> imwrite(const std::string &filename,
                                                    const Mat &mat);
[[nodiscard]] Mat ones(const size_t rows, const size_t cols,
                       const size_t channels = 1) noexcept;
[[nodiscard]] Mat zeros(const size_t rows, const size_t cols,
                        const size_t channels = 1) noexcept;

[[nodiscard]] Mat gaussian(
    const size_t rows, const size_t cols, const size_t channels = 1,
    const std::optional<float> sigma_rows = std::nullopt,
    const std::optional<float> sigma_cols = std::nullopt) noexcept;
};  // namespace core
