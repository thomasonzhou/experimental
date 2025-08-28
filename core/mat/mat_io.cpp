#include "core/mat/mat_io.hpp"

#include <stb_image.h>
#include <stb_image_write.h>

#include <cmath>

namespace core {

::core::v1::MatLayout to_proto(const MatLayout layout) noexcept {
  switch (layout) {
    case MatLayout::NHWC:
      return ::core::v1::MatLayout::MAT_LAYOUT_NHWC;
    case MatLayout::NCHW:
      return ::core::v1::MatLayout::MAT_LAYOUT_NCHW;
    default:
      return ::core::v1::MatLayout::MAT_LAYOUT_UNSPECIFIED;
  }
}

::core::v1::Mat to_proto(const Mat &mat) noexcept {
  ::core::v1::Mat proto;
  proto.set_batch_size(mat.batch_size());
  proto.set_rows(mat.rows());
  proto.set_cols(mat.cols());
  proto.set_channels(mat.channels());
  proto.set_layout(to_proto(mat.layout()));

  auto bytes = std::as_bytes(std::span<const float>(mat.data(), mat.size()));
  proto.set_data(bytes.data(), static_cast<int>(bytes.size()));
  return proto;
}

MatLayout from_proto(const ::core::v1::MatLayout layout) noexcept {
  switch (layout) {
    case ::core::v1::MatLayout::MAT_LAYOUT_NHWC:
      return MatLayout::NHWC;
    case ::core::v1::MatLayout::MAT_LAYOUT_NCHW:
      return MatLayout::NCHW;
    default:
      return MatLayout::NHWC;
  }
}

std::expected<Mat, MatError> from_proto(const ::core::v1::Mat &proto) {
  Mat mat(MatShape::make_4d(proto.batch_size(), proto.rows(), proto.cols(),
                            proto.channels()),
          std::nullopt, from_proto(proto.layout()));

  const std::string &byte_data = proto.data();
  if (byte_data.size() != mat.size() * sizeof(float)) {
    return std::unexpected(MatError::ProtoDataMismatch);
  }
  std::memcpy(mat.data(), byte_data.data(), byte_data.size());
  return mat;
}

std::expected<Mat, MatError> imread(const std::string &filename,
                                    std::optional<MatLayout> layout) {
  if (filename.empty()) {
    return std::unexpected(MatError::InvalidFilename);
  }

  int rows, cols, channels;
  constexpr int kKeepChannels = 0;
  unsigned char *img_data =
      stbi_load(filename.c_str(), &cols, &rows, &channels, kKeepChannels);

  if (!img_data) {
    return std::unexpected(MatError::ImageLoadFailed);
  }

  Mat mat(
      MatShape::make_3d(static_cast<size_t>(rows), static_cast<size_t>(cols),
                        static_cast<size_t>(channels)),
      std::nullopt, layout);

  if (mat.layout() == MatLayout::NHWC) {
    for (size_t i = 0; i < mat.size(); ++i) {
      mat.data()[i] = static_cast<float>(img_data[i]) / 255.0f;
    }
  } else {
    for (size_t r = 0; r < mat.rows(); ++r) {
      for (size_t c = 0; c < mat.cols(); ++c) {
        for (size_t ch = 0; ch < mat.channels(); ++ch) {
          mat(r, c, ch) =
              static_cast<float>(
                  img_data[(r * mat.cols() + c) * mat.channels() + ch]) /
              255.0f;
        }
      }
    }
  }

  stbi_image_free(img_data);
  return mat;
}

std::expected<void, MatError> imwrite(const std::string &filename,
                                      const Mat &mat) {
  // assumes range of 0.0 to 1.0
  if (filename.empty() || mat.size() == 0) {
    return std::unexpected(MatError::InvalidFilename);
  }
  if (mat.channels() < 1 || mat.channels() > 4) {
    return std::unexpected(MatError::InvalidChannelsForOperation);
  }

  std::unique_ptr<unsigned char[]> scaled =
      std::make_unique<unsigned char[]>(sizeof(unsigned char) * mat.size());

  if (mat.layout() == MatLayout::NHWC) {
    for (size_t i = 0; i < mat.size(); ++i) {
      scaled[i] = static_cast<unsigned char>(
          std::clamp(mat.data()[i], 0.0f, 1.0f) * 255.0f);
    }
  } else {
    for (size_t r = 0; r < mat.rows(); ++r) {
      for (size_t c = 0; c < mat.cols(); ++c) {
        for (size_t ch = 0; ch < mat.channels(); ++ch) {
          scaled[(r * mat.cols() + c) * mat.channels() + ch] =
              static_cast<unsigned char>(std::clamp(mat(r, c, ch), 0.0f, 1.0f) *
                                         255.0f);
        }
      }
    }
  }

  int result = stbi_write_png(filename.c_str(), mat.cols(), mat.rows(),
                              mat.channels(), scaled.get(), 0);
  if (result == 0) {
    return std::unexpected(MatError::WriteImageFailed);
  }
  return {};
}

Mat ones(const MatShape shape) noexcept { return Mat(shape, 1.0f); }

Mat zeros(const MatShape shape) noexcept { return Mat(shape, 0.0f); }

Mat gaussian(const MatShape shape, const std::optional<float> sigma_rows,
             const std::optional<float> sigma_cols) noexcept {
  const size_t rows = shape.rows;
  const size_t cols = shape.cols;
  const size_t channels = shape.channels;

  const float mean_x = static_cast<float>(cols - 1) / 2.0f;
  const float mean_y = static_cast<float>(rows - 1) / 2.0f;
  constexpr float kDefaultSigmaFactor = 0.25f;
  const float sigma_x = sigma_cols.value_or(cols * kDefaultSigmaFactor);
  const float sigma_y = sigma_rows.value_or(rows * kDefaultSigmaFactor);

  const float inv2sx2 = 1.0f / (2.0f * sigma_x * sigma_x);
  const float inv2sy2 = 1.0f / (2.0f * sigma_y * sigma_y);

  Mat mat(shape);
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      float res_x = static_cast<float>(c) - mean_x;
      float res_y = static_cast<float>(r) - mean_y;
      float value = std::exp(-(res_x * res_x) * inv2sx2) *
                    std::exp(-(res_y * res_y) * inv2sy2);
      for (size_t ch = 0; ch < channels; ++ch) {
        mat(r, c, ch) = value;
      }
    }
  }
  return mat;
}

};  // namespace core
