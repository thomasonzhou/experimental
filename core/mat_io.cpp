#include "core/mat_io.hpp"

namespace core {

::core::v1::Mat to_proto(const Mat &mat) noexcept {
  ::core::v1::Mat proto;
  proto.set_rows(mat.rows());
  proto.set_cols(mat.cols());
  proto.set_channels(mat.channels());

  auto bytes = std::as_bytes(std::span<const float>(mat.data(), mat.size()));
  proto.set_data(bytes.data(), static_cast<int>(bytes.size()));
  return proto;
}

std::expected<Mat, MatError> from_proto(const ::core::v1::Mat &proto) {
  Mat mat(proto.rows(), proto.cols(), proto.channels());

  const std::string &byte_data = proto.data();
  if (byte_data.size() != mat.size() * sizeof(float)) {
    return std::unexpected(MatError::ProtoDataMismatch);
  }
  std::memcpy(mat.data(), byte_data.data(), byte_data.size());
  return mat;
}

std::expected<Mat, MatError> imread(const std::string &filename) {
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

  Mat mat(static_cast<size_t>(rows), static_cast<size_t>(cols),
          static_cast<size_t>(channels));

  for (size_t i = 0; i < mat.size(); ++i) {
    mat.data()[i] = static_cast<float>(img_data[i]) / 255.0f;
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
  for (size_t i = 0; i < mat.size(); ++i) {
    scaled[i] = static_cast<unsigned char>(
        std::clamp(mat.data()[i], 0.0f, 1.0f) * 255.0f);
  }

  int result = stbi_write_png(filename.c_str(), mat.cols(), mat.rows(),
                              mat.channels(), scaled.get(), 0);
  if (result == 0) {
    return std::unexpected(MatError::WriteImageFailed);
  }
  return {};
}

Mat ones(const size_t rows, const size_t cols, const size_t channels) noexcept {
  return Mat(rows, cols, channels, 1.0f);
}

Mat zeros(const size_t rows, const size_t cols,
          const size_t channels) noexcept {
  return Mat(rows, cols, channels, 0.0f);
}
};  // namespace core
