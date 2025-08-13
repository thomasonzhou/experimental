#include "mat.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace core {
static constexpr int KEEP_CHANNELS = 0;  // retain all channels from source

Mat::Mat() noexcept : data_ptr_(nullptr), rows_(0), cols_(0), channels_(0) {};

Mat::Mat(const std::string& filename) {
  if (filename.empty()) {
    std::cerr << "Error: No filename provided" << std::endl;
    exit(1);
  }

  unsigned char* img_data =
      stbi_load(filename.c_str(), &cols_, &rows_, &channels_, KEEP_CHANNELS);

  if (!img_data) {
    std::cerr << "Error: Failed to load image " << filename << std::endl;
    exit(1);
  }

  size_t total_size = static_cast<size_t>(rows_) * cols_ * channels_;
  data_ptr_ = std::make_unique<float[]>(total_size);

  for (size_t i = 0; i < total_size; ++i) {
    data_ptr_[i] = static_cast<float>(img_data[i]) / 255.0f;
  }

  stbi_image_free(img_data);
}

Mat::Mat(const int rows, const int cols, const int channels, const float value)
    : data_ptr_(std::make_unique<float[]>(static_cast<size_t>(rows) * cols *
                                          channels)),
      rows_(rows),
      cols_(cols),
      channels_(channels) {
  if (rows <= 0 || cols <= 0 || channels <= 0) {
    std::cerr << "Error: Invalid dimensions or channels" << std::endl;
    exit(1);
  }

  size_t total_size = static_cast<size_t>(rows) * cols * channels;
  std::fill(data_ptr_.get(), data_ptr_.get() + total_size, value);
}

};  // namespace core
