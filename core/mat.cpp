#include "mat.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace core {
static constexpr int KEEP_CHANNELS = 0;  // retain all channels from source

Mat::Mat(const std::string& filename) : data_ptr(nullptr) {
  if (filename.empty()) {
    std::cerr << "Error: No filename provided" << std::endl;
    exit(1);
  }

  unsigned char* img_data =
      stbi_load(filename.c_str(), &width_, &height_, &channels_, KEEP_CHANNELS);

  if (!img_data) {
    std::cerr << "Error: Failed to load image " << filename << std::endl;
    exit(1);
  }

  size_t total_size = static_cast<size_t>(height_) * width_ * channels_;
  data_ptr = std::make_unique<float[]>(total_size);

  for (size_t i = 0; i < total_size; ++i) {
    data_ptr[i] = static_cast<float>(img_data[i]) / 255.0f;
  }

  stbi_image_free(img_data);
}

Mat::Mat(const int height, const int width, const int channels,
         const float value)
    : data_ptr(std::make_unique<float[]>(static_cast<size_t>(height) * width *
                                         channels)),
      height_(height),
      width_(width),
      channels_(channels) {
  if (height <= 0 || width <= 0 || channels <= 0) {
    std::cerr << "Error: Invalid dimensions or channels" << std::endl;
    exit(1);
  }

  size_t total_size = static_cast<size_t>(height) * width * channels;
  std::fill(data_ptr.get(), data_ptr.get() + total_size, value);
}

};  // namespace core
