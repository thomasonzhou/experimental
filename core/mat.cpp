#include "mat.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace core {
static constexpr int KEEP_CHANNELS = 0;  // retain all channels from source

Mat::Mat(const std::string& filename) : img_ptr(nullptr, stbi_image_free) {
  if (filename.empty()) {
    std::cerr << "Error: No filename provided" << std::endl;
    exit(1);
  }

  img_ptr.reset(stbi_load(filename.c_str(), &width_, &height_, &channels_,
                          KEEP_CHANNELS));

  if (!img_ptr) {
    std::cerr << "Error: Failed to load image " << filename << std::endl;
    exit(1);
  }
}

Mat::Mat(const int height, const int width, const int channels,
         const unsigned char value)
    : img_ptr(nullptr, stbi_image_free),
      height_(height),
      width_(width),
      channels_(channels) {
  if (height <= 0 || width <= 0 || channels <= 0) {
    std::cerr << "Error: Invalid dimensions or channels" << std::endl;
    exit(1);
  }

  size_t total_size = static_cast<size_t>(height) * width * channels;
  unsigned char* data = static_cast<unsigned char*>(malloc(total_size));

  if (!data) {
    std::cerr << "Error: Failed to allocate memory for batched matrix"
              << std::endl;
    exit(1);
  }

  std::fill(data, data + total_size, value);

  img_ptr.reset(data);
}

Mat imread(const std::string& filename) {
  Mat img(filename);
  if (!img.data()) {
    throw std::runtime_error("Failed to read image");
  }
  return img;
}

};  // namespace core
