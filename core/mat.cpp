#include "mat.hpp"

#include <iostream>
#include <string>

namespace core {
static constexpr int KEEP_CHANNELS = 0;  // retain all channels from source

Mat::Mat(const std::string filename) : img_ptr(nullptr, stbi_image_free) {
  if (filename.empty()) {
    std::cerr << "Error: No filename provided" << std::endl;
    exit(1);
  }

  img_ptr.reset(
      stbi_load(filename.c_str(), &width, &height, &channels, KEEP_CHANNELS));

  if (!img_ptr) {
    std::cerr << "Error: Failed to load image " << filename << std::endl;
    exit(1);
  }

  std::cout << "Loaded image with dimensions " << height << "x" << width << "x"
            << channels << std::endl;
}

};  // namespace core
