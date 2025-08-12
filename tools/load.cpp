#include <gflags/gflags.h>
#include <stdio.h>

#include <iostream>
#include <memory>
#include <string>

#include "core/mat.hpp"

DEFINE_string(image_path, "", "Path to the image file");

int main(int argc, char* argv[]) {
  if (FLAGS_image_path.empty()) {
    std::cerr
        << "No image path provided. Use --image_path to specify the image file."
        << std::endl;
    return 1;
  }

  core::Mat mat(FLAGS_image_path);

  return 0;
}
