#include <gflags/gflags.h>
#include <stdio.h>

#include <iostream>
#include <memory>
#include <string>

#include "core/mat.hpp"

DEFINE_string(image_in_path, "", "Path to the input image file");
DEFINE_string(image_out_path, "", "Path to save the output image file");

int main(int argc, char* argv[]) {
  if (FLAGS_image_in_path.empty() && FLAGS_image_out_path.empty()) {
    std::cerr << "No image paths provided. Use --image_in_path or "
                 "--image_out_path to specify files."
              << std::endl;
    return 1;
  }

  core::Mat mat;
  if (!FLAGS_image_in_path.empty()) {
    mat = core::imread(FLAGS_image_in_path);
  } else {
    mat = core::zeros(100, 100, 3);  // Default to a 100x100 black image
  }

  if (!FLAGS_image_out_path.empty()) {
    core::imwrite(FLAGS_image_out_path, mat);
  }

  return 0;
}
