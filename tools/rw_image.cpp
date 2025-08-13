#include <gflags/gflags.h>
#include <stdio.h>

#include <iostream>
#include <memory>
#include <string>

#include "core/mat.hpp"

DEFINE_string(image_in_path, "", "Path to the input image file");
DEFINE_string(image_out_path, "", "Path to save the output image file");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
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
    mat = core::zeros(5, 5, 4);
  }

  mat(0, 0, 0) = 1.0f;  // R
  mat(0, 0, 3) = 1.0f;
  mat(1, 1, 1) = 1.0f;  // G
  mat(1, 1, 3) = 1.0f;
  mat(2, 2, 2) = 1.0f;  // B
  mat(2, 2, 3) = 1.0f;

  if (!FLAGS_image_out_path.empty()) {
    core::imwrite(FLAGS_image_out_path, mat);
  }

  return 0;
}
