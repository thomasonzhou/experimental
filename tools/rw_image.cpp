#include <gflags/gflags.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "core/inference/onnx_infer.hpp"
#include "core/io/load_bazel_runfile.hpp"
#include "core/mat/mat_io.hpp"

DEFINE_string(in_image_path, "", "Path to the input image file");
DEFINE_string(out_image_path, "", "Path to save the output image file");
DEFINE_string(model_path, "model_weights/moge-2-vits-normal.onnx",
              "Path to the ONNX model file");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_in_image_path.empty() && FLAGS_out_image_path.empty()) {
    std::cerr << "No image paths provided. Use --in_image_path or "
                 "--out_image_path to specify files."
              << std::endl;
    return 1;
  }

  core::Mat mat;
  if (!FLAGS_in_image_path.empty()) {
    auto result = core::imread(FLAGS_in_image_path);
    if (result.has_value()) {
      mat = result.value();
    } else {
      LOG(FATAL) << "Failed to read image from " << FLAGS_in_image_path
                 << std::endl;
      return 1;
    }
    LOG(INFO) << "Loaded input image";
  }

  std::string runfile_path =
      core::io::find_runfile_path(argv[0], FLAGS_model_path);
  core::inference::onnx::CUDAModel model(runfile_path);

  core::Mat out_mat;
  if (!FLAGS_in_image_path.empty()) {
    out_mat = model.infer(mat);
  }

  if (!FLAGS_out_image_path.empty()) {
    if (!core::imwrite(FLAGS_out_image_path, out_mat)) {
      LOG(FATAL) << "Failed to write image to " << FLAGS_out_image_path;
      return 1;
    }
  }

  return 0;
}
