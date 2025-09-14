#pragma once

#include <optional>
#include <vector>

#include "core/inference/onnx_infer.hpp"
#include "core/mat/mat.hpp"

namespace core::inference::onnx {

struct Keypoint {
  float x, y;
};

struct Match {
  int left_idx;
  int right_idx;
  float score;
};

struct FeatureMatchingResult {
  std::vector<Keypoint> left_keypoints;
  std::vector<Keypoint> right_keypoints;
  std::vector<Match> matches;
};

class FeatureMatcher {
 public:
  explicit FeatureMatcher(const std::string& model_path,
                          std::optional<int> device_id = std::nullopt);

  FeatureMatchingResult match_features(const core::Mat& left_image,
                                       const core::Mat& right_image);

 private:
  std::unique_ptr<CUDAModel> model_;

  // frontend processing, e.g. greyscale -> superpoint -> features
  core::Mat preprocess_image(const core::Mat& rgb_image);

  FeatureMatchingResult parse_outputs(const std::vector<Ort::Value>& outputs);
};

}  // namespace core::inference::onnx
