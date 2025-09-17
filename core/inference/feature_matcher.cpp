#include "core/inference/feature_matcher.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <memory>

#include "absl/log/log.h"
#include "core/mat/utils/color_conversion.hpp"

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                           \
  do {                                                             \
    cudaError_t _e = (expr);                                       \
    if (_e != cudaSuccess) {                                       \
      LOG(FATAL) << "CUDA error " << static_cast<int>(_e) << " : " \
                 << cudaGetErrorString(_e);                        \
    }                                                              \
  } while (0)
#endif

namespace core::inference::onnx {

FeatureMatcher::FeatureMatcher(const std::string& model_path,
                               std::optional<int> device_id)
    : model_(std::make_unique<CUDAModel>(model_path, device_id)) {}

FeatureMatchingResult FeatureMatcher::match_features(
    const core::Mat& left_image, const core::Mat& right_image) {
  core::Mat left_gray = core::video::convert_rgb_to_grayscale(left_image);
  core::Mat right_gray = core::video::convert_rgb_to_grayscale(right_image);

  // Create interleaved batch: [left, right]
  // Input shape: (2, 1, H, W) using 4D constructor
  size_t h = left_gray.rows();
  size_t w = left_gray.cols();

  core::Mat batch_input(core::MatShape::make_4d(2, h, w, 1), std::nullopt,
                        core::MatLayout::NCHW);

  for (size_t r = 0; r < h; ++r) {
    for (size_t c = 0; c < w; ++c) {
      batch_input(0, 0, r, c) = left_gray(r, c, 0);
      batch_input(1, 0, r, c) = right_gray(r, c, 0);
    }
  }

  // Run inference
  std::vector<Ort::IoBinding> bindings_vec;
  auto outputs = model_->infer_raw(batch_input, std::ref(bindings_vec));

  // Parse outputs into structured result
  return parse_outputs(outputs);
}

FeatureMatchingResult FeatureMatcher::parse_outputs(
    const std::vector<Ort::Value>& outputs) {
  FeatureMatchingResult result;

  if (outputs.size() != 3) {
    LOG(ERROR) << "Expected 3 outputs (keypoints, matches, scores), got "
               << outputs.size();
    return result;
  }

  // Output 0: Keypoints (2, 1024, 2) - int64 tensor
  const auto& keypoints_tensor = outputs[0];
  auto kp_shape = keypoints_tensor.GetTensorTypeAndShapeInfo().GetShape();
  LOG(INFO) << "Keypoints shape: [" << kp_shape[0] << ", " << kp_shape[1]
            << ", " << kp_shape[2] << "]";

  // Copy keypoints data to host as int64
  std::vector<int64_t> kp_data(kp_shape[0] * kp_shape[1] * kp_shape[2]);
  CUDA_CHECK(
      cudaMemcpy(kp_data.data(), keypoints_tensor.GetTensorData<int64_t>(),
                 kp_data.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));

  // some keypoints may be zero (no keypoint)
  size_t num_keypoints_per_image = kp_shape[1];  // 1024

  // store alll 1024 keypoints from both images (even zero ones)
  std::vector<Keypoint> all_left_keypoints(num_keypoints_per_image);
  std::vector<Keypoint> all_right_keypoints(num_keypoints_per_image);

  for (size_t i = 0; i < num_keypoints_per_image; ++i) {
    // Left image keypoints (batch 0) - convert int64 to float
    float x_left = static_cast<float>(
        kp_data[0 * num_keypoints_per_image * 2 + i * 2 + 0]);
    float y_left = static_cast<float>(
        kp_data[0 * num_keypoints_per_image * 2 + i * 2 + 1]);

    // Right image keypoints (batch 1)
    float x_right = static_cast<float>(
        kp_data[1 * num_keypoints_per_image * 2 + i * 2 + 0]);
    float y_right = static_cast<float>(
        kp_data[1 * num_keypoints_per_image * 2 + i * 2 + 1]);

    all_left_keypoints[i] = {x_left, y_left};
    all_right_keypoints[i] = {x_right, y_right};
  }

  // Output 1: Matches (M, 3) - int64 tensor with [batch_idx, left_kpt_idx,
  // right_kpt_idx]
  const auto& matches_tensor = outputs[1];
  auto matches_shape = matches_tensor.GetTensorTypeAndShapeInfo().GetShape();
  LOG(INFO) << "Matches shape: [" << matches_shape[0] << ", "
            << matches_shape[1] << "]";

  if (matches_shape.size() >= 1 && matches_shape[0] > 0) {
    std::vector<int64_t> matches_data(matches_shape[0] * matches_shape[1]);
    CUDA_CHECK(cudaMemcpy(
        matches_data.data(), matches_tensor.GetTensorData<int64_t>(),
        matches_data.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Output 2: Match scores (M,) - float32 tensor
    const auto& scores_tensor = outputs[2];
    auto scores_shape = scores_tensor.GetTensorTypeAndShapeInfo().GetShape();
    LOG(INFO) << "Scores shape: [" << scores_shape[0] << "]";

    std::vector<float> scores_data(scores_shape[0]);
    CUDA_CHECK(
        cudaMemcpy(scores_data.data(), scores_tensor.GetTensorData<float>(),
                   scores_data.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Parse matches - only add keypoints that are actually matched
    for (size_t i = 0; i < matches_shape[0]; ++i) {
      int batch_idx = matches_data[i * 3 + 0];
      int left_idx = matches_data[i * 3 + 1];
      int right_idx = matches_data[i * 3 + 2];
      float score = scores_data[i];

      // Only consider matches for batch 0 and valid indices
      if (batch_idx == 0 && left_idx >= 0 && right_idx >= 0 &&
          left_idx < all_left_keypoints.size() &&
          right_idx < all_right_keypoints.size()) {
        // Get the actual matched keypoints
        Keypoint left_kp = all_left_keypoints[left_idx];
        Keypoint right_kp = all_right_keypoints[right_idx];

        // Only add if coordinates are reasonable (valid pixel coordinates)
        // SuperPoint outputs pixel coordinates, so check for valid image bounds
        if (left_kp.x >= 0 && left_kp.y >= 0 && left_kp.x < 1000 &&
            left_kp.y < 1000 && right_kp.x >= 0 && right_kp.y >= 0 &&
            right_kp.x < 1000 && right_kp.y < 1000) {
          // Add to final result keypoints
          result.left_keypoints.push_back(left_kp);
          result.right_keypoints.push_back(right_kp);

          // Add match (indices in the final filtered lists)
          int final_left_idx = result.left_keypoints.size() - 1;
          int final_right_idx = result.right_keypoints.size() - 1;
          result.matches.push_back({final_left_idx, final_right_idx, score});
        }
      }
    }
  }

  LOG(INFO) << "Found " << result.left_keypoints.size() << " left keypoints, "
            << result.right_keypoints.size() << " right keypoints, "
            << result.matches.size() << " matches";

  return result;
}

}  // namespace core::inference::onnx
