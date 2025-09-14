#pragma once

#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>

#include <string>

#include "absl/log/log.h"
#include "core/mat/mat.hpp"

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

struct CudaMemoryDeleter {
  explicit CudaMemoryDeleter(Ort::Allocator* alloc) { alloc_ = alloc; }

  void operator()(void* ptr) const { alloc_->Free(ptr); }

  Ort::Allocator* alloc_;
};

class CUDAModel {
 public:
  CUDAModel(const std::string& model_path,
            std::optional<int> device_id = std::nullopt);
  void infer_inplace(const core::Mat& input, core::Mat& output);
  inline core::Mat infer(const core::Mat& input) {
    core::Mat output(input.shape());
    infer_inplace(input, output);
    return output;
  };

  std::vector<Ort::Value> infer_raw(
      const core::Mat& input,
      std::reference_wrapper<std::vector<Ort::IoBinding>> bindings);

  // Create a reusable binding for streaming inference
  std::unique_ptr<Ort::IoBinding> create_streaming_binding(
      const core::Mat& template_input,
      const std::vector<std::string>& output_names = {});

  // For true streaming, you'd want a helper class
  class StreamingInference {
   public:
    StreamingInference(CUDAModel& model, const core::Mat& template_input,
                       const std::vector<std::string>& output_names = {});

    std::vector<Ort::Value> infer(const core::Mat& input);

   private:
    CUDAModel& model_;
    std::unique_ptr<Ort::IoBinding> binding_;
    std::unique_ptr<void, CudaMemoryDeleter> d_input_;
    std::vector<int64_t> input_shape_;
    size_t input_size_;
  };

  std::string model_path() const noexcept { return model_path_; }

 private:
  std::string model_path_;
  int device_id_;

  Ort::Env env_;
  OrtApi api_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions cpu_alloc_;

  Ort::MemoryInfo info_cuda_;
  Ort::Allocator cuda_alloc_;
};

};  // namespace core::inference::onnx
