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

#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <onnxruntime_cxx_api.h>
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
#include "core/inference/onnx_utils.hpp"
#include "core/mat/mat_io.hpp"

DEFINE_string(in_image_path, "", "Path to the input image file");
DEFINE_string(out_image_path, "", "Path to save the output image file");
DEFINE_string(model_path,
              "/home/thchzh/src/experimental/weights/moge-2-vits-normal.onnx",
              "Path to the ONNX model file");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "WHAT";
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

  struct CudaMemoryDeleter {
    explicit CudaMemoryDeleter(Ort::Allocator* alloc) { alloc_ = alloc; }

    void operator()(void* ptr) const { alloc_->Free(ptr); }

    Ort::Allocator* alloc_;
  };

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rw_image");
  Ort::SessionOptions session_options;

  const OrtApi& api = Ort::GetApi();

  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
  std::unique_ptr<OrtCUDAProviderOptionsV2,
                  decltype(api.ReleaseCUDAProviderOptions)>
      cuda_opts_raii(cuda_options, api.ReleaseCUDAProviderOptions);

  std::vector<const char*> keys{"device_id"};
  std::vector<const char*> vals{"0"};
  Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda_options, keys.data(),
                                                  vals.data(), keys.size()));
  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
      session_options, cuda_options));

  // Now create the session
  Ort::Session session(env, FLAGS_model_path.c_str(), session_options);

  LOG(INFO) << "Model loaded: " << FLAGS_model_path;

  Ort::AllocatorWithDefaultOptions cpu_alloc;
  Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtDeviceAllocator,
                            /*device_id=*/0, OrtMemTypeDefault);
  Ort::Allocator cuda_alloc(session, info_cuda);

  core::inference::onnx::print_model_input_output(session, cpu_alloc);

  auto in0 = session.GetInputNameAllocated(0, cpu_alloc);
  auto out0 = session.GetOutputNameAllocated(0, cpu_alloc);

  if (mat.layout() == core::MatLayout::HWC) {
    mat = mat.to_layout(core::MatLayout::CHW);
  }

  const std::array<int64_t, 4> input_shape = {
      1, (int64_t)mat.channels(), (int64_t)mat.rows(), (int64_t)mat.cols()};

  auto d_in = std::unique_ptr<void, CudaMemoryDeleter>(
      cuda_alloc.Alloc(mat.size() * sizeof(float)),
      CudaMemoryDeleter(&cuda_alloc));
  CUDA_CHECK(cudaMemcpy(d_in.get(), mat.data(), sizeof(float) * mat.size(),
                        cudaMemcpyHostToDevice));

  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
      info_cuda, static_cast<float*>(d_in.get()), mat.size(),
      input_shape.data(), input_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput(in0.get(), in_tensor);

  binding.BindOutput(out0.get(), info_cuda);

  Ort::RunOptions run_opt;
  LOG(INFO) << "Running model...";
  session.Run(run_opt, binding);
  LOG(INFO) << "Model run completed";

  auto out_vals = binding.GetOutputValues();
  if (out_vals.size() != 1 || !out_vals[0].IsTensor()) {
    LOG(FATAL) << "Unexpected number/type of outputs.";
  }
  if (out_vals[0].GetTensorTypeAndShapeInfo().GetElementType() !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    LOG(FATAL) << "Unexpected output tensor type.";
  }

  float* d_out = out_vals[0].GetTensorMutableData<float>();
  auto out_shape = out_vals[0]
                       .GetTensorTypeAndShapeInfo()
                       .GetShape();  // e.g. [1,H,W,C] or [1,C,H,W]
  int64_t out_elems = 1;
  for (auto d : out_shape) out_elems *= d;

  core::Mat out_mat(mat.rows(), mat.cols(), mat.channels());

  if ((int64_t)out_mat.size() != out_elems) {
    LOG(FATAL) << "Output size mismatch: out_mat.size()=" << out_mat.size()
               << " vs runtime elements=" << out_elems;
  }

  CUDA_CHECK(cudaMemcpy(out_mat.data(), d_out, sizeof(float) * out_mat.size(),
                        cudaMemcpyDeviceToHost));

  if (!FLAGS_out_image_path.empty()) {
    if (!core::imwrite(FLAGS_out_image_path, out_mat)) {
      LOG(FATAL) << "Failed to write image to " << FLAGS_out_image_path;
      return 1;
    }
  }

  return 0;
}
