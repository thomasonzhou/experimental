#include "core/inference/onnx_infer.hpp"

#include <memory>

namespace core::inference::onnx {

CUDAModel::CUDAModel(const std::string& model_path,
                     std::optional<int> device_id)
    : model_path_(model_path),
      device_id_(device_id.value_or(0)),
      env_(ORT_LOGGING_LEVEL_WARNING, "CUDAModel"),
      api_(Ort::GetApi()),
      session_(nullptr),
      info_cuda_(nullptr),
      cuda_alloc_(nullptr) {
  if (model_path.empty()) {
    LOG(FATAL) << "Model path is empty";
  }

  Ort::SessionOptions session_options;

  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  Ort::ThrowOnError(api_.CreateCUDAProviderOptions(&cuda_options));
  std::unique_ptr<OrtCUDAProviderOptionsV2,
                  decltype(api_.ReleaseCUDAProviderOptions)>
      cuda_opts_raii(cuda_options, api_.ReleaseCUDAProviderOptions);

  std::vector<const char*> keys{"device_id"};
  std::vector<const char*> vals{"0"};
  Ort::ThrowOnError(api_.UpdateCUDAProviderOptions(cuda_options, keys.data(),
                                                   vals.data(), keys.size()));
  Ort::ThrowOnError(api_.SessionOptionsAppendExecutionProvider_CUDA_V2(
      session_options, cuda_options));

  session_ = Ort::Session(env_, model_path.c_str(), session_options);

  LOG(INFO) << "Model loaded: " << model_path;

  info_cuda_ = Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator,
                               device_id_, OrtMemTypeDefault);
  cuda_alloc_ = Ort::Allocator(session_, info_cuda_);
}

void CUDAModel::infer_inplace(const core::Mat& const_input, core::Mat& output) {
  auto in0 = session_.GetInputNameAllocated(0, cpu_alloc_);
  auto out0 = session_.GetOutputNameAllocated(0, cpu_alloc_);

  const core::Mat* in_ptr = std::addressof(const_input);
  std::optional<core::Mat> chw_input;
  if (const_input.layout() == core::MatLayout::NHWC) {
    chw_input.emplace(const_input.to_layout(core::MatLayout::NCHW));
    in_ptr = std::addressof(chw_input.value());
  }
  const core::Mat& input = *in_ptr;

  const std::array<int64_t, 4> input_shape = {1, (int64_t)input.channels(),
                                              (int64_t)input.rows(),
                                              (int64_t)input.cols()};

  auto d_in = std::unique_ptr<void, CudaMemoryDeleter>(
      cuda_alloc_.Alloc(input.size() * sizeof(float)),
      CudaMemoryDeleter(&cuda_alloc_));
  CUDA_CHECK(cudaMemcpy(d_in.get(), input.data(), sizeof(float) * input.size(),
                        cudaMemcpyHostToDevice));

  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
      info_cuda_, static_cast<float*>(d_in.get()), input.size(),
      input_shape.data(), input_shape.size());

  Ort::IoBinding binding(session_);
  binding.BindInput(in0.get(), in_tensor);

  binding.BindOutput(out0.get(), info_cuda_);

  // Ort::RunOptions run_opt;
  // LOG(INFO) << "Running model...";
  // session_.Run(run_opt, binding);
  // LOG(INFO) << "Model run completed";

  // auto out_vals = binding.GetOutputValues();

  // ******

  std::vector<Ort::IoBinding> bindings_vec;
  bindings_vec.push_back(std::move(binding));
  /***** */
  auto out_vals = infer_raw(input, std::ref(bindings_vec));
  /***** */

  if (out_vals[0].GetTensorTypeAndShapeInfo().GetElementType() !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    LOG(FATAL) << "Unexpected output tensor type.";
  }

  float* d_out = out_vals[0].GetTensorMutableData<float>();
  auto out_shape = out_vals[0].GetTensorTypeAndShapeInfo().GetShape();
  int64_t out_elems = 1;
  for (auto d : out_shape) out_elems *= d;

  if ((int64_t)output.size() != out_elems) {
    LOG(FATAL) << "Output size mismatch: output.size()=" << output.size()
               << " vs runtime elements=" << out_elems;
  }

  CUDA_CHECK(cudaMemcpy(output.data(), d_out, sizeof(float) * output.size(),
                        cudaMemcpyDeviceToHost));
}

// assume all preprocessing is done
std::vector<Ort::Value> CUDAModel::infer_raw(
    const core::Mat& input,
    std::reference_wrapper<std::vector<Ort::IoBinding>> binding) {
  auto in0 = session_.GetInputNameAllocated(0, cpu_alloc_);

  std::vector<int64_t> input_shape{
      (int64_t)input.batch_size(), (int64_t)input.channels(),
      (int64_t)input.rows(), (int64_t)input.cols()};

  auto d_in = std::unique_ptr<void, CudaMemoryDeleter>(
      cuda_alloc_.Alloc(input.size() * sizeof(float)),
      CudaMemoryDeleter(&cuda_alloc_));
  CUDA_CHECK(cudaMemcpy(d_in.get(), input.data(), sizeof(float) * input.size(),
                        cudaMemcpyHostToDevice));

  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
      info_cuda_, static_cast<float*>(d_in.get()), input.size(),
      input_shape.data(), input_shape.size());

  Ort::IoBinding binding(session_);
  binding.BindInput(in0.get(), in_tensor);

  size_t num_inputs = session_.GetInputCount();
  if (num_inputs > 1) {
    auto in1 = session_.GetInputNameAllocated(1, cpu_alloc_);

    const int64_t num_tokens_value = 1800;
    std::vector<int64_t> num_tokens_shape{};  // empty shape for scalar

    auto d_num_tokens = std::unique_ptr<void, CudaMemoryDeleter>(
        cuda_alloc_.Alloc(sizeof(int64_t)), CudaMemoryDeleter(&cuda_alloc_));
    CUDA_CHECK(cudaMemcpy(d_num_tokens.get(), &num_tokens_value,
                          sizeof(int64_t), cudaMemcpyHostToDevice));

    Ort::Value num_tokens_tensor = Ort::Value::CreateTensor<int64_t>(
        info_cuda_, static_cast<int64_t*>(d_num_tokens.get()), 1,
        num_tokens_shape.data(), num_tokens_shape.size());

    binding.BindInput(in1.get(), num_tokens_tensor);
  }

  // Bind all outputs to CUDA memory
  size_t num_outputs = session_.GetOutputCount();
  for (size_t i = 0; i < num_outputs; ++i) {
    auto out_name = session_.GetOutputNameAllocated(i, cpu_alloc_);
    binding.BindOutput(out_name.get(), info_cuda_);
  }

  Ort::RunOptions run_opt;
  LOG(INFO) << "Running feature matching model...";
  session_.Run(run_opt, binding);
  LOG(INFO) << "Feature matching model run completed";

  auto out_vals = binding.GetOutputValues();
  LOG(INFO) << "Got " << out_vals.size() << " outputs from model";

  return out_vals;
}

};  // namespace core::inference::onnx
