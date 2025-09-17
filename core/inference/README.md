# Goals of core/inference

Given an ONNX file
- inspect inputs and outputs using a helper script
    - detect dynamic size inputs
    - identify NCHW/NHWC

- specify desired outputs via binding
- pass Mat objects as inputs and recieve a raw vector of Ort::Value objects OR Mat objects


Wishlist
- build TensorRT engines
- TensorRT ONNX backend
- inference profiling (time, VRAM)
- inference consistency (quantization)
- better pipeline to generate ad-hoc ONNX files for specific dimensions
    - idea: pin to a Git SHA and maintain the conversion/fine-tuning
- NVIDIA DALI


# To use CUDA 12.9

Install cudnn
https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=RHEL&target_version=9&target_type=rpm_network&Configuration=Full

```sh
LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
```
