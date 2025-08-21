- onnx
- tensorrt

- inference profiling (time, VRAM)
- inference consistency (quantization)


To use CUDA 12.9

Install cudnn
https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=RHEL&target_version=9&target_type=rpm_network&Configuration=Full

```sh
LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
```
