# mat32

This is a library for computer vision inference for NVIDIA GPUs on Linux.

The main data structure is Mat, a 4D block of float32 values.

# Inference bottlenecks

- Streaming from cameras
- Copies to and from GPU
- Unnecessary memory copies
