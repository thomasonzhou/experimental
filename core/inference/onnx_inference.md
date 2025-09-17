# How ONNX inference works

Working document of my understanding of the onnxruntime_gpu flow for C++

1. There is a global instance of the Ort API to enable concurrent inference
2. Most configuration is done by passing in a config object
    1. Ort::Env (logging and thread pools)
    2. Ort::SessionOptions (per ONNX file)
    3. Ort::RunOptions (per inference)

3. Session instances run inference on chunks of memory and are associated with one ONNX model. They are reusable.
4. Inputs must all be bound, outputs are optionally bound to specify only a subset of the computation graph
