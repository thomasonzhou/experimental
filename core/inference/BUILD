cc_library(
    name = "onnx_infer",
    srcs = [
        "onnx_infer.cpp",
    ],
    hdrs = [
        "onnx_infer.hpp",
    ],
    deps = [
        "@onnxruntime_gpu//:onnxruntime_gpu",
        "@rules_cuda//cuda:runtime",
        "@abseil-cpp//absl/log:log",
        "//core/mat:mat",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "feature_matcher",
    srcs = [
        "feature_matcher.cpp",
    ],
    hdrs = [
        "feature_matcher.hpp",
    ],
    deps = [
        ":onnx_infer",
        "@rules_cuda//cuda:runtime",
        "@abseil-cpp//absl/log:log",
        "//core/mat:mat",
    ],
    visibility = ["//visibility:public"]
)
