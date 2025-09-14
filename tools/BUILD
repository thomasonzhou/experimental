cc_binary(
    name="rw_image",
    srcs=[
        "rw_image.cpp",
    ],
    deps=[
        "//core/mat:mat_io",
        "@gflags//:gflags",
        "@abseil-cpp//absl/log:log",
        "//core/inference:onnx_infer",
        "//core/io:load_bazel_runfile",
    ],
    data = [
        "@model_weights//:moge_2_vits_normal",
    ],
)
