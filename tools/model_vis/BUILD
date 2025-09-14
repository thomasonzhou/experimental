cc_binary(
    name="moge2",
    srcs=[
        "moge2.cpp",
    ],
    deps=[
        "@abseil-cpp//absl/log:log",
        "//third_party/imgui:imgui_glfw_opengl3",
        "//core/video/v4l2:v4l2_camera",
        "//core/inference:feature_matcher",
        "//core/io:load_bazel_runfile",
    ],
    data = [
        "@model_weights//:moge_2_vits_normal",
    ],
)


cc_binary(
    name="lightglue",
    srcs=[
        "lightglue.cpp",
    ],
    deps=[
        "//third_party/imgui:imgui_glfw_opengl3",
        "//core/video/v4l2:v4l2_camera",
        "//core/inference:feature_matcher",
        "//core/io:load_bazel_runfile",
    ],
    data = [
        "@model_weights//:superpoint_lightglue",
    ],
)
