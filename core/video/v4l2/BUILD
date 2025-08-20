cc_library(
    name = "v4l2_utils",
    hdrs = ["v4l2_utils.hpp"],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "v4l2_camera",
    srcs = ["v4l2_camera.cpp"],
    hdrs = ["v4l2_camera.hpp"],
    deps = [
        ":v4l2_utils",
        "//core/mat:mat",
        "//core/video:camera_interface",
        "//core/video:color_conversion",
    ],
    visibility = ["//visibility:public"]
)

# Convenience target for everything
cc_library(
    name = "v4l2",
    deps = [
        ":v4l2_utils",
        ":v4l2_camera",
    ],
    visibility = ["//visibility:public"]
)
