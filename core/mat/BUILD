cc_library(
    name="mat",
    srcs=[
        "mat.cpp",
    ],
    hdrs=[
        "mat.hpp",
    ],
    visibility=["//visibility:public"],
)

cc_library(
    name = "mat_io",
    srcs = [
        "mat_io.cpp",
    ],
    hdrs = [
        "mat_io.hpp",
    ],
    deps = [
        "@stb//:stb_image",
        "@stb//:stb_image_write",
        ":mat",
        ":mat_cc_proto",
    ],
    defines = ["STB_IMAGE_WRITE_IMPLEMENTATION"],
    visibility = ["//visibility:public"],
)

proto_library(
    name="mat_proto",
    srcs=[
        "mat.proto",
    ],
    visibility=["//visibility:public"],
)

cc_proto_library(
    name="mat_cc_proto",
    deps=[":mat_proto"],
    visibility=["//visibility:public"],
)
