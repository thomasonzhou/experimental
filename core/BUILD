cc_library(
    name="mat",
    srcs=[
        "mat.cpp",
    ],
    hdrs=[
        "mat.hpp",
    ],
    deps=[
        "//core/stb:stb",
        ":mat_cc_proto",
    ],
    visibility=["//visibility:public"],
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
