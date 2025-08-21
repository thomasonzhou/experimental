# https://bazel.build/rules/lib/repo/http
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


# https://github.com/microsoft/onnxruntime
ONNXRUNTIME_VERSION = "1.22.0"
ONNXRUNTIME_SHA256 = "8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3"
ONNXRUNTIME_GPU_SHA256 = "2a19dbfa403672ec27378c3d40a68f793ac7a6327712cd0e8240a86be2b10c55"

def _onnxruntime_impl(module_ctx):
    # CPU version
    http_archive(
        name = "onnxruntime",
        url = "https://github.com/microsoft/onnxruntime/releases/download/v{}/onnxruntime-linux-x64-{}.tgz".format(ONNXRUNTIME_VERSION, ONNXRUNTIME_VERSION),
        sha256 = ONNXRUNTIME_SHA256,
        strip_prefix = "onnxruntime-linux-x64-{}".format(ONNXRUNTIME_VERSION),
        build_file_content = """
cc_library(
    name = "onnxruntime",
    hdrs = [
        "include/onnxruntime_c_api.h",
        "include/onnxruntime_cxx_api.h",
        "include/onnxruntime_cxx_inline.h",
        "include/onnxruntime_float16.h",
        "include/cpu_provider_factory.h",
    ],
    srcs = [
        "lib/libonnxruntime.so",
        "lib/libonnxruntime.so.1",
        "lib/libonnxruntime.so.{version}",
    ],
    includes = ["include"],
    linkopts = ["-ldl", "-lrt", "-pthread"],
    visibility = ["//visibility:public"],
)

""".format(version=ONNXRUNTIME_VERSION),
    )


    # GPU version, do not link libonnxruntime_providers_cuda.so
    # https://github.com/microsoft/onnxruntime/issues/16146#issuecomment-1681849453
    http_archive(
        name = "onnxruntime_gpu",
        url = "https://github.com/microsoft/onnxruntime/releases/download/v{}/onnxruntime-linux-x64-gpu-{}.tgz".format(ONNXRUNTIME_VERSION, ONNXRUNTIME_VERSION),
        sha256 = ONNXRUNTIME_GPU_SHA256,
        strip_prefix = "onnxruntime-linux-x64-gpu-{}".format(ONNXRUNTIME_VERSION),
        build_file_content = """
cc_library(
    name = "onnxruntime_gpu",
    hdrs = [
        "include/onnxruntime_c_api.h",
        "include/onnxruntime_cxx_api.h",
        "include/onnxruntime_cxx_inline.h",
        "include/onnxruntime_float16.h",
        "include/cpu_provider_factory.h",
    ],
    srcs = [
        "lib/libonnxruntime.so",
        "lib/libonnxruntime.so.1",
        "lib/libonnxruntime.so.{version}",
    ],
    data = [
        "lib/libonnxruntime_providers_shared.so",
    ],

    includes = ["include"],
    linkopts = ["-ldl", "-lrt", "-pthread"],
    visibility = ["//visibility:public"],
)

""".format(version=ONNXRUNTIME_VERSION),
    )

onnxruntime = module_extension(implementation = _onnxruntime_impl)
