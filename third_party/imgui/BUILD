load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "imgui",
    srcs = [
        "imgui.cpp",
        "imgui_demo.cpp",
        "imgui_draw.cpp",
        "imgui_tables.cpp",
        "imgui_widgets.cpp",
    ],
    hdrs = [
        "imconfig.h",
        "imgui.h",
        "imgui_internal.h",
        "imstb_rectpack.h",
        "imstb_textedit.h",
        "imstb_truetype.h",
    ],
    includes = ["."],
    defines = [
        "IMGUI_DEFINE_MATH_OPERATORS",
    ],
)

cc_library(
    name = "imgui_glfw",
    srcs = ["imgui_impl_glfw.cpp"],
    hdrs = ["imgui_impl_glfw.h"],
    deps = [
        ":imgui",
        "@glfw",
    ],
    includes = ["."],
)

cc_library(
    name = "imgui_opengl3",
    srcs = [
        "imgui_impl_opengl3.cpp",
        "imgui_impl_opengl3_loader.h",
    ],
    hdrs = ["imgui_impl_opengl3.h"],
    deps = [
        ":imgui",
    ],
    includes = ["."],
    linkopts = [
        "-lGL",
        "-ldl",
    ],
)
