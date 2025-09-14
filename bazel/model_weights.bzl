def _model_weights_impl(repository_ctx):
    models = {
        "moge_2_vits_normal": {
            "url": "https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx/resolve/main/model.onnx",
            "sha256": "24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0",
            "filename": "moge-2-vits-normal.onnx"
        },
        "superpoint_lightglue": {
            "url": "https://huggingface.co/thomasonzhou/superpoint-lightglue/resolve/main/model.onnx",
            "sha256": "542057b08d51f3a790d1136bd842e4ff44079e7ce63dba627c2e25e8e8d4b1b0",
            "filename": "superpoint-lightglue.onnx"
        }
    }

    for name, config in models.items():
        repository_ctx.download(
            url = config["url"],
            output = config["filename"],
            sha256 = config["sha256"]
        )

    build_content = """
filegroup(
    name = "all_models",
    srcs = glob(["*.onnx"]),
    visibility = ["//visibility:public"],
)
"""

    for name, config in models.items():
        build_content += """
filegroup(
    name = "%s",
    srcs = ["%s"],
    visibility = ["//visibility:public"],
)
""" % (name, config["filename"])

    repository_ctx.file("BUILD", build_content)

_model_weights_repo = repository_rule(
    implementation = _model_weights_impl,
    attrs = {}
)

def _model_weights_extension_impl(module_ctx):
    _model_weights_repo(name = "model_weights")

model_weights = module_extension(
    implementation = _model_weights_extension_impl
)
