cc_library(
    name = "element",
    srcs = [
        "Element.cpp",
    ],
    hdrs = [
        "Element.h",
    ],
    deps = [
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:IR",
    ],
)

cc_library(
    name = "interpreter",
    srcs = [
        "Interpreter.cpp",
    ],
    hdrs = [
        "Interpreter.h",
    ],
    deps = [
        ":ops",
        ":tensor",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:FuncDialect",
        "//third_party/llvm/llvm-project/mlir:IR",
        "//third_party/llvm/llvm-project/mlir:Support",
        "//third_party/tensorflow/compiler/xla/mlir_hlo",
    ],
)

cc_library(
    name = "ops",
    srcs = [
        "Ops.cpp",
    ],
    hdrs = [
        "Ops.h",
    ],
    deps = [
        ":element",
        ":tensor",
        "//third_party/tensorflow/compiler/xla/mlir_hlo",
    ],
)

cc_library(
    name = "tensor",
    srcs = [
        "Tensor.cpp",
    ],
    hdrs = [
        "Tensor.h",
    ],
    deps = [
        ":element",
        "//third_party/llvm/llvm-project/llvm:Support",
        "//third_party/llvm/llvm-project/mlir:AsmParser",
        "//third_party/llvm/llvm-project/mlir:IR",
    ],
)
