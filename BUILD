# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 The StableHLO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//:build_tools/bazel/glob_lit_test.bzl", "glob_lit_tests")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "base",
    srcs = [
        "dialect/Base.cpp",
    ],
    hdrs = [
        "dialect/Base.h",
    ],
    deps = [
        ":base_attr_interfaces_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "base_attr_interfaces_inc_gen",
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "dialect/BaseAttrInterfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "dialect/BaseAttrInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/Base.td",
    deps = [":stablehlo_td_files"],
)

td_library(
    name = "base_td_files",
    srcs = [
        "dialect/Base.td",
    ],
    deps = [
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:QuantizationOpsTdFiles",
    ],
)

cc_library(
    name = "broadcast_utils",
    srcs = [
        "dialect/BroadcastUtils.cpp",
    ],
    hdrs = [
        "dialect/BroadcastUtils.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ShapeDialect",
    ],
)

gentbl_cc_library(
    name = "chlo_attrs_inc_gen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "dialect/ChloAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "dialect/ChloAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/ChloOps.td",
    deps = [
        ":chlo_td_files",
    ],
)

gentbl_cc_library(
    name = "chlo_enums_inc_gen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "dialect/ChloEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "dialect/ChloEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/ChloOps.td",
    deps = [
        ":chlo_td_files",
    ],
)

gentbl_cc_library(
    name = "chlo_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "dialect/ChloOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "dialect/ChloOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/ChloOps.td",
    deps = [
        ":chlo_td_files",
    ],
)

td_library(
    name = "chlo_td_files",
    srcs = [
        "dialect/ChloEnums.td",
        "dialect/ChloOps.td",
    ],
    deps = [
        ":base_td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "chlo_ops",
    srcs = [
        "dialect/ChloOps.cpp",
    ],
    hdrs = [
        "dialect/ChloOps.h",
    ],
    deps = [
        ":base",
        ":broadcast_utils",
        ":chlo_attrs_inc_gen",
        ":chlo_enums_inc_gen",
        ":chlo_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:QuantOps",
    ],
)

cc_library(
    name = "register",
    srcs = [
        "dialect/Register.cpp",
    ],
    hdrs = [
        "dialect/Register.h",
    ],
    deps = [
        ":chlo_ops",
        ":stablehlo_ops",
        "@llvm-project//mlir:IR",
    ],
)

gentbl_cc_library(
    name = "stablehlo_attrs_inc_gen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "dialect/StablehloAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "dialect/StablehloAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/StablehloOps.td",
    deps = [
        ":stablehlo_td_files",
    ],
)

gentbl_cc_library(
    name = "stablehlo_enums_inc_gen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "dialect/StablehloEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "dialect/StablehloEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/StablehloOps.td",
    deps = [
        ":stablehlo_td_files",
    ],
)

gentbl_cc_library(
    name = "stablehlo_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "dialect/StablehloOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "dialect/StablehloOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect/StablehloOps.td",
    deps = [
        ":stablehlo_td_files",
    ],
)

td_library(
    name = "stablehlo_td_files",
    srcs = [
        "dialect/Base.td",
        "dialect/StablehloAttrs.td",
        "dialect/StablehloEnums.td",
        "dialect/StablehloOps.td",
    ],
    deps = [
        ":base_td_files",
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ShapeOpsTdFiles",
    ],
)

cc_library(
    name = "stablehlo_ops",
    srcs = [
        "dialect/StablehloOps.cpp",
    ],
    hdrs = [
        "dialect/StablehloOps.h",
    ],
    deps = [
        ":base",
        ":stablehlo_attrs_inc_gen",
        ":stablehlo_enums_inc_gen",
        ":stablehlo_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_binary(
    name = "stablehlo-opt",
    srcs = ["tools/StablehloOptMain.cpp"],
    deps = [
        ":register",
        ":test_utils",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

glob_lit_tests(
    data = [":test_data"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)

filegroup(
    name = "test_data",
    testonly = True,
    data = [
        ":stablehlo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

gentbl_cc_library(
    name = "test_utils_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=HloTest",
            ],
            "tests/TestUtils.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tests/TestUtils.td",
    deps = [
        ":test_utils_td_files",
    ],
)

td_library(
    name = "test_utils_td_files",
    srcs = [
        "tests/TestUtils.td",
    ],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "test_utils",
    srcs = [
        "tests/TestUtils.cpp",
    ],
    hdrs = [
        "tests/TestUtils.h",
    ],
    deps = [
        ":test_utils_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
