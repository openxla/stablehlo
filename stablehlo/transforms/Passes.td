/* Copyright 2022 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

include "mlir/Pass/PassBase.td"

def StablehloCanonicalizeDynamismPass : Pass<"stablehlo-canonicalize-dynamism", "func::FuncOp"> {
  let summary = "Canonicalizes dynamic StableHLO ops into static ops.";
  let description = [{
    Replaces dynamic StableHLO ops like DynamicReshapeOp with the corresponding
    static counterparts like ReshapeOp if all the dynamic elements of these ops
    are actually constant.

    For example, if the output_shape operand of DynamicReshapeOp is a constant
    value, then the operation can be transformed to ReshapeOp.
  }];
}

def StablehloLegalizeToVhloPass : Pass<"stablehlo-legalize-to-vhlo", "ModuleOp"> {
  let summary = "Legalize StableHLO to VHLO.";
  let dependentDialects = ["mlir::vhlo::VhloDialect"];
}

def StablehloRefineShapesPass : Pass<"stablehlo-refine-shapes", "ModuleOp"> {
  let summary = "Refines shapes across a StableHLO program.";
  let description = [{
    Walks through a StableHLO program refining shapes within ops.

    The flagship use case for this pass is specializing dynamically-shaped
    programs to static shapes. If a dynamically-shaped StableHLO program has the
    right structure, then updating its argument types from dynamic shapes to
    static shapes and running this pass will propagate static shapes across
    the program.
  }];
}

def StablehloRefineArgumentsPass : Pass<"stablehlo-refine-arguments", "ModuleOp"> {
  let summary = "Refines the argument shapes of the main function.";
  let description = [{
    Modifies the arguments of the main function using the input type signature.
    Wraps arguments in custom_call @stablehlo.shape_refinement_operand_wrapper
    to keep the IR valid before shape refinement is run.

    The `refinedTypesOption` can be used to specify a list of refined types.
    This can be specified in MLIR with `--types='tensor<...>,tensor<...>'`, or
    passed to the pass create method. The refinement type list must specify the
    type of every argument to the `main` method being refined.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
  let options = [
    ListOption<"refinedTypesOption", "types", "std::string",
           "The new types to be used for the main function's arguments, specified as an MLIR TypeRange 'tensor<1x2xf32>, ...'">,
  ];
}

def VhloLegalizeToStablehloPass : Pass<"vhlo-legalize-to-stablehlo", "ModuleOp"> {
  let summary = "Legalize VHLO to StableHLO.";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::quant::QuantizationDialect",
    "mlir::shape::ShapeDialect",
    "mlir::stablehlo::StablehloDialect",
  ];
}

def VhloToVersionPass : Pass<"vhlo-to-version"> {
  let summary = "Convert between versions of VHLO.";
  let options = [
    Option<"targetVersionOption", "target", "std::string", "",
           "The target version. Must be a version of the form #.#.# or 'current'.">,
  ];
}

def ChloLegalizeToStablehloPass : Pass<"chlo-legalize-to-stablehlo", "func::FuncOp"> {
  let summary = "Legalizes from CHLO ops flow to StableHLO and Shape ops";
  let dependentDialects = [
    "mlir::shape::ShapeDialect",
    "mlir::stablehlo::StablehloDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def StablehloAggressiveFolderPass
    : Pass<"stablehlo-aggressive-folder", "func::FuncOp"> {
  let summary = "Folds StableHLO operations";
  let dependentDialects = [
    "mlir::tensor::TensorDialect",
  ];
  let options = [
    Option<"foldFloat", "fold-float", "bool", /*default=*/"true",
           "Allow for potentially lossy computations using float type.">,
  ];
}

def StablehloAggressiveSimplificationPass
    : Pass<"stablehlo-aggressive-simplification", "func::FuncOp"> {
  let summary = "Canonicalizes StableHLO operations";
  let dependentDialects = [
    "mlir::tensor::TensorDialect",
  ];
}

def StablehloConvertToSignlessPass : Pass<"stablehlo-convert-to-signless", "ModuleOp"> {
  let summary = "Pass to transform the IR to be on signless integers.";
}

def ShapeLegalizeToStablehloPass : Pass<"shape-legalize-to-stablehlo", "func::FuncOp"> {
  let summary = "Legalize shape-related ops to StableHLO.";
  let description = [{
    An experimental pass that legalizes shape-related ops to StableHLO ops.

    Bringing shape and data computations together via an optional pass will
    make it possible for the StableHLO ecosystem to potentially leverage the
    compilation pipelines that use StableHLO operations to model dynamism.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def StablehloLegalizeDeprecatedOpsPass : Pass<"stablehlo-legalize-deprecated-ops", "func::FuncOp"> {
  let summary = "Legalize deprecated ops to well-supported ops.";
  let description = [{
    The StableHLO v1.0 Opset Deprecations RFC (#2283) proposes to remove
    several redundant ops. This pass helps to evaluate the impact of these op
    removals in various compilation pipelines by legalizing them to their
    long-term supported counterparts.
  }];
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
  let options = [
    Option<"failOnUnusedOps", "fail-on-unused", "bool", /*default=*/"true",
           "Fail on (mostly) unused ops that are deprecated without any fallback.">,
  ];
}

def StablehloLegalizeCompositeToCallPass :
    Pass<"stablehlo-legalize-composite-to-call", "func::FuncOp"> {
  let summary = "Replaces composite ops with a call to their decomposition";
  let description = [{
    Replaces composite ops with a call to their decomposition, e.g. the below:

    ```mlir
    stablehlo.composite "my_namespace.my_op" %arg0, %arg1 {
      decomposition = @bar,
      version = 1,
      composite_attributes = {
        "my_attribute": "my_value"
      }
    }
    ```

    Will become:

    ```mlir
    func.call @bar(%arg0, %arg1)
    ```

    A subset of composites can be excepted from this transformation using the
    "except" flag, e.g.:

    ```bash
    stablehlo-opt --stablehlo-legalize-composite-to-call=except='foo.baz,foo.qux'
    ```
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
  ];
  let options = [
    ListOption<"exceptListOption", "except", "std::string", "Names of composites that should not be replaced with calls.">
  ];
}

def StablehloLegalizeQuantToIntPass : Pass<"stablehlo-legalize-quant-to-int", "mlir::func::FuncOp"> {
  let summary = "Convert from StableHLO quantized ops to StableHLO primitive ops.";

  let description = [{
    Convert StableHLO programs using UniformQuantized types to semantically
    equivalent integer math.
  }];
  let dependentDialects = [
    "mlir::chlo::ChloDialect",
    "mlir::stablehlo::StablehloDialect",
  ];
}

def StablehloLegalizeQuantizedOpToQDQPass : Pass<"stablehlo-legalize-quantized-op-to-qdq", "mlir::func::FuncOp"> {
  let summary = "Decompose StableHLO quantized ops using uniform quantize/dequantize ops.";

  let description = [{
    Decompose StableHLO quantized programs using uniform quantize/dequantize
    operations. For example, the following program

    ```mlir
    func.func @add(%arg0: tensor<!quant.uniform<i8:f32,1.0:0>>, %arg1: tensor<!quant.uniform<i8:f32,2.0:1>>) ->  tensor<!quant.uniform<i8:f32,3.0:2>> {
      %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<!quant.uniform<i8:f32,1.0:0>>, tensor<!quant.uniform<i8:f32,2.0:1>>) -> tensor<!quant.uniform<i8:f32,3.0:2>>
      func.return %0 : tensor<!quant.uniform<i8:f32,3.0:2>>
    }
    ```

    Will become:

    ```mlir
    func.func @add(%arg0: tensor<!quant.uniform<i8:f32, 1.000000e+00>>, %arg1: tensor<!quant.uniform<i8:f32, 2.000000e+00:1>>) -> tensor<!quant.uniform<i8:f32, 3.000000e+00:2>> {
      %0 = stablehlo.uniform_dequantize %arg0 : (tensor<!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<f32>
      %1 = stablehlo.uniform_dequantize %arg1 : (tensor<!quant.uniform<i8:f32, 2.000000e+00:1>>) -> tensor<f32>
      %2 = stablehlo.add %0, %1 : tensor<f32>
      %3 = stablehlo.uniform_quantize %2 : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 3.000000e+00:2>>
      return %3 : tensor<!quant.uniform<i8:f32, 3.000000e+00:2>>
    }
    ```
  }];
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
  ];
}
