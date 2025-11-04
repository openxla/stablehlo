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

def ChloLegalizeToStablehloPass : Pass<"chlo-legalize-to-stablehlo", "func::FuncOp"> {
  let summary = "Legalizes from CHLO ops flow to StableHLO and Shape ops";
  let dependentDialects = [
    "mlir::shape::ShapeDialect",
    "mlir::stablehlo::StablehloDialect",
    "mlir::tensor::TensorDialect",
  ];
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

def StablehloCanonicalizeDynamismPass : Pass<"stablehlo-canonicalize-dynamism", "func::FuncOp"> {
  let summary = "Canonicalizes dynamic StableHLO ops into static ops.";
  let description = [{
    Replaces dynamic StableHLO ops like DynamicReshapeOp with the corresponding
    static counterparts like `DynamicReshapeOp` to `ReshapeOp` or
    `DynamicBroadcastInDim` to `BroadcastInDim` if all the dynamic elements of =
    these ops are actually constants.

    ```
      %c = stablehlo.constant dense<16> : tensor<1xi32>
      %0 = stablehlo.dynamic_broadcast_in_dim %cst, %c, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<16xf32>

      ==>

      %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16xf32>
    ```
  }];
}

def StablehloCheckShapeAssertionsPass
    : Pass<"stablehlo-check-shape-assertions", "func::FuncOp"> {
  let summary = "Check stablehlo.custom_call @shape_assertion ops.";

  let description = [{
    Validate shape_assertion custom calls.

    Shape assertions validate constraints on dynamic dimensions in StableHLO.
    For example if a framework needed to enforce a constraint of `DimA < 2`,
    the following IR could be emitted:

    ```mlir
    %dimA = <get_dimension_size or input arg> : tensor<i32>
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %is_lt = stablehlo.compare LT %dimA, %c2 : tensor<i1>
    stablehlo.custom_call @shape_assertion(%is_lt) { error_message = "DimA must be less than 2" }
    ```

    After the pass, if the shapes are correct, the `stablehlo.custom_call` 
    will be removed.
  }];

  let options = [
    Option<"enable_shape_assertions", "enable-shape-assertions", "bool",
           /*default=*/"true",
           "Whether shape assertions may generate errors.">
  ];
}

def StablehloCompatibilityExpanderPass : Pass<"stablehlo-compatibility-expander", "mlir::ModuleOp"> {
  let summary = "Compatibility expander for StableHLO operations.";

  let description = [{
    StableHLO ops gets updates or new op is introduced in the latest versions.
    This opt-in pass expands backward compatibility with older StableHLO
    versions by decomposing newer StableHLO operations into equivalent
    operations supported by those older versions.

    Why is this an opt-in pass?

    Occasionally, StableHLO op enhancements are used to greatly simplify the
    handling of certain common patterns in the OpenXLA ecosystem. This
    includes things like TanOp, which has high framework and compiler support,
    as well as gather/scatter batching dimensions, which can be represented
    using slices, but makes sharding much more difficult. For this category of
    new features, we do not offer automatic downgrade, since it may throw away
    important information used in subsequent optimizations. This pass can be
    used to expand these ops based on a target version to maximize compatibility
    at the expense of potentially less optimal compilation.

    ```mlir
    func.func @tan_op_non_complex(%arg0: tensor<4xf64>) -> tensor<4xf64> {
      %1 = stablehlo.tan %arg0 : tensor<4xf64>
      func.return %1 : tensor<4xf64>
    }

    ==>

    func.func @tan_op_non_complex(%arg0: tensor<4xf64>) -> tensor<4xf64> {
      %0 = stablehlo.sine %arg0 : tensor<4xf64>
      %1 = stablehlo.cosine %arg0 : tensor<4xf64>
      %2 = stablehlo.divide %0, %1 : tensor<4xf64>
      return %2 : tensor<4xf64>
    }
    ```
  }];
  let options = [
    Option<"targetVersionOption", "target", "std::string", "",
           "The target version. Must be a version of the form #.#.#.">,
  ];
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
    "mlir::chlo::ChloDialect",
  ];
}

def StablehloComplexMathExpanderPass : Pass<"stablehlo-complex-math-expander", "mlir::func::FuncOp"> {
  let summary = "Expander for StableHLO complex math operations.";

  let description = [{
    StableHLO complex math operations are decompositions using
    StableHLO real math operations.

    This statement is based on the assumption that no hardware exists
    that supports complex numbers nor complex math operations
    natively. This means that the fallback mechanisms on complex math
    operations that compilers may implement, are redundant. With
    enabling this pass, all StableHLO complex math operations will be
    expanded.

    ```mlir
    func.func @sqrt_op_complex(%arg0: tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>> {
      %1 = stablehlo.sqrt %arg0 : tensor<4xcomplex<f64>>
      func.return %1 : tensor<4xcomplex<f64>>
    }

    ==>

    func.func @sqrt_op_complex(%arg0: tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>> {
      TBD
      return %2 : tensor<4xcomplex<f64>>
    }
    ```
  }];
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
    "mlir::chlo::ChloDialect",
  ];
}

def StablehloConvertToSignlessPass : Pass<"stablehlo-convert-to-signless", "ModuleOp"> {
  let summary = "Pass to transform the IR to be on signless integers.";
}

def StablehloLegalizeCompositeToCallPass :
    Pass<"stablehlo-legalize-composite-to-call", "func::FuncOp"> {
  let summary = "Replaces composite ops with a call to their decomposition.";
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

def StablehloLegalizeQDQToQuantizedOpPass : Pass<"stablehlo-legalize-qdq-to-quantized-op", "mlir::func::FuncOp"> {
  let summary = "Fuse (de-quantize, floating-point operation and quantize) pattern into StableHLO quantized operation";

  let description = [{
    Fuse (de-quantize, floating-point operation and quantize) pattern into StableHLO quantized operation
    Note: The pass does not delete any preexisting op.
    For example, the following program

    ```mlir
    func.func @add(%arg0: tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
      %0 = stablehlo.uniform_dequantize %arg0 : (tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<16x16xf32>
      %1 = stablehlo.abs %0 : tensor<16x16xf32>
      %2 = stablehlo.uniform_quantize %1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
      func.return %2 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    }
    ```

    Will become:

    ```mlir
    func.func @add(%arg0: tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>> {
      %0 = stablehlo.uniform_dequantize %arg0 : (tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<16x16xf32>
      %1 = stablehlo.abs %0 : tensor<16x16xf32>
      %2 = stablehlo.abs %arg0 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
      %3 = stablehlo.uniform_quantize %1 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
      return %2 : tensor<16x16x!quant.uniform<u8:f32, 3.400000e+01:16>>
    }
    ```
  }];
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
  ];
}

def StablehloLegalizeQuantizedOpToQDQPass : Pass<"stablehlo-legalize-quantized-op-to-qdq", "mlir::func::FuncOp"> {
  let summary = "Decompose quantized StableHLO operation to (de-quantize, floating-point operation and quantize) pattern.";

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

def StablehloLegalizeQuantToMathPass : Pass<"stablehlo-legalize-quant-to-math", "mlir::func::FuncOp"> {
  let summary = "Convert from StableHLO quantized ops to StableHLO primitive math ops.";

  let description = [{
    Convert StableHLO programs using UniformQuantized types to semantically
    equivalent integer math operations.

    ```mlir
    func.func @add(%arg0: tensor<!quant.uniform<i8:f32,1.0:0>>, %arg1: tensor<!quant.uniform<i8:f32,2.0:1>>) ->  tensor<!quant.uniform<i8:f32,3.0:2>> {
      %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<!quant.uniform<i8:f32,1.0:0>>, tensor<!quant.uniform<i8:f32,2.0:1>>) -> tensor<!quant.uniform<i8:f32,3.0:2>>
      func.return %0 : tensor<!quant.uniform<i8:f32,3.0:2>>
    }
    ```

    Will become:

    ```mlir
    func.func @add(%arg0: tensor<i8>, %arg1: tensor<i8>) -> tensor<i8> {
      %0 = stablehlo.convert %arg0 : (tensor<i8>) -> tensor<f32>
      %cst = stablehlo.constant dense<0.333333343> : tensor<f32>
      %1 = chlo.broadcast_multiply %0, %cst : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %2 = chlo.broadcast_add %1, %cst_0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %3 = stablehlo.round_nearest_even %2 : tensor<f32>
      %4 = stablehlo.convert %3 : (tensor<f32>) -> tensor<i32>
      %5 = stablehlo.convert %arg1 : (tensor<i8>) -> tensor<f32>
      %cst_1 = stablehlo.constant dense<0.666666686> : tensor<f32>
      %6 = chlo.broadcast_multiply %5, %cst_1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_2 = stablehlo.constant dense<1.33333337> : tensor<f32>
      %7 = chlo.broadcast_add %6, %cst_2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %8 = stablehlo.round_nearest_even %7 : tensor<f32>
      %9 = stablehlo.convert %8 : (tensor<f32>) -> tensor<i32>
      %c = stablehlo.constant dense<2> : tensor<i32>
      %10 = chlo.broadcast_add %4, %9 : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %11 = chlo.broadcast_subtract %10, %c : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %c_3 = stablehlo.constant dense<-128> : tensor<i32>
      %c_4 = stablehlo.constant dense<127> : tensor<i32>
      %12 = stablehlo.clamp %c_3, %11, %c_4 : tensor<i32>
      %13 = stablehlo.convert %12 : (tensor<i32>) -> tensor<i8>
      return %13 : tensor<i8>
    }
    ```
  }];
  let dependentDialects = [
    "mlir::chlo::ChloDialect",
    "mlir::stablehlo::StablehloDialect",
  ];
}

def StablehloLegalizeToVhloPass : Pass<"stablehlo-legalize-to-vhlo", "ModuleOp"> {
  let summary = "Legalize StableHLO to VHLO.";
  let description = [{
    Legalize StableHLO to the latest version of ops in VHLO. These ops can then
    be downgraded to older versions of VHLO for forward compatibility using
    `VhloToVersionPass`.

    ```mlir
    stablehlo.exponential %[[ARG0]] <{result_accuracy = DEFAULT}> : tensor<f32>
    # ====>
    "vhlo.exponential_v2"(%[[ARG0]]) <{result_accuracy = #vhlo.DEFAULT_v1}> : !vhlo.tensor_v1<!vhlo.f32_v1>
    ```

    See [vhlo.md > The VHLO dialect](https://github.com/openxla/stablehlo/blob/main/docs/vhlo.md)
    for full details on how VHLO is used to preserve forward and backward
    compatibility.
  }];
  let dependentDialects = ["mlir::vhlo::VhloDialect"];
  let options = [
    Option<"allowOtherDialects", "allow-other-dialects", "bool", /*default=*/"false",
           "Allow serialization to use other (potentially unstable) dialects, inserts unrealized casts between dialects.">,
  ];
}

def StablehloRefineArgumentsPass : Pass<"stablehlo-refine-arguments", "ModuleOp"> {
  let summary = "Refines the argument shapes of the main function.";
  let description = [{
    Modifies the arguments of the main function using the input type signature.
    Wraps arguments in `custom_call @stablehlo.shape_refinement_operand_wrapper`
    to keep the IR valid before shape refinement is run.

    ```
    func.func public @main(%arg0: tensor<?xf32>) -> tensor<?xf32> {
      ...
    }

    ==>

    func.func public @main(%arg0: tensor<16xf32>) -> tensor<?xf32> {
      %c = stablehlo.constant dense<16> : tensor<1xi64>
      %0 = stablehlo.custom_call @stablehlo.shape_refinement_operand_wrapper(%arg0, %c) {...}
        : (tensor<16xf32>, tensor<1xi64>) -> tensor<?xf32>
      ...
    }
    ```

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

def StablehloRefineShapesPass : Pass<"stablehlo-refine-shapes", "ModuleOp"> {
  let summary = "Refines shapes across a StableHLO program.";
  let description = [{
    Walks through a StableHLO program refining shapes within ops.

    The flagship use case for this pass is specializing dynamically-shaped
    programs to static shapes. If a dynamically-shaped StableHLO program has the
    right structure, then updating its argument types from dynamic shapes to
    static shapes and running this pass will propagate static shapes across the
    program.

    This pass removes `custom_call @shape_refinement_operand_wrapper` by
    replacing uses of the result with the operand directly, and propagates
    static shapes throughout the program.

    ```
    %c = stablehlo.constant dense<16> : tensor<1xi64>
    %0 = stablehlo.custom_call @stablehlo.shape_refinement_operand_wrapper(%arg0, %c) {...}
        : (tensor<16xf32>, tensor<1xi64>) -> tensor<?xf32>
    %1 = stablehlo.add %0, %0 : tensor<?xf32>

    ==>

    %1 = stablehlo.add %arg0, %arg0 : tensor<16xf32>
    ```

  Modules valid for shape refinement must have the following properties:

    * All the dynamic shapes depend only on the input shapes (no shape
      dependency on the input array contents). We refer to the operations that
      depend transitively only on the input shapes (e.g., as given by
      `stablehlo.get_dimension_size`) or global constants like the resolved
      values of symbolic integers (i.e. tensor<Axf32> : A = 5), as `dimension`
      operations. All dimension values can be resolved to constants through
      inter-procedural constant folding.
    * Intermediate functions may take a number of token arguments (of type
      !stablehlo.token) at the start of the argument list, followed by some
      global constant arguments which are constant integer scalars, such as the
      resolved values of symbolic integers (i.e. tensor<Axf32> : A = 5).
    * Some intermediate functions may return computations on global constants,
      i.e. `floordiv` on symint values. These functions are indicated by only
      returning constant values after refinement. These functions are inlined.
    * All calls to a single function resolve to the same argument shapes, and no
      recursive / co-recursive function calls are made.
  }];
}

def VhloLegalizeToStablehloPass : Pass<"vhlo-legalize-to-stablehlo", "ModuleOp"> {
  let summary = "Legalize VHLO to StableHLO.";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::quant::QuantDialect",
    "mlir::shape::ShapeDialect",
    "mlir::stablehlo::StablehloDialect",
  ];
}

def VhloToVersionPass : Pass<"vhlo-to-version"> {
  let summary = "Convert between versions of VHLO for compatibility.";
  let description = [{
    Converts between versions of VHLO for IR upgrading and downgrading to
    preserve forward and backward compatibility.

    ```mlir
    "vhlo.exponential_v2"(%[[ARG0]]) <{result_accuracy = DEFAULT}>
    # ==( -target=1.0.0 )==>
    "vhlo.exponential_v1"(%[[ARG0]])
    # ==( -target=1.9.0 )==>
    "vhlo.exponential_v2"(%[[ARG0]]) <{result_accuracy = DEFAULT}>
    ```

    See [vhlo.md > The VHLO dialect](https://github.com/openxla/stablehlo/blob/main/docs/vhlo.md)
    for full details on how VHLO is used to preserve forward and backward
    compatibility.
  }];
  let options = [
    Option<"targetVersionOption", "target", "std::string", "",
           "The target version. Must be a version of the form #.#.# .">,
  ];
  let dependentDialects = ["mlir::vhlo::VhloDialect"];
}

def StablehloWrapInCompositePass : Pass<"stablehlo-wrap-in-composite", "ModuleOp"> {
  let summary = "Wraps a non-composite  StableHLO op in a composite op.";
  let description = [{
    Wraps StableHLO operations in `stablehlo.composite` operations.

    For instance, consider a simple StableHLO program:

    ```mlir
    func.func @main(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>) -> tensor<2xf32> {
      %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
      return %0 : tensor<2xf32>
    }
    ```

    Applying this pass to wrap `stablehlo.add` operations will result in the
    following program:

    ```mlir
    func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
      %0 = stablehlo.composite "stablehlo.add" %arg0, %arg1 {decomposition = @stablehlo.add.impl} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
      return %0 : tensor<2xf32>
    }
    func.func private @stablehlo.add.impl(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
      %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
      return %0 : tensor<2xf32>
    }
    ```

    Notes:

      - The `name` attribute of the generated `stablehlo.composite` operation
        will always be the same as the name of the original operation that was
        wrapped (e.g., if you wrap a `stablehlo.add` operation, the composite
        will also be named `"stablehlo.add"`).
      - The private function that encapsulates the original operation
        (referenced by the `decomposition` attribute of the
        `stablehlo.composite` operation) will be named using the pattern
        `<op_name>.impl[.N]`, where `<op_name>` is the name of the original
        operation, and `N` is a unique integer identifier generated to prevent
        naming conflicts within the module.

    This pass can be used in two distinct ways:

    **Mode 1: Command-line Usage**

    This mode is intended for debugging or testing, as it offers minimal control
    over the attributes of the generated `stablehlo.composite` operations.
    It wraps **all instances** of operations specified using the `op-names`
    (a comma-separated list of operation names) options. The attributes of the
    newly created `stablehlo.composite` operation will be the same as the
    attributes of the original operation.

    **Usage Example:**

    ```bash
    stablehlo-opt input.mlir --stablehlo-wrap-in-composite=op-names='stablehlo.add,stablehlo.mul' -o output.mlir
    ```

    **Mode 2: Programmatic Module-Wide Wrapping with customized Attribute Handling**

    This mode extends programmatic wrapping to the entire module, offering
    fine-grained control over which operations are wrapped and their attributes.
    This is achieved by using the `createStablehloWrapInCompositePass` API,
    which takes an `CompositeAttributeProviderMap` as an argument.

    The `CompositeAttributeProviderMap` is a map that dictates which operations
    should be considered for wrapping and how their attributes should be
    handled. Its semantics are as follows:

    - **Keys (mlir::TypeID):** `TypeID` of an MLIR operation. If an operation's
        `TypeID` matches a key in the map, it becomes a candidate for wrapping.
    - **Values (Lambda Functions):** Lambda function of type
        `std::function<std::optional<NamedAttrList>(Operation*)>`. This function
        is applied to each candidate operation.
        - **Input:** An `mlir::Operation*`, which is an instance of the
          operation type corresponding to the `TypeID` key.
        - **Return Value:** An `std::optional<NamedAttrList>`.
          - If the lambda returns a `NamedAttrList` (wrapped in
            `std::optional`), the operation is wrapped in a
            `stablehlo::composite` operation, and the returned attributes are
            used to set the composite's attributes.
          - If the lambda returns `std::nullopt`, the operation is **not**
            wrapped. This allows for selective wrapping based on custom
            criteria.

    **Example (C++):**

    ```cpp

    stablehlo::CompositeAttributeProviderMap compositeAttributeProviderMap;

    compositeAttributeProviderMap[mlir::TypeID::get<mlir::stablehlo::AddOp>()] =
      [](mlir::Operation* op) -> std::optional<mlir::NamedAttrList> {
      // Custom logic to determine if and how to wrap the operation.
      // Example: Only wrap if it's on a specific type.
      if (mlir::isa<mlir::Float32Type>(op->getOperand(0).getType())) {
        return mlir::NamedAttrList(op->getAttrs());
      }
      return std::nullopt; // Do not wrap.
    };

    pm.addPass(createStablehloWrapInCompositePass(compositeAttributeProviderMap, compositeVersion));
    if (mlir::failed(pm.run(module))) {
      return;
    }
    ```
  }];
  let options = [
    ListOption<"opNamesOption", "op-names", "std::string",
    "The names of the ops to wrap.">,
    Option<"versionOption", "version", "int32_t", /*default=*/"0",
           "The version number of the composite op.">,
  ];
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::stablehlo::StablehloDialect",
  ];
}
