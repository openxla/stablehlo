# Declarative MLIR Builder APIs

Goal: Provide a builder that abstracts away the notion of location and insertion
point for use cases that construct full graphs from C++.

See `MlirBuilderTest.cpp` for examples.

## Usage

The builders look fairly similar to XlaBuilder's declarative style, see
`MlirBuilderTest.cpp` for a few example programs:

```c++
StablehloModuleBuilder mb;
{  // Build Main Func
  ScopedBuilderLocation loc(mb.get(), fileLineColLoc(mb.get(), "main.mlir"));
  func::FunctionBuilder fb(mb.get(), mb->getLoc(), "main");
  auto type4xi64 = RankedTensorType::get({4}, fb.getOpBuilder().getI64Type());
  auto arg0 = func::Argument(fb, type4xi64);
  auto cst = stablehlo::Constant(fb, 1);
  auto add = chlo::BroadcastAdd(arg0, cst);
  auto topkAndIndices = chlo::TopK(add, 2);
  auto broadcast =
      stablehlo::BroadcastInDim(topkAndIndices[0].getType(), cst, {});
  auto equal = tosa::Equal(topkAndIndices[0], broadcast);
  func::Return(fb, {equal});
}

mb->build()->dump();
// module {
//  func.func @main(%arg0: tensor<4xi64>) -> tensor<2xi1> {
//    %c = stablehlo.constant dense<1> : tensor<i64>
//    %0 = chlo.broadcast_add %arg0, %c : (tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
//    %values, %indices = chlo.top_k(%0, k = 2) : tensor<4xi64> -> (tensor<2xi64>, tensor<2xi32>)
//    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<2xi64>
//    %2 = tosa.equal %values, %1 : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
//    return %2 : tensor<2xi1>
//  }
// }
```

## Technical Details - Add support for a new dialect

### 1. Add a BUILD rule for builder generation

Build rule requires opset tablegen file -
[stablehlo_builder](stablehlo/integrations/cpp/builder/BUILD)
example:

```bazel
gentbl_cc_library(
    name = "stablehlo_builder_inc",
    tbl_outs = {
        "StablehloBuilder.h.inc": ["-gen-builder-decls"],
        "StablehloBuilder.cpp.inc": ["-gen-builder-defs"],
        "StablehloBuilder.md": ["-gen-builder-docs"],
    },
    tblgen = ":mlir_builder_tblgen",
    td_file = "//party/stablehlo:stablehlo/dialect/StablehloOps.td",
    deps = [
        "//party/llvm/llvm-project/mlir:InferTypeOpInterfaceTdFiles",
        "//party/llvm/llvm-project/mlir:OpBaseTdFiles",
        "//party/llvm/llvm-project/mlir:SideEffectInterfacesTdFiles",
        "//party/stablehlo:stablehlo_ops_td_filegroup",
    ],
)
```

This will generate `StablehloBuilder.h.inc` and `StablehloBuilder.cpp.inc` files
that can be used in a cc_library target:

```cpp
$ bazel build -- //stablehlo/integrations/cpp/builder:stablehlo_builder_inc_filegroup
MlirOp Abs(MlirOp &operand);
MlirOp Add(MlirOp &lhs, MlirOp &rhs);
MlirOp AfterAll(MlirBuilder &builder, ArrayRef<MlirOp> inputs);
...
MlirOp BitcastConvert(Type resultType, MlirOp &operand);
MlirOp BroadcastInDim(Type resultType, MlirOp &operand, ::llvm::ArrayRef<int64_t> broadcast_dimensions);
...
```

### 2. Make a cc_library target for generated files

Add a [Builder.h][header] declaration file and [Builder.cpp][impl] impl file:

```cpp
// MyBuilder.h
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h.inc"

// MyBuilder.cpp
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.cpp.inc"
```

[header]:TODO
[impl]:TODO

### 3. Add custom methods for high-priority UX methods

In some cases the default generated method doesn't have adequate UX for a very
important construct that doesn't capture its semantics well statically in ODS
(like `func.func` or `stablehlo.constant`). For these methods, you can add any
declarations to the cc_library to improve UX. These methods should be kept to
a minimum, we should aim to generate as much as possible.

```cpp
// Builder for stablehlo.constant : tensor<i64> scalar
MlirOp Constant(MlirBuilder& builder, int64_t value);

// Builder impl
MlirOp Constant(MlirBuilder& builder, int64_t value) {
  return builder.create<stablehlo::ConstantOp>(DenseIntElementsAttr::get(
      RankedTensorType::get({}, builder.getOpBuilder().getI64Type()), value));
}
```

In a perfect world all ops would capture their semantic information in ODS and
we can generate perfect builders - currently we're missing details like "WhileOp
forwards its operands to each of its regions" or "func op must have its
signature match its region operand / return". These are the cases that require
custom builders, and we should design them in a future-codegenable way.

## Current status

### Outcomes

Some positive outcomes to these APIs:

+ All dialects own their own APIs and can interop pretty easily by abstracting
  away source location and insertion point behind an abstract type.
+ Provides *pretty good* out of the box builder methods (tried with TOSA and
  generated reasonable methods for 73 ops).
+ We can likely make the surface level of builder APIs MLIR-free for g3
  building without visibility.
+ This is extensible to arbitrary types so long as the opsets support type
  inference or accept explicit types as references.
+ Uses the simpler Attribute forms, i.e. int64_t instead of IntegerAttr(64)

### Opset Coverage

When we can't generate a viable interface yet, we skip the op.

With generated build rules, today we are generating:

+ 112/114 StableHLO Ops
+ 48/48 CHLO ops
+ 75/76 TOSA ops
+ 15/16 Shardy ops
+ 3/5 Func ops

```txt
Skipping CaseOp: Variadic regions not supported
Skipping ConstantLikeOp: Attributes must be after operands
Skipping CustomOp: Attributes must be after operands
Skipping NamedComputationOp: Attributes must be after operands
Skipping RngBitGeneratorOp: Attributes must be after operands
Skipping TriangularSolveOp: Attributes have known bad types
Skipping VariableWriteOp: Attributes must be after operands
```

## Next steps

There are some limitations to the current codegen, and we should add support.
These restrictions are captured in a code comment in `MlirBuilderTblgen.cpp`:

```txt
// Some supported patterns:
// - [X] Op has one or more Value operands.
// - [X] Op has one or more results.
// - [X] Op has no required attributes.
// - [X] Op has no operands.
// - [X] Op cannot infer type (take result type as argument).
// - [X] Op has no results.
// - [X] Op has no required MlirOp operands.
// - [X] Op has single variadic operand / result.
// - [X] Op has required attributes.
// - [X] Op has optional attribute followed by non-optional attribute.
// - [ ] Op has multiple operands / results, some variadic.
// - [ ] Op has a region.
// - [ ] Op declares attributes before operands (chlo.constant_like).
// - [ ] Op method is a name conflict (triangular_solve Transpose is enum & fn).
// - [ ] Op uses `FirstAttrDerivedResultTypes` to infer result type (tosa.const)
```

Notably simple Attributes are the next thing to figure out how to support (ints,
i64 arrays, etc), followed by dialect-specific attribute (channel_handle,
result_accuracy).

In general most of these work items are not massive, O(hours) not days.

The potentially trickier design points are:

+ How to build Types?
  + A builder for common upstream types would probably suffice.
  + We probably want our own `MakeShape` method for StableHLO types.
+ How to build ops with regions.
  + Can likely take some hints from FunctionBuilder.

## The rough edges

### Should RegionBuilders come before attributes?

Regions are required arguments (at least we only support required regions
currently), and attributes can be optional. To allow max-default-values, we
push all optional attributes to the end of the function declaration.

The question becomes, where should regions go? Before attributes or after the
last required attribute?

```cpp
// (1) Before attributes
// There's something odd about region builder coming before function name here:
void Func(MlirBuilder &builder, const RegionBuilderCallback &body, ::llvm::StringRef sym_name,
          ::mlir::FunctionType function_type, /*optional*/::mlir::StringAttr sym_visibility = {},
          /*optional*/::mlir::ArrayAttr arg_attrs = {}, /*optional*/::mlir::ArrayAttr res_attrs = {},
          /*optional*/bool no_inline = false);

// (2) After the last required attr
// There's something odd about region being between attributes if specifying optional attrs
void Func(MlirBuilder &builder, ::llvm::StringRef sym_name, ::mlir::FunctionType function_type,
         const RegionBuilderCallback &body, /*optional*/::mlir::StringAttr sym_visibility = {},
        /*optional*/::mlir::ArrayAttr arg_attrs = {}, /*optional*/::mlir::ArrayAttr res_attrs = {},
        /*optional*/bool no_inline = false);
```

Today we chose option (1), high priority functions like Func can provide a
custom method which takes name before region if needed.

### Type Inference Crashes on Failure today

*This is certainly fixable, but unclear if we want to.*

XlaBuilder crashes on failure as well, so maybe not an issue. But XlaBuilder
also tries to implicitly broadcast for many of its implementations, which we
probably should avoid.

### Generated Func API is tricky, may need a way to filter this

Currently we filter FunctionOpInterface ops, they tend to require more custom
logic to update type signatures and register things outside of just creating a
region.

+ Requires overloaded `func::Return` to update the function signature when
  value is returned.
+ Currently requires specifying the FuncType up front which is bad UX.
