# StableHLO Bounded Dynamism

[StableHLO](https://github.com/openxla/stablehlo) is an operation set that
expresses ML computations. It has been originally bootstrapped from the [MHLO
dialect](https://github.com/tensorflow/mlir-hlo#meta-hlo-dialect-mhlo),
including inheriting the type and some ops. This RFC aims to describe the
current status and rationale for bounded dynamism constructs in StableHLO.  In
particular, this RFC doesn’t propose any further changes to the current state.

Bounded dynamism allows programs to represent the maximum runtime size that a
particular dynamic dimension of a tensor can have. This makes it possible to run
such programs on platforms that don't support dynamic tensors but could support
it if the upper bounds of tensor dimensions are known at the compilation time.
Applications include:

* Real time inference without having to wait for accumulation up to a particular
  batch size.
* Programs whose intermediate tensor shapes depend on the operands. For example,
  [`stablehlo.dynamic_broadcast_in_dim`](https://github.com/openxla/stablehlo/blob/ff55f9346d54e9e38de807a79f8ae03faffda274/stablehlo/dialect/StablehloOps.td#L1838)
  op but with statically known upper bounds of `output_dimensions` operand.
* Bounded dynamism can also open up performance optimizations opportunities.

## Non Goals

* Provide [value inference](https://github.com/openxla/xla/blob/9e05932a2ceadea080dc9494cfe9d735f94c4e68/xla/client/value_inference.h)
  like utility for producers that want to generate the `set_dimension_size` ops.
  Value inference depends on constant folding for StableHLO ops which is a work
  in progress currently. There will be separate RFC for value inference subject
  to separate approval. Note that producers that generate unbounded programs
  don't need this
  in StableHLO.
* Provide a transformation that converts StableHLO programs to bounded StableHLO
  programs. However, there is a plan to have such a conversion in MHLO, although
  the details of this are out of scope of this RFC.

## Detailed Proposal

### (P1) Bounded tensor type using the encoding field in the RankedTensorType

Bounds of a dynamic tensor are represented using the `TypeExtensionsAttr` in the
`RankedTensorType` encoding field. Bounds in `TypeExtensionsAttr` is an
`int64_t` array of size equal to rank of the tensor. Values corresponding to
static dimensions must be `ShapedType::kDynamicSize` which is printed as `?` in
the IR. Disallowing a static bound value for static dimensions makes the IR
canonical and makes it possible to infer that the dimension is dynamic if the
bound value is static.

The following type represents a 2D tensor, with the size of the 0th dimension
being up to 3 and the size of the 1st dimension being exactly 5:

```mlir
tensor<?x5xf32, #stablehlo.type_extensions<bounds = [3, ?]>>
```

Type compatibility in StableHLO also checks for compatibility of the bounds.
For example, the example type above is compatible with `tensor<2x5xf32>` but not
with `tensor<7x5xf32>` as it doesn’t respect the bound `3` on the first
dimension.

```mlir
func.func @bounds_compatibility(%arg0: tensor<?xf32, #stablehlo.type_extensions<bounds = [3]>>,
                                %arg1: tensor<?xf32, #stablehlo.type_extensions<bounds = [2]>>,
                                %arg2: tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>>,
                                %arg3: tensor<2xf32>,
                                %arg4: tensor<4xf32>) {
  // %arg0 is compatible with %arg1, %arg2 and %arg3 as bounded types could have
  // tensor<2xf32> type during runtime.
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<?xf32, #stablehlo.type_extensions<bounds = [3]>>, tensor<?xf32, #stablehlo.type_extensions<bounds = [2]>>) -> tensor<?xf32, #stablehlo.type_extensions<bounds = [2]>>
  %1 = "stablehlo.add"(%arg0, %arg2) : (tensor<?xf32, #stablehlo.type_extensions<bounds = [3]>>, tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>>) -> tensor<?xf32, #stablehlo.type_extensions<bounds = [3]>>
  %2 = "stablehlo.add"(%arg0, %arg3) : (tensor<?xf32, #stablehlo.type_extensions<bounds = [3]>>, tensor<2xf32>) -> tensor<2xf32>

  // This is illegal as operands have incompatible types. %arg0 can either be
  // tensor<0xf32>, tensor<1xf32>, tensor<2xf32> or tensor<3xf32> at runtime,
  // none of these are compatible with tensor<4xf32>
  %3 = "stablehlo.add"(%arg0, %arg4) : (tensor<?xf32, #stablehlo.type_extensions<bounds = [3]>>, tensor<4xf32>) -> tensor<*xf32>
  func.return
}
```

Currently, the StableHLO dialect uses the MLIR ranked tensor type to represent
bounds. In the future we plan to introduce StableHLO type which supports bounds,
along with a custom pretty printing format.  There will be a separate RFC on
this. Also, the proposal will follow StableHLO backward compatibility policies
so it is safe to use `TypeExtensionsAttr` currently.

### (P2) StableHLO op semantics with bounded operands or results

All ops that support dynamic operands or results can have bounds specified for
them. However, the result types need to be compatible with the inferred result
types. This allows result types to be more generic or specific as long as it is
compatible with the inferred type.

Separately, the StableHLO specification will be updated to cover bounded types
for all the relevant ops.

### (P3) get\_dimension\_size / set\_dimension\_size ops

The `get_dimension_size` op takes a tensor and a dimension index and returns the
runtime size as `tensor<i32>`.

The following example returns the size of the result after concatenating input
that has up to `16` elements with self and returns the runtime size of the
concatenation result.

```mlir
func.func @self_concat_size(%data: tensor<?xi32, #stablehlo.type_extensions<bounds = [16]>>) -> tensor<i32> {
  %concat = "stablehlo.concatenate"(%data, %data) {dimension = 0 : i64}
    : (tensor<?xi32, #stablehlo.type_extensions<bounds = [16]>>,
       tensor<?xi32, #stablehlo.type_extensions<bounds = [16]>>)
      -> tensor<?xi32, #stablehlo.type_extensions<bounds = [32]>>

  %result = stablehlo.get_dimension_size %concat, dim = 0
    : (tensor<?xi32, #stablehlo.type_extensions<bounds = [32]>>) -> tensor<i32>

  func.return %result : tensor<i32>
}
```

The `set_dimension_size` op takes a static or bounded tensor, runtime size and a
dimension index and returns a tensor whose size of the particular dimension is
set to the specified size. This size needs to be less than or equal to the
static size or bound for the dimension. This operation can be thought as either
a slice or pad operation depending on if the earlier dimension size is larger or
smaller, respectively. In case the dimension size is increased, the padded
values are undefined.

In the following example, `set_dimension_size` op is used to set the size of the
first dimension so that it performs a sum reduction on the first `batch_size`
elements in the input. With data argument `[1, 2, 3, 4]` and batch\_size
argument `2`, the following function returns `3` but it returns `6` for the same
data argument when the batch\_size is `3`. The `set_dimension_size` op also sets
the bound on the returned tensor. This bound depends on operand's static size if
the operand shape is static. It is `4` in this example. If the operand dimension
is not static, then the returned tensor has same type as the operand.

```mlir
func.func @dynamic_sum(%data: tensor<4xi32>, %batch_size: tensor<i32>) -> tensor<i32> {
  %dynamic_data = stablehlo.set_dimension_size %data, %batch_size, dim = 0
    : (tensor<4xi32>, tensor<i32>) -> tensor<?xi32, #stablehlo.type_extensions<bounds = [4]>>

  %zero = stablehlo.constant dense<0> : tensor<i32>
  %sum = "stablehlo.reduce"(%dynamic_data, %zero) ({
   ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %add = stablehlo.add %arg1, %arg2 : tensor<i32>
    stablehlo.return %add : tensor<i32>
  }) {dimensions = dense<[0]> : tensor<1xi64>}
    : (tensor<?xi32, #stablehlo.type_extensions<bounds = [4]>>,
       tensor<i32>) -> tensor<i32>
  func.return %sum : tensor<i32>
}
```

### (P4) Aspirational: Migration to unbounded dynamism

In addition to `set_dimension_size` and `get_dimension_size` ops, StableHLO
producers may also use unbounded dynamic ops like `real_dynamic_slice` and
`dynamic_pad` to perform operations on dynamically shaped tensors. For example,
the above `@dynamic_sum` computation can be performed by using the
`real_dynamic_slice` op instead of the `set_dimension_size` op. With that, the
above example can be rewritten as,

```mlir
func.func @dynamic_sum(%data: tensor<4xi32>, %batch_size: tensor<i32>) -> tensor<i32> {
  %start = stablehlo.constant dense<0> : tensor<1xi32>
  %limit = stablehlo.reshape %batch_size : (tensor<i32>) -> tensor<1xi32>
  %strides = stablehlo.constant dense<1> : tensor<1xi32>
  %dynamic_data =  stablehlo.real_dynamic_slice %data, %start, %limit, %strides
    : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)
      -> tensor<?xi32>

  %zero = stablehlo.constant dense<0> : tensor<i32>
  %sum = "stablehlo.reduce"(%dynamic_data, %zero) ({
   ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %add = stablehlo.add %arg1, %arg2 : tensor<i32>
    stablehlo.return %add : tensor<i32>
  }) {dimensions = dense<[0]> : tensor<1xi64>}
    : (tensor<?xi32>, tensor<i32>) -> tensor<i32>
  func.return %sum : tensor<i32>
}
```

Originally, HLO introduced `set_dimension_size` op as it neither had dynamic
types nor dynamic ops. StableHLO dialect doesn't have these limitations and
therefore new users don't need to make use of this low level op unless they are
moving from HLO or MHLO to StableHLO. The TensorFlow and JAX teams believe that
this hypothesis should be correct based on their experiences so far. Dynamic
operations `real_dynamic_slice` and `dynamic_pad` can be used instead.

The following example demonstrates the differences between programs using
unbounded dynamism and bounded dynamism.

```mlir
func.func @slice_with_unbounded_dynamism(%data: tensor<7xf32>, %start: tensor<1xi32>, %limit: tensor<1xi32>) -> tensor<?xf32> {
  %strides = stablehlo.constant dense<1> : tensor<1xi32>
  %result =  stablehlo.real_dynamic_slice %data, %start, %limit, %strides
    : (tensor<7xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)
      -> tensor<?xf32>
  func.return %result : tensor<?xf32>
}
```

```mlir
func.func @slice_with_bounded_dynamism(%data: tensor<7xf32>, %start: tensor<1xi32>, %limit: tensor<1xi32>) -> tensor<?xf32, #stablehlo.type_extensions<bounds = [7]>> {
  // Add padding to avoid OOM access in the following slice op.
  %pad_value = stablehlo.constant dense<0.0> : tensor<f32>
  %padded_data = stablehlo.pad %data, %pad_value, low = [0], high = [7], interior = [0]
    : (tensor<7xf32>, tensor<f32>) -> tensor<14xf32>

  // Extract the largest possible slice starting at the start index.
  %scalar_start = stablehlo.reshape %start : (tensor<1xi32>) -> tensor<i32>
  %padded_result = stablehlo.dynamic_slice %padded_data, %scalar_start, sizes = [7]
    : (tensor<14xf32>, tensor<i32>) -> tensor<7xf32>

  // Remove the extra elements extracted beyond the limit.
  %slice_size = stablehlo.subtract %limit, %start : tensor<1xi32>
  %scalar_size = stablehlo.reshape %slice_size : (tensor<1xi32>) -> tensor<i32>
  %result = stablehlo.set_dimension_size %padded_result, %scalar_size, dim = 0
    : (tensor<7xf32>, tensor<i32>)
      -> tensor<?xf32, #stablehlo.type_extensions<bounds = [7]>>

  func.return %result : tensor<?xf32, #stablehlo.type_extensions<bounds = [7]>>
}

```

Use of unbounded dynamic ops over `set_dimension_size` op has a couple of
benefits:

* Greatly simplifies the lowering from higher level frameworks to StableHLO as
  they don't need to compute upper bounds of dynamic dimensions.
* Makes lowerings to StableHLO hardware agnostic as they don't depend on if the
  hardware requires unbounded or bounded programs.

Benefit of the `set_dimension_size` op:

Given that the runtime size argument of `set_dimension_size` op is required to
be less than or equal to the static size or bound, compiler could separately
track runtime size of the tensor and keep a buffer of fixed size according to
the bound. This helps avoid any data movements for the `set_dimension_size` op
at the cost of extra memory. However, compilers should be making the trade-off
between copy and additional memory based on the hardware capabilities and not
the frameworks. It is possible to lower slice op to `set_dimension_size` op
easily but going in the other direction is tricky. That would require program
analysis to make sure that the size of the buffer is not increased later on.

## Alternatives Considered

### Not having bounded type and/or set\_dimension\_size op

Given the use of unbounded dynamism, could StableHLO just have a function
attribute to store the input bounds instead of having bounded type and
`set_dimension_size` op? This might be possible but this will pose significant
challenges for existing users generating bounded programs. Current proposal
allows users to incrementally move to unbounded dynamism for new implementations
while immediately making use of StableHLO without generating a mix of StableHLO
and MHLO programs.
