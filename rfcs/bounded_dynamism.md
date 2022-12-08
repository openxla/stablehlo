# StableHLO Bounded Dynamism

[StableHLO](https://github.com/openxla/stablehlo) is an operation set that
expresses ML computations. It has been originally bootstrapped from the [MHLO
dialect](https://github.com/tensorflow/mlir-hlo#meta-hlo-dialect-mhlo),
including inheriting the type and some ops. This RFC aims to describe the
current status and rationale for bounded dynamism constructs in StableHLO and
provides recommendations to StableHLO producers and consumers. In particular,
this RFC doesn’t propose any further changes to the current state but this RFC
should still be used to revisit those decisions given that those weren't
reviewed.

Bounded dynamism allows programs to represent the maximum runtime size that a
particular dynamic dimensions of tensor can have. This makes it possible to run
such programs on hardware that don't support dynamic tensors but could support
it if the upper bounds of tensor dimensions are known at the compilation time.
With bounded dynamism, real time inference systems don't need to wait for
accumulation up to a particular batch size on these hardware. Bounded dynamism
also makes it possible to support programs whose intermediate tensor shapes
depend on the inputs. For example,
[`stablehlo.dynamic_broadcast_in_dim`](https://github.com/openxla/stablehlo/blob/ff55f9346d54e9e38de807a79f8ae03faffda274/stablehlo/dialect/StablehloOps.td#L1838)
op but with statically known upper bounds of `output_dimensions` operand. Even
on hardware that supports dynamic tensors, bounded dynamism can open up
opportunities of performance optimizations.

# Recommendations

## StableHLO Producers

* Producers should use bounded tensor type representation as described in P1.
* Producers are encouraged to use unbounded dynamic operations for reasons
  described in P4.
* Producers can still use `get_dimension_size` and `set_dimension_size` ops
  described in P3 for the ease of transition to StableHLO and faster adoption of
  StableHLO.

## StableHLO Consumers

* Consumers should aim to support unbounded programs and can optionally make use
  of bounds on tensors for optimizations.
* Consumers that support unbounded programs can safely ignore the bounds
  completely without affecting the correctness.
* Consumers that only support bounded programs could first transform the given
  program to a bounded one through program analysis.
* Consumers can choose to not support `get_dimension_size` and
  `set_dimension_size` ops until there is a motivating use-case.

# Non Goals

* Provide [value inference](https://github.com/openxla/xla/blob/9e05932a2ceadea080dc9494cfe9d735f94c4e68/xla/client/value_inference.h)
  like utility for producers that want to generate the `set_dimension_size` op.
  Value inference depends on constant folding for StableHLO ops which is a work
  in progress currently. Value inference will be designed and provided
  separately in the future. Note that producers that generate unbounded programs
  don't need this in StableHLO.
* Provide a transformation that converts StableHLO programs to bounded StableHLO
  programs, if possible. There is a plan to have such a conversion in MHLO and
  StableHLO users can utilize that by round tripping to MHLO. Details of this
  are outside the scope of this RFC.

# Detailed Proposal


## (P1) Bounded tensor type using the encoding field in the RankedTensorType

Bounds for a dynamic tensor are represented using the `TypeExtensionsAttr` in
the `RankedTensorType` encoding field. Bounds in `TypeExtensionsAttr` is an
`int64_t` array of size equal to rank of the tensor. Values corresponding to
static dimensions must be `ShapedType::kDynamicSize` which is printed as `?` in
the IR. Disallowing a static bound value for static dimensions makes the IR
canonical and makes it possible to infer that the dimension is dynamic if the
bound value is static.

The following type represents a 2D tensor with up to `3` rows and exactly `5`
columns.

```
tensor<?x5xf32, #stable_hlo.type_extensions<bounds = [3, ?]>>
```

Type compatibility in StableHLO also checks for compatibility of the bounds. Two
types are compatible if there exists a runtime tensor that could match both the
types. So, two types with different bounds are compatible but a type with bounds
that is lower than the static size in the other type are not compatible. The
above example type is compatible with `tensor<2x5xf32>` but not with
`tensor<7x5xf32>` as it doesn’t respect the bound `3` on the first dimension.

Currently, the StableHLO dialect is using the MLIR core ranked tensor type to
represent bounds. It should be noted that there is a plan to introduce a custom
StableHLO type in the future that could natively support bounds along with
custom pretty printing format. There will be a separate RFC on this. Also, the
proposal will follow StableHLO backward compatibility policies so it is safe to
use this type currently.


## (P2) StableHLO op semantics with bounded operands or results

All ops that support dynamic operands or results can have bounds specified for
them. However, the result types need to be compatible with the inferred result
types. This allows result types to be more generic or specific as long as it is
compatible with the inferred type.

Separately, StableHLO specification will be updated to cover bounded types for
all the relevant ops.

## (P3) get\_dimension\_size / set\_dimension\_size ops

The `get_dimension_size` op takes a tensor and dimension index as operands and
returns the actual size of the dimension at runtime as an `i32` type scalar.

The following example returns the size of the result after concatenating input
that has up to `16` elements with self and returns the actual runtime size of
the concatenation result.

```
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

The `set_dimension_size` op takes a bounded tensor, runtime size and dimension
index as operands and returns a tensor whose logical size of the particular
dimension is set to the specified size. This size needs to be less than or equal
to the static size or bound for the dimension. This operation can be thought as
either a slice or pad operation depending on if the earlier logical dimension
size is larger or smaller, respectively. In case the dimension size is
increased, the padded values are undefined.

In the following example, `set_dimension_size` op is used to set the logical
size of the first dimension so that it performs a sum reduction on the first
`batch_size` elements in the input. With data argument `[1, 2, 3, 4]` and
batch\_size argument `2`, the following function returns `3` but it returns `6`
for the same data argument when the batch\_size is `3`.

```
func.func @dynamic_sum(%data: tensor<4xi32>, %batch_size: tensor<i32>) -> tensor<i32> {
  %dynamic_data = stablehlo.set_dimension_size %data, %batch_size, dim = 0
    : (tensor<4xi32>, tensor<i32>) -> tensor<?xi32, #stablehlo.type_extensions<bounds = [4]>>

  %zero = stablehlo.constant dense<0> : tensor<i32>
  %sum = "stablehlo.reduce"(%dynamic_data, %zero) ({
   ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %add = stablehlo.add %arg1, %arg2 : tensor<i32>
    "stablehlo.return"(%add) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>}
    : (tensor<?xi32, #stablehlo.type_extensions<bounds = [4]>>,
       tensor<i32>) -> tensor<i32>
  func.return %sum : tensor<i32>
}
```

## (P4) Prefer generic dynamic ops over set\_dimension\_size op

Note that in the above example of `@dynamic_sum` function, the same computation
can be done by using the `real_dynamic_slice` op instead of the
`set_dimension_size` op. The following example demonstrates this.

```
func.func @dynamic_sum(%data: tensor<4xi32>, %batch_size: tensor<i32>) -> tensor<i32> {
  %start = stablehlo.constant dense<0> : tensor<1xi32>
  %limit = stablehlo.reshape %batch_size : (tensor<i32>) -> tensor<1xi32>
  %strides = stablehlo.constant dense<0> : tensor<1xi32>
  %dynamic_data =  stablehlo.real_dynamic_slice %data, %start, %limit, %strides
    : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)
      -> tensor<?xi32>

  %zero = stablehlo.constant dense<0> : tensor<i32>
  %sum = "stablehlo.reduce"(%dynamic_data, %zero) ({
   ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %add = stablehlo.add %arg1, %arg2 : tensor<i32>
    "stablehlo.return"(%add) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>}
    : (tensor<?xi32>, tensor<i32>) -> tensor<i32>
  func.return %sum : tensor<i32>
}
```

Originally, XLA HLO introduced `set_dimension_size` op as it neither had dynamic
types nor dynamic ops. StableHLO dialect doesn't have these limitations and
therefore new users don't need to make use of this low level op. The TensorFlow
and JAX teams believe belives that this hypothesis should be correct based on
their experiences so far. Dynamic operations `real_dynamic_slice` and
`dynamic_pad` can be used instead.

Use of dynamic ops over `set_dimension_size` op has various benefits:

* Greatly simplifies lowering from higher level frameworks to StableHLO as they
  don't need to make use value inference to compute bounds or generate low level
  ops.
* Opens up the opportunity to share conversion to bounded programs between
  frameworks and compilers. Therefore, frameworks can immediately target new
  hardware requiring bounded programs even if they didn't already support that.
  Data dependent bounded dynamism won't require any changes and input dependent
  bounded dynamism can be supported by just specifying the bounds on the inputs.
* Makes lowerings to StableHLO hardware agnostic and they don't depend on if the
  compiler requires unbounded or bounded programs.
* Reduces the potential confusion in making use of `set_dimension_size` as the
  users are generally not familiar with this op and also the semantics are also
  not intuitive.

It is true that the `set_dimension_size` semantics allows making in-place
updates. However, compilers should be making the trade-off between copy and
additional memory based on the hardware capabilities. It is also possible to
lower slice op to `set_dimension_size` op easily but going in the other
direction is tricky. That would require program analysis to make sure that the
logical size of the buffer is not increased later on.

# Alternatives Considered

## Not having bounded type and/or set\_dimension\_size op

Given the recommendation of using unbounded dynamism, could StableHLO just have
a function attribute to store the input bounds instead of having bounded type
and `set_dimension_size` op? This might be possible but this will pose
significant challenges for existing users generating bounded programs. Current
proposal allows users to incrementally move to the recommended approach for new
implementations while immediately making use of StableHLO without generating a
mix of StableHLO and MHLO programs.

It is true that having the bounded type and `set_dimension_size` op introduces
some complexity but given that the bounds are optional, users that don't care
about bounded dynamism don't need to worry about these. All the code complexity
is limited to the StableHLO shape functions. These also affects the op
specifications but these should be intuitive to users based on the op semantics
and making use of StableHLO shape functions should hide that as well.
