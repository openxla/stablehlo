# StableHLO Bounded Dynamism

[StableHLO](https://github.com/openxla/stablehlo) is an operation set that expresses ML computations. It has been originally bootstrapped from [the MHLO dialect](https://github.com/tensorflow/mlir-hlo#meta-hlo-dialect-mhlo), including inheriting the type and some of the ops. This RFC aims to describe the current status and future plans for bounded dynamism in StableHLO. In particular, this RFC doesn’t propose any further major changes to the current state. The only minor change proposed is allowing `i64` typed result for `get_dimension_size` op and `i64` type for `set_dimension_size` `size` operand. This is further described in P3.

Bounded dynamism allows computations to encode the maximum runtime size of dynamic dimensions of tensors. This makes it possible to run particular computations on hardware that doesn’t support dynamic tensors and needs hints on the largest possible. For example, real time inference might require support for dynamic batch so that inference doesn’t need to wait for accumulation up to the static batch size. Similarly, during training the last batch in a dataset might be smaller than the max batch size if the number of examples in the dataset is not divisible by the batch size.


# Proposal


## (P1) Use RankedTensorType encoding field for bounds

Bounds for a dynamic tensor are represented using the `TypeExtensionsAttr` using the encoding field in the `RankedTensorType`. Bounds in `TypeExtensionsAttr` is an `int64_t` array of size equal to rank of the tensor. Values corresponding to static dimensions must be `ShapedType::kDynamicSize` which is `-1`.

For example, the following type represents a 2D tensor with up to `3` rows and exactly `5` columns.


```
tensor<?x5xf32, #stable_hlo.type_extensions<bounds = [3, -1]>>
```


Type compatibility in StableHLO also checks for compatibility of the bounds. Two types are compatible if there exists a runtime tensor that could match both the types. So, two types with different bounds are compatible but a type with bounds that is lower than the static size in the other type are not compatible. The above example type is compatible with `tensor<2x5xf32>` but not with `tensor<7x5xf32>` as it doesn’t respect the bound `3` on the first dimension.

Currently, StableHLO dialect is using the MLIR core ranked tensor type to encode bounds. It should be noted that there is a plan to introduce a custom StableHLO type in the future that could natively support bounds along with custom pretty printing format. There will be a separate RFC describing all the details. Also, the proposal will follow StableHLO backward compatibility policies so it is safe to use this type currently.


## (P2) StableHLO op semantics with bounded operands or results

All ops that support dynamic operands or results can have bounds specified for them. However, the result types needs to be compatible with the inferrable result types using the operands and attributes. This allows result types to be more generic or specific as long as it is compatible with the inferred type.


## (P3) get\_dimension\_size / set\_dimension\_size ops

The `get_dimension_size` op takes a tensor and dimension index as operands and returns the actual size of the dimension at runtime as `i32` type scalar.

The following example returns the size of the result after concatenating input that has up to `16` elements with self and returns the actual runtime size of the concatenation result.


```
func.func @self_concat_size(%data: tensor<?xi32, #stablehlo.type_extensions<bounds = [16]>>) -> tensor<i32> {
  %concat = "stablehlo.concatenate"(%data, %data) {dimension = 0 : i64}
    : (tensor<?xi32, #stablehlo.type_extensions<bounds = [16]>>, 
       tensor<?xi32, #stablehlo.type_extensions<bounds = [16]>>)
      -> tensor<?xi32, #stablehlo.type_extensions<bounds = [32]>>

  %result = "stablehlo.get_dimension_size"(%concat) {dimension = 0 : i64}
    : (tensor<?xi32, #stablehlo.type_extensions<bounds = [32]>>) -> tensor<i32>
  
  func.return %result : tensor<i32>
}
```


The `set_dimension_size` op takes a tensor, runtime size and dimension index as operands and returns a tensor whose logical size of the particular dimension is set to the specified value. This size needs to be less than or equal to the static size, if available. 

In the following example, set dimension size is used to set the logical size of the first dimension. With input `[1, 2, 3, 4]`, it returns `3` with batch\_size equal to `2` but returns `6` with batch\_size `3`.


```
func.func @dynamic_sum(%data: tensor<4xi32>, %batch_size: tensor<i32>) -> tensor<i32> {
  %dim = stablehlo.constant dense<0> : tensor<i32>
  %dynamic_data =  "stablehlo.set_dimension_size"(%data, %batch_size, %dim)
    : (tensor<4xi32>, tensor<i32>, tensor<i32>) 
      -> tensor<?xi32, #stablehlo.type_extensions<bounds = [4]>>
  
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


Currently, `get_dimension_size` result` and set_dimension_size `size operand only supports `i32` type for legacy reasons. This is not consistent with other types for dimension sizes so it should be expanded to also allow `i64` typed operand for `size`. The default result type for `get_dimension_size` should be `i64` in the shape inference functions but allows `i32` types as well.


## (P4) Prefer generic dynamic ops over set\_dimension\_size op

Note that in the above example `@dynamic_sum`, the same computation can be done by using the `slice` op instead of the `set_dimension_size` op. Note that the `slice` op is not a drop in replacement for the `set_dimension_size` op as it doesn’t allow setting the dimension size to higher than the existing logical size. However, extending size is not generally required in real world programs. Therefore, it is preferable to use relevant dynamic ops over restricting the StableHLO program to have all static or bounded types. This applies to hardware like XLA:TPU as well that requires programs to be either static or bounded. But, this need not happen at the StableHLO level and it could have dynamic types that are then refined to bounded types in later stages of the compilation. This way StableHLO producers can be hardware and compiler agnostics and semantics are obvious to anyone not familiar with bounded dynamism ops. However, this approach won’t work if the intended consumer of StableHLO doesn’t support dynamic ops by converting inputs to bounded versions internally. In that case, `set_dimension_size` op would be the only way to represent bounded dynamism.
