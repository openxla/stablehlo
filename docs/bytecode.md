# StableHLO Bytecode

## Basic Example
A minimalist example which only encodes a single attribute in StableHLO can be found [at this commit](https://github.com/openxla/stablehlo/commit/00e3dc98d8d956e5e494be3022df973821e58e91
). 

There is an image at the bottom of the page which shows difference in the bianry file.

## Currently Encoded Attributes / Types

### Attributes

```
FftTypeAttr
  FftType: varint
}

ComparisonTypeAttr
  ComparisonType: varint
}

ComparisonDirectionAttr
  ComparisonDirection: varint
}

TransposeAttr {
  Transpose: varint
}

PrecisionAttr {
  Precision: varint
}

RngAlgorithmAttr {
  RngAlgorithm: varint
}

RngDistributionAttr {
  RngDistribution: varint
}

ChannelHandleAttr {
  handle: varint
  type: varint
}

ConvDimensionNumbersAttr {
  inputBatchDimension: varint
  inputFeatureDimension: varint
  inputSpatialDimensions: Dim
  kernelInputFeatureDimension: varint
  kernelOutputFeatureDimension: varint
  kernelSpatialDimensions: Dim
  outputBatchDimension: varint
  outputFeatureDimension: varint
  outputSpatialDimensions: Dim
}

ScatterDimensionNumbersAttr {
  updateWindowDims: Dim
  insertedWindowDims: Dim
  scatterDimsToOperandDims: Dim
  indexVectorDim: varint
}

GatherDimensionNumbersAttr {
  offsetDims: Dim
  collapsedSliceDims: Dim
  startIndexMap: Dim
  indexVectorDim: varint
}

GatherDimensionNumbersAttr {
  lhsBatchingDimensions: Dim
  rhsBatchingDimensions: Dim
  lhsContractingDimensions: Dim
  rhsContractingDimensions: varint
}

```

### Types

```
TokenType {
}
```

### Maybe todo:
- StableHLO_BoolElementsAttr
  + Only used in window_reversal in ConvolutionOp. Looks like it may be handled
    by builtin dialect?
- StableHLO_FlatSymbolRefArrayAttr
  + Only used in CustomCallOp. DefaultValuedAttr. Not traversed into.
- StableHLO_ArrayOfLayoutAttr
  + Only used in CustomCallOp for operand_layout (StableHLO_ArrayOfLayoutAttr). 
  + Also not hit, maybe because OptionalAttr is not traversed into?
- StableHLO_LayoutAttr
  + Only used in StableHLO_ArrayOfLayoutAttr.
- StableHLO_TypeExtensions 
  + Looks like this isnt hit in serialization, because function types 
    arent seriazlied? Tensor dims aren't?
- StableHLO_ArgResultAlias
  + This class may be unused in StableHLO/MHLO?
- UniformQuantizedSignedInt
  + Also not called in serialization API.

- CHLO_ComparisonDirectionAttr
- CHLO_ComparisonTypeAttr

## Other Notes

### Testing Bytecode with Round Trips
Testing that the round-trip of an MLIR file produces the same results is a good
way to test that the bytecode is implemented properly.

```
$ stablehlo-opt -emit-bytecode stablehlo/tests/print_stablehlo.mlir | stablehlo-opt
```

### Find out what attributes or types are not encoded:
Since attributes and types that don't get encoded are instead stored as strings,
the `strings` command can be used to see what attributes were missed:

```
$ stablehlo-opt -emit-bytecode file.mlir | strings | grep stablehlo
#stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>
#stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>
#stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>
tuple<tensor<3x4xi32>, !stablehlo.token>
tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>>
tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !stablehlo.token>
stablehlo
stablehlo.frontend_attributes
stablehlo.sharding
```

## Open Questions
### Any way to reduce the reliance on attribute/op name?
What would happen if we renamed `fft_length` to `fft_length_int` for example, 
is this a breaking change?

Similar quesiton for renaming `cross-replica-sum` to `cross_replica_sum`.

### What Attributes get encoded?
- Attributes from `BuiltinAttributes` like `ElementsAttr`?
- `StrAttr` or `DefaultValuedStrAttr`?
- Should `I64Attr` be encoded?
- ArrayAttrs like `PrecisionConfigAttr`?

### Some attributes wrapped in other types are not encoded
Is there anything we can do about this, or do we need DefaultValuedAttr to be
modified?

```
DefaultValuedAttr<
  StableHLO_CustomCallApiVersionAttr,
  "::mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL">
```

Or is this not called because CustomCallApiVersion is a Builtin Attribute
under the hood?

### What is the proper way to store int64_t data?
Could memcpy it into a uint64? Not sure if that's valid on all platforms?

### How do to encode array ref params?
Example: `StableHLO_Dim` in `ConvDimensionNumbers`

### Encoding `enum class` values
Enum class values can be encoded as their underlying numeric types using `varint`.