# StableHLO Bytecode

## Basic Example
A minimalist example which only encodes a single attribute in StableHLO can be found [at this commit](https://github.com/openxla/stablehlo/commit/00e3dc98d8d956e5e494be3022df973821e58e91).

There is an image at the bottom of the page which shows difference in the bianry file.

## Currently Encoded Attributes / Types

### Attributes

```
ArgResultAlias {
  argTupleIndices: svarint[]
  resultIndex: svarint
  resultIndex: svarint[]
  isMustAlias: varint
}

ChannelHandleAttr {
  handle: svarint
  type: svarint
}

ComparisonDirectionAttr
  ComparisonDirection: varint
}

ComparisonTypeAttr
  ComparisonType: varint
}

ConvDimensionNumbersAttr {
  inputBatchDimension: svarint
  inputFeatureDimension: svarint
  inputSpatialDimensions: svarint[]
  kernelInputFeatureDimension: svarint
  kernelOutputFeatureDimension: svarint
  kernelSpatialDimensions: svarint[]
  outputBatchDimension: svarint
  outputFeatureDimension: svarint
  outputSpatialDimensions: svarint[]
}

GatherDimensionNumbersAttr {
  lhsBatchingDimensions: svarint[]
  rhsBatchingDimensions: svarint[]
  lhsContractingDimensions: svarint[]
  rhsContractingDimensions: svarint
}

FftTypeAttr
  FftType: varint
}

GatherDimensionNumbersAttr {
  offsetDims: svarint[]
  collapsedSliceDims: svarint[]
  startIndexMap: svarint[]
  indexVectorDim: svarint
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

ScatterDimensionNumbersAttr {
  updateWindowDims: svarint[]
  insertedWindowDims: svarint[]
  scatterDimsToOperandDims: svarint[]
  indexVectorDim: svarint
}

TransposeAttr {
  Transpose: varint
}

TypeExtensionsAttr {
  bounds : svarint[]
}
```

### Types

```
TokenType {
}
```

### Not Included:
The following attributes / types are subclasses of builtin machinery and call
into the bytecode implementations in the Builtin Dialect.

- StableHLO_BoolElementsAttr
- StableHLO_FlatSymbolRefArrayAttr
- StableHLO_ArrayOfLayoutAttr
- StableHLO_LayoutAttr
- HLO_ComplexTensor
- HLO_DimensionTensor
- HLO_DimensionValue
- HLO_Fp32Or64Tensor
- HLO_FpOrComplexTensor
- HLO_FpTensor
- HLO_IntFpOrComplexTensor
- HLO_IntOrFpTensor
- HLO_IntTensor
- HLO_PredIntOrFpTensor
- HLO_PredOrIntTensor
- HLO_PredTensor
- HLO_QuantizedInt
- HLO_QuantizedIntTensor
- HLO_QuantizedSignedInt
- HLO_QuantizedUnsignedInt
- HLO_ScalarIntTensor
- HLO_StaticShapeTensor
- HLO_Tensor
- HLO_TensorOrToken
- HLO_TensorOrTokenOrTuple
- HLO_Tuple

### Still to do:

The following attributes / types are not yet implemented:

- CHLO_ComparisonDirectionAttr
  + CHLO bytecode will come in a future changelist.
- CHLO_ComparisonTypeAttr
  + CHLO bytecode will come in a future changelist.

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

_Note: The following trace is from a previous revision where the `scatter` attribute was not
implemented. Currently all types/attrs are implemented and log only shows 
the dialect name `stablehlo` and the custom `stablehlo.frontend_attributes` and `stablehlo.sharding` properties._

```
$ stablehlo-opt -emit-bytecode file.mlir | strings | grep stablehlo
#stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>
stablehlo
stablehlo.frontend_attributes
stablehlo.sharding
```

### Debugging Bytecode with Traces

Each read/write function called during bytecoding is traced, and can be viewed using the flag: `-debug-only=stablehlo-bytecode`.

```
stablehlo-opt -emit-bytecode -debug-only=stablehlo-bytecode ../tmp.mlir
Called: writeType(mlir::Type, mlir::DialectBytecodeWriter &)::(anonymous class)::operator()(auto) const [type:auto = mlir::stablehlo::TokenType]
Called: writeAttribute(mlir::Attribute, mlir::DialectBytecodeWriter &)::(anonymous class)::operator()(auto) const [attr:auto = mlir::stablehlo::TransposeAttr]
Called: writeAttribute(mlir::Attribute, mlir::DialectBytecodeWriter &)::(anonymous class)::operator()(auto) const [attr:auto = mlir::stablehlo::RngAlgorithmAttr]
Called: writeAttribute(mlir::Attribute, mlir::DialectBytecodeWriter &)::(anonymous class)::operator()(auto) const [attr:auto = mlir::stablehlo::ChannelHandleAttr]
Called: writeAttribute(mlir::Attribute, mlir::DialectBytecodeWriter &)::(anonymous class)::operator()(auto) const [attr:auto = mlir::stablehlo::ChannelHandleAttr]
Called: writeAttribute(mlir::Attribute, mlir::DialectBytecodeWriter &)::(anonymous class)::operator()(auto) const [attr:auto = mlir::stablehlo::TypeExtensionsAttr]
...

stablehlo-opt -emit-bytecode -debug-only=stablehlo-bytecode bytecoded_file.mlir
Called: readComparisonDirectionAttr(mlir::DialectBytecodeReader &) const
Called: readTypeExtensionsAttr(mlir::DialectBytecodeReader &) const
Called: readChannelHandleAttr(mlir::DialectBytecodeReader &) const
Called: readChannelHandleAttr(mlir::DialectBytecodeReader &) const
Called: readRngAlgorithmAttr(mlir::DialectBytecodeReader &) const
```

### Adding Bytecode for a New Type / Attribute

Adding bytecode for a new type or attribute is simple. In the file 
`StablehloBytecode.cpp` search for the term `TO ADD ATTRIBUTE` or `TO ADD TYPE`
depending on the change. Ensure that each location tagged with `TO ADD` 
instructions is addressed. If so, bytecode for the attr/type should be generated
on next call to `stablehlo-opt -emit-bytecode`.


### Encoding `enum class` values
Enum class values can be encoded as their underlying numeric types using `varint`. Currently all enums in StableHLO use `uint32_t` as the underlying value.

## Open Questions
### Is the compatibility guarantee of StableHLO dependent on other dialects?
If something about the way the builtin dialect serializes information changes,
are all other dialects compatibility broken? Since our artifacts will include
bits that need to be (de)serialized using builtin dialect information.

### Any way to reduce the reliance on attribute/op name?
What would happen if we renamed `fft_length` to `fft_length_int` for example, 
is this a breaking change?

Similar quesiton for renaming `cross-replica-sum` to `cross_replica_sum`.
