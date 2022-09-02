# StableHLO Bytecode

## Currently Encoded Attributes / Types

### Attributes

```
ArgResultAliasAttr {
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
  value: varint (encoded enum)
}

ComparisonTypeAttr
  value: varint (encoded enum)
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

DotDimensionNumbersAttr {
  lhsBatchingDimensions: svarint[]
  rhsBatchingDimensions: svarint[]
  lhsContractingDimensions: svarint[]
  rhsContractingDimensions: svarint[]
}

FftTypeAttr
  value: varint (encoded enum)
}

GatherDimensionNumbersAttr {
  offsetDims: svarint[]
  collapsedSliceDims: svarint[]
  startIndexMap: svarint[]
  indexVectorDim: svarint
}

PrecisionAttr {
  value: varint (encoded enum)
}

RngAlgorithmAttr {
  value: varint (encoded enum)
}

RngDistributionAttr {
  value: varint (encoded enum)
}

ScatterDimensionNumbersAttr {
  updateWindowDims: svarint[]
  insertedWindowDims: svarint[]
  scatterDimsToOperandDims: svarint[]
  indexVectorDim: svarint
}

TransposeAttr {
  value: varint (encoded enum)
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

- `StableHLO_ArrayOfLayoutAttr`
- `StableHLO_BoolElementsAttr`
- `StableHLO_FlatSymbolRefArrayAttr`
- `StableHLO_LayoutAttr`
- `HLO_ComplexTensor`
- `HLO_Complex`
- `HLO_DimensionTensor`
- `HLO_DimensionValue`
- `HLO_Float32Or64`
- `HLO_Float`
- `HLO_Fp32Or64Tensor`
- `HLO_FpOrComplexTensor`
- `HLO_FpTensor`
- `HLO_IntFpOrComplexTensor`
- `HLO_IntOrFpTensor`
- `HLO_IntTensor`
- `HLO_Int`
- `HLO_PredIntOrFpTensor`
- `HLO_PredOrIntTensor`
- `HLO_PredTensor`
- `HLO_Pred`
- `HLO_QuantizedIntTensor`
- `HLO_QuantizedInt`
- `HLO_QuantizedSignedInt`
- `HLO_QuantizedUnsignedInt`
- `HLO_SInt`
- `HLO_ScalarIntTensor`
- `HLO_StaticShapeTensor`
- `HLO_TensorOrTokenOrTuple`
- `HLO_TensorOrToken`
- `HLO_Tensor`
- `HLO_Tuple`
- `HLO_UInt`

Special Cases:
- `StableHLO_ConvolutionAttributes`
  + Despite its name,  is not an attribute and is not encoded.
    Rather, it is a dag which gets expanded into several attributes
    which are all encoded separately.
- `StableHLO_CustomCallApiVersionAttr`
  + This enum is defined strictly as an attribute of `I32EnumAttr`
    and not an `EnumAttr` of the `StablehloDialect`. This differs from
   `FftType` and other enum attributes. Because of this, it is handled by
    the builtin encoding.

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

_Note: Currently all types/attrs are implemented and log only shows 
the dialect name `stablehlo` and the unregistered `stablehlo.frontend_attributes` 
and `stablehlo.sharding` attributes._

```
$ stablehlo-opt -emit-bytecode file.mlir | strings | grep stablehlo
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
