# StableHLO Bytecode

## Currently Encoded Attributes / Types

### StableHLO Attributes and Types

Documentation on the structure of the encoded attributes and types can be found in the
following code comments:

**Attributes:** See `stablehlo_encoding::AttributeCode` in `StablehloBytecode.cpp`
[[link](https://github.com/openxla/stablehlo/search?q=filename%3AStablehloBytecode+AttributeCode)]

**Types:**
See `stablehlo_encoding::TypeCode` in `StablehloBytecode.cpp`
[[link](https://github.com/openxla/stablehlo/search?q=filename%3AStablehloBytecode+TypeCode)]

### CHLO Attributes and Types

Documentation on the structure of the encoded attributes and types can be found in the
following code comments:

**Attributes:** See `chlo_encoding::AttributeCode` in `ChloBytecode.cpp`
[[link](https://github.com/openxla/stablehlo/search?q=filename%3AChloBytecode+AttributeCode)]

**Types:** See `chlo_encoding::TypeCode` in `ChloBytecode.cpp`
[[link](https://github.com/openxla/stablehlo/search?q=filename%3AChloBytecode+TypeCode)]

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

**Special Cases:**
- `StableHLO_ConvolutionAttributes`
  + Despite its name,  is not an attribute and is not encoded.
    Rather, it is a dag which gets expanded into several attributes
    which are all encoded separately.
- `StableHLO_CustomCallApiVersionAttr`
  + This enum is defined strictly as an attribute of `I32EnumAttr`
    and not an `EnumAttr` of the `StablehloDialect`. This differs from
   `FftType` and other enum attributes. Because of this, it is handled by
    the builtin encoding.

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

Each read/write function called during bytecoding is traced, and can be viewed using the flag `-debug-only=stablehlo-bytecode` for StableHLO and `-debug-only=chlo-bytecode` for CHLO.

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
`StablehloBytecode.cpp` or `ChloBytecode.cpp` search for the term `TO ADD ATTRIBUTE` or `TO ADD TYPE`
depending on the change. Ensure that each location tagged with `TO ADD` 
instructions is addressed. If so, bytecode for the attr/type should be generated
on next call to `stablehlo-opt -emit-bytecode`. This can be verified using the proper bytecode trace.

### Encoding `enum class` values
Enum class values can be encoded as their underlying numeric types using `varint`. Currently all enums in StableHLO use `uint32_t` as the underlying value.
