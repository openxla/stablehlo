# StableHLO Bytecode

## MLIR Bytecode Format

StableHLO uses the [MLIR Bytecode Format](https://mlir.llvm.org/docs/BytecodeFormat/)
for artifact serialization.

The MLIR Bytecode Format is a serialization format used to encode MLIR
programs. It was built with high consideration for serialization
and deserialization performance, disk and memory footprint, mmap-ability.
Performance, serialization size, and memory tests were run using test
files from upstream dialects and larger CIRCT test files to validate the
format. MLIR bytecode was not specifically built to make MLIR stable, but
the [MLIR RFC](https://discourse.llvm.org/t/rfc-a-binary-serialization-format-for-mlir/63518)
notes that it would not be difficult to build stability on top of this format,
which we successfully did for StableHLO in the [StableHLO Compatibility RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20220912-compatibility.md).

## VHLO Attribute / Type Encodings

MLIR Bytecode Format allows dialects to specify custom encodings for dialect
specific types and attributes. VHLO is the stable serialization dialect for
StableHLO. As such, VHLO type and attribute bytecode encodings are maintained
in the StableHLO repo:

**Attributes:** See `vhlo_encoding::AttributeCode` in `VhloBytecode.cpp`
[[link](https://github.com/openxla/stablehlo/search?q=filename%3AVhloBytecode+AttributeCode)]

**Types:** See `vhlo_encoding::TypeCode` in `VhloBytecode.cpp`
[[link](https://github.com/openxla/stablehlo/search?q=filename%3AVhloBytecode+TypeCode)]

See [vhlo.md](vhlo.md) for more details and instructions for generating
and loading stable bytecode artifacts.
