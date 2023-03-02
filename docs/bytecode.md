# StableHLO Bytecode

## MLIR Bytecode Format

StableHLO uses the [MLIR Bytecode Format](https://mlir.llvm.org/docs/BytecodeFormat/)
for serialization.

The MLIR Bytecode Format is a serialization format used to encode MLIR
programs. From the [MLIR RFC](https://discourse.llvm.org/t/rfc-a-binary-serialization-format-for-mlir/63518),
it was built for "the benefits that a binary format brings to the table; namely
serialization speed and size, mmap capabilities, more easily enabled
versioning, etc." Performance, serialization size, and memory tests were run
using large test from various dialects to validate the format. MLIR bytecode
was not specifically built to make MLIR stable, but the RFC notes that it would
not be difficult to build stability on top of this format, which we
successfully did for StableHLO in the [StableHLO Compatibility RFC](https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md).

## VHLO Attribute / Type Encodings

MLIR Bytecode Format allows dialects to specify custom encodings for dialect
specific types and attributes. VHLO is the stable serialization dialect for
StableHLO. As such, VHLO type and attribute bytecode encodings are maintained
in the StableHLO repo:

**Attributes:** See `vhlo_encoding::AttributeCode` in [`VhloBytecode.cpp`](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/VhloBytecode.cpp)

**Types:** See `vhlo_encoding::TypeCode` in [`VhloBytecode.cpp`](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/VhloBytecode.cpp)

See [vhlo.md](vhlo.md) for more details and instructions for generating
and loading stable bytecode artifacts.
