# StableHLO Bytecode

## MLIR bytecode format

StableHLO uses the [MLIR Bytecode Format](https://mlir.llvm.org/docs/BytecodeFormat/)
for serialization.

The MLIR Bytecode Format is a serialization format used to encode MLIR
programs. From the [MLIR RFC](https://discourse.llvm.org/t/rfc-a-binary-serialization-format-for-mlir/63518),
it was built for "the benefits that a binary format brings to the table; namely
serialization speed and size, mmap capabilities, more easily enabled
versioning, etc." Performance, serialization size, and memory tests were run
using large test from various dialects to validate the format.

MLIR bytecode was not specifically built to make MLIR stable, but the MLIR RFC
notes that it would be possible to provide compatibility guarantees on top of
this format, which we successfully did for StableHLO
(see [compatibility.md](compatibility.md)).
