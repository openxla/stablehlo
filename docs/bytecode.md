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

## Creating portable artifacts

Portable artifacts can be created using either the `stablehlo-translate` tool,
or directly in C++ or Python APIs. Serialization needs a target version of
StableHLO to write an artifact (the current version is 0.9.0). Deserialization
uses the current version of StableHLO to read an artifact.

### Using the `stablehlo-translate` tool

This is the easiest way to create and read a portable artifact.

```bash
# Write a StableHLO program to a portable artifact
$ stablehlo-translate --serialize file.mlir --target=0.9.0 > portable_artifact.mlir.bc

# Read StableHLO portable artifact
$ stablehlo-translate --deserialize portable_artifact.mlir.bc
```

### Using C++ APIs

For programmatic workflows, StableHLO provides the following APIs to create
portable artifacts:

```c++
// From: #include "stablehlo/dialect/Serialization.h"

// Write a StableHLO program to a portable artifact
LogicalResult serializePortableArtifact(ModuleOp module,
                                        StringRef targetVersion,
                                        raw_ostream& os);

// Read StableHLO portable artifact
OwningOpRef<ModuleOp> deserializePortableArtifact(StringRef sourceStr,
                                                  MLIRContext* context);
```

### Using Python APIs

In the near future, we are also planning to add Python APIs for serialization
and deserialization ([#1301](https://github.com/openxla/stablehlo/issues/1301)).
