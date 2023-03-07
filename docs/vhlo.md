# The VHLO Dialect

## What is the VHLO Dialect?

The VHLO (Versioned StableHLO) Dialect is used for serialization and stability.
It provides a snapshot of the StableHLO dialect at a given point in time by
versioning individual program elements

VHLO is an **add-only dialect** with **versioned ops, types and attributes**,
which means that once an feature is added to the dialect, it cannot be modified
in any way that impact the semantics.

Any changes to an op, type or attribute require a new version to be added to
the dialect. For example, if a hypothetical `my_op` was added to StableHLO in
0.9.0, but was changed in 0.11.0, we would have the following in VHLO:

```tablegen
// This represents the StableHLO version of the op from 0.9.0 -> 0.10.0
// Both the lower and the upper bound of versions are inclusive
def VHLO_MyOpV1 : VHLO_Op<"my_op_v1", "0.9.0", "0.10.0"> {
  let arguments = (ins
    VHLO_AnyType:$operand
  );
  let results = (outs VHLO_AnyType:$result);
}

// This represents the StableHLO version of the op from 0.11.0 -> current
def VHLO_MyOpV2 : VHLO_Op<"my_op_v2", "0.11.0", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$attr  // New attribute added to StableHLO in 0.11.0
  );
  let results = (outs VHLO_AnyType:$result);
}
```

The StableHLO dialect only has the latest version of the ops. In the running
example, StableHLO dialect at v0.11.0 would only have `StableHLO_MyOp` that has
`operand` and `attr`, while VHLO captures each phase of the op's evolution.

## Why is VHLO useful?

Having a versioned dialect allows us to target previous versions of the
StableHLO opset. This encapsulates forward and backward compatibility in
conversions between ops in the VHLO dialect.

**Forward compatibility:** Forward compatibility is provided by converting
to VHLO and downgrading ops to a target version. If every op/type/attr in a
VHLO program can be downgraded to the target version, it is guaranteed to be
deserializable and convertable to StableHLO on a consumer running a version
greater than or equal to the target version, since VHLO has a snapshot of the
opset at that time.

![Forward compatibility image](images/vhlo/forward_compatibility.png)

This downgrade conversion will fail if ops or features that do not exist in the
previous version of the opset are used. This means that forward compatibility
are discovered on the producer, rather than at runtime.

**Backward compatibility:** Backward compatibility is provided by upgrading
VHLO ops to their latest version (if needed), then converting an op back to
StableHLO. All VHLO programs within the compatibility window are upgradable
to StableHLO, meaning different versions of consumers can deserialize the same
VHLO payload from a previous version.

![Backward compatibility image](images/vhlo/backward_compatibility.png)

More importantly, VHLO is abstracted behind serialization. This means that ML
frameworks (producers) only need to target StableHLO ops, and compiler
backends (consumers) only need to support the latest version, which is the
StableHLO op set. Conversions to and from VHLO are taken care of with machinery
maintainted in the StableHLO repo.

## VHLO Cookbook

MLIR Bytecode Format is the serialization format VHLO uses to offer
compatibility guarantees. See [bytecode.md](https://github.com/openxla/stablehlo/blob/main/docs/bytecode.md)
for more information.

Portable artifacts can be created using either the `stablehlo-opt` tool, or
directly in C++ or Python APIs.

The following examples use a StableHLO program in a file called `file.mlir`
and downgrade to version `0.9.0`. All _"vhlo-to-version"_ related passes are
optional and should only be used if forward compatibility is needed.

### Using the `stablehlo-opt` tool

This is the easiest way to create and read a portable artifact. MLIR passes and
flags can be used to convert and serialize programs.

```bash
# Create a portable artifact
$ stablehlo-translate --serialize file.mlir --target=0.9.0 > portable_artifact.mlir.bc

# Load program (guaranteed within compatibility window)
# Works on both the old and new versions of stablehlo-opt
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

These APIs are still being designed. Please let us know if these APIs are
needed to help us adjust timeline and planning.
