# StableHLO Compatibility

StableHLO is a backward compatible ML compute opset inspired by HLO/MHLO.
This document explains the kind and the extent of the compatibility guarantees
that StableHLO provides, based on the process established in
[the compatibility RFC](https://github.com/openxla/stablehlo/tree/main/rfcs/20220912-compatibility.md).

## Versions

The current version of StableHLO can be found in
[Version.h](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/Version.h#:~:text=getCurrentVersion).

The minor version is bumped every time changes to the StableHLO opset or
[the StableHLO serialization format](bytecode.md) are made, and the patch
version is bumped every time we integrate StableHLO downstream, i.e. into the
openxla/xla repository.

## Guarantees

Per the [StableHLO v1.0 Compatibility RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20230623-compatibility.md),
the compatibility window includes the following:

**5 years of backward compatibility:** Portable artifacts serialized by an old
version of libStablehlo have the same semantics* when deserialized by a new
version of libStablehlo if these versions are built from openxla/stablehlo
commits which are less than 5 years apart.

**2 years of forward compatibility:** Portable artifacts serialized by a new
version of libStablehlo have the same semantics* when deserialized by an old
version of libStablehlo if these versions are built from openxla/stablehlo
commits which are less than 2 years apart, unless the program is using new
features introduced since the old version.

\* StableHLO programs are converted to/from portable artifacts via
[compatibility APIs](#apis), and the semantics of these programs are
defined by [the StableHLO spec](spec.md). Consult
[the "Out of scope" section](#out-of-scope) to see examples of what is not
covered by this definition of compatibility.

## APIs

Portable artifacts can be created using either the `stablehlo-translate` tool,
or directly in C++ or Python APIs. Serialization needs a target version of
StableHLO to write an artifact written in `#.#.#` format (See [Version.h](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/Version.h#:~:text=getCurrentVersion)
for current version). Since patch versions do not impact compatibility, any
target with non-zero patch version defaults to zero during serialization.
Deserialization uses the current version of StableHLO to read an artifact.

### `stablehlo-translate`

This is the easiest way to create and read a portable artifact.

```bash
# Write a StableHLO program to a portable artifact
$ stablehlo-translate --serialize file.mlir --target=0.9.0 > portable_artifact.mlir.bc

# Read StableHLO portable artifact
$ stablehlo-translate --deserialize portable_artifact.mlir.bc
```

### C++

For programmatic workflows, StableHLO provides the following compatibility APIs:

```c++
// From: #include "stablehlo/api/PortableApi.h"

// Get the current StableHLO version.
//
// This value can be used as the `targetVersion` argument to
// `serializePortableArtifact`.
std::string getCurrentVersion();

// Get the minimum supported StableHLO version.
//
// This value can be used as the `targetVersion` argument to
// `serializePortableArtifact`.
std::string getMinimumVersion();

// From:  #include "stablehlo/dialect/Version.h"

// CompatibilityRequirement is used to get a viable target version to use for
// `serializePortableArtifact` given a compatibility requirement specified as
// a duration.
//
// New enum values can be added per use case.
//
// Values represent a minimum requirement, i.e. WEEK_4 will return a >=4w
// old version, the specific implementation detail can be updated at any time
// by the community as long as it satisfies the requirement.
//
// Given that integration into XLA is not immediate, coarse intervals work
// better than providing a specific date.
enum class CompatibilityRequirement {
  NONE = 0,     // No compat requirement, use latest version.
  WEEK_4 = 1,   // 1 month requirement
  WEEK_12 = 2,  // 3 month requirement
  MAX = 3,      // Maximum compat, use minimum supported version
};

// Get a viable target version to use for `serializePortableArtifact` for a
// given compatibility requirement. See `CompatibilityRequirement` for
// details.
Version::fromCompatibilityRequirement(CompatibilityRequirement requirement);

// From: #include "stablehlo/dialect/Serialization.h"

// Write a StableHLO program to a portable artifact
// Writes a stable payload for `module` to `os`. If compatibility with a
// previous version of StableHLO is required, provide the required version
// string `#.#.#` for `targetVersion`.
//
// Can fail if `module` cannot be expressed in the `targetVersion` version of
// StableHLO, e.g. if it's using new or removed features, or if it involves
// unsupported dialects.
LogicalResult serializePortableArtifact(ModuleOp module,
                                        StringRef targetVersion,
                                        raw_ostream& os);

// Read StableHLO portable artifact
//
// Can fail if `sourceStr` cannot be expressed in the current version of
// StableHLO, e.g. if it's using incompatible features. Returns nullptr if
// `sourceStr` is invalid or fails to deserialize.
OwningOpRef<ModuleOp> deserializePortableArtifact(StringRef sourceStr,
                                                  MLIRContext* context);
```

See [`stablehlo/api/PortableApi.h`](https://github.com/openxla/stablehlo/blob/main/stablehlo/api/PortableApi.h)
and [`stablehlo/dialect/Serialization.h`](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/Serialization.h)
for full APIs.

See [`StablehloTranslateMain.cpp`](https://github.com/openxla/stablehlo/blob/main/stablehlo/tools/StablehloTranslateMain.cpp#:~:text=serializePortableArtifact)
for example usage of these APIs.

### Python

StableHLO also provides Python bindings to the C++ compatibility APIs:

```python
class StablehloCompatibilityRequirement(enum.Enum):
  NONE,     # No compat, same as get_current_version
  WEEK_4,   # 1mo compat
  WEEK_12,  # 3mo compat
  MAX       # Max compat, same as get_minimum_version

def get_version_from_compatibility_requirement(requirement : StablehloCompatibilityRequirement) -> str: ...
def get_current_version() -> str: ...
def get_minimum_version() -> str: ...
def get_smaller_version(v1 : str, v2 : str) -> str: ...
def get_api_version() -> int: ...
def serialize_portable_artifact(module: ir.Module, target_version: str) -> bytes: ...
def serialize_portable_artifact_str(module: str, target_version: str) -> bytes: ...
def deserialize_portable_artifact(context: ir.Context, artifact: bytes) -> ir.Module: ...
def deserialize_portable_artifact_str(artifact: bytes) -> str: ...
def eval_module(module : ir.Module, args : List[ir.Attribute])
```

See [`StablehloModule.cpp`](https://github.com/openxla/stablehlo/blob/main/stablehlo/integrations/python/StablehloModule.cpp)
for full Python APIs.

See [`stablehlo.py > test_serialization_apis`](https://github.com/openxla/stablehlo/blob/main/stablehlo/integrations/python/tests/stablehlo.py#:~:text=test_serialization_apis)
for roundtrip examples of using the Python Serialization APIs.

## Tests

We have a compatibility suite in [stablehlo/tests/vhlo](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/vhlo)
that involves [a comprehensive compendium of StableHLO ops](https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/vhlo/stablehlo_legalize_to_vhlo.mlir)
serialized for all supported StableHLO versions. For every pull request, we are
testing both backward and forward compatibility - i.e. that the suite can be
deserialized targeting HEAD (backward compatibility), that the compendium
can be serialized targeting all supported StableHLO versions (forward
compatibility), and that the results are syntactically identical to the
original StableHLO programs.

## Future work

**Create a compatibility suite in MLIR upstream:** Using the learnings from
establishing and maintaining StableHLO guarantees, we are planning to contribute
a compatibility suite to MLIR upstream to provide early detection for
accidental compatibility breakages in the MLIR bytecode infrastructure
([#1632](https://github.com/openxla/stablehlo/issues/1632)).

**Use reference implementation:** At the moment, compatibility testing consists
of deserializing the compatibility suite serialized by older versions of
libStablehlo and making sure that deserialization produces syntactically
identical programs. We are planning to also use a reference implementation in
these tests, relaxing the overly onerous requirement of syntactical identity
and comprehensively testing the reference implementation
([#1245](https://github.com/openxla/stablehlo/issues/1245)).

## Out of scope

**Non-portable artifacts:** Compatibility guarantees are only provided for
portable artifacts which are created in [a very specific way](#apis).
Other kinds of artifacts, e.g. prettyprinted representation of the StableHLO
dialect or even bytecode representation of the StableHLO dialect, do not have
compatibility guarantees.

**Unspecced features:** We may make incompatible changes to features which
are historically supported in StableHLO programs but are not yet part of the
StableHLO specification, e.g. we do not provide compatibility guarantees for
unregistered attributes.

**Bug compatibility:** We may make incompatible changes if the implementation in
libStablehlo contradicts the StableHLO specification, e.g. if a definition in
the VHLO dialect is wrong, or if a verifier in the StableHLO dialect does not
match the spec.

**Numerical accuracy:** StableHLO has multiple ops that have
implementation-defined accuracy across consumers and even within the same
consumer across versions. As a result, StableHLO doesn't aim to make
guarantees about numerical accuracy, although this may change in the future
([#1156](https://github.com/openxla/stablehlo/issues/1156)).

**Source compatibility** for C, C++ and Python APIs within libStablehlo is
an aspirational goal. At the moment, we don't offer source compatibility
guarantees, but please let us know if this is an important use case for you,
and we can have a discussion about supporting it
([#1247](https://github.com/openxla/stablehlo/issues/1247)).
