# VHLO Checklist

See [`vhlo.md`](vhlo.md) for information about VHLO.

## Contributing Incompatible Changes

All changes with compatibility implications must go through the RFC process.
This includes adding, deprecating, or renaming a feature. Once the RFC is
approved, the following steps must be completed:

### 1. Bump the Version Number in Version.h and Update the Version Log

Prior to updating VHLO ops, attributes, types, or conversions, increment the
minor version number in [Version.h](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/Version.h),
and update the version log in [VhloDialect.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/VhloDialect.td)

Any new VHLO features added would use this bumped version, for example after
bumping `0.10.0 --> 0.11.0`, a new op in [VhloOps.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/VhloOps.td)
would use:

```tablegen
VHLO_Op<"abs_v2", "0.11.0", "current">
```

### 2. Add Required VHLO Implementation and Conversions

The exact code needed to integrate a new feature will vary, but for the most
part the following will need to change:

* For new ops:
  1. Add the op in [VhloOps.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/VhloOps.td)
  1. Add StableHLO → VHLO conversion in [StablehloLegalizeToVhlo.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/transforms/StablehloLegalizeToVhlo.cpp)
  1. Add VHLO → StableHLO conversion in [VhloLegalizeToStablehlo.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/transforms/VhloLegalizeToStablehlo.cpp)
* For new versions of existing ops:
  1. Add the op in [VhloOps.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/VhloOps.td)
  1. Update StableHLO to VHLO mapping in [MapStablehloToVhlo.h](https://github.com/openxla/stablehlo/blob/main/stablehlo/transforms/MapStablehloToVhlo.h)
  1. Add a conversion between new and old op versions in [VhloToVersion.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/transforms/VhloToVersion.cpp)
* For new types or attributes:
  1. Add the type in [VhloTypes.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/VhloTypes.td)
  or the attribute in [VhloAttrs.td](https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/VhloAttrs.td)
  1. Add StableHLO → VHLO conversion in [StablehloLegalizeToVhlo.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/transforms/StablehloLegalizeToVhlo.cpp)
  1. Add VHLO → StableHLO conversion in [VhloLegalizeToStablehlo.cpp](https://github.com/openxla/stablehlo/blob/main/stablehlo/transforms/VhloLegalizeToStablehlo.cpp)

### 3. Add / Update Unit Tests

The contributor of an incompatible change is responsible for both positive and
negative unit tests of the feature, as well as compatibility unit tests.

Compatibility unit testing involves updating [stablehlo_legalize_to_vhlo.mlir](https://github.com/openxla/stablehlo/blob/main/stablehlo/tests/vhlo/stablehlo_legalize_to_vhlo.mlir)
to ensure that StableHLO round trips with the latest version of VHLO, as well
as any additional forward or backward compatibility tests required. For example,
if adding a new op at version `X` with `Y = X - 1`, add a test file like
`vhlo_to_version_downgrade_invalid.0_Y_0.mlir` that shows the op is unsupported
before version `X`. If adding a new version of an op, add a test file like
`vhlo_to_version_downgrade.0_Y_0.mlir` that shows that the op can be downgraded
successfully.

If your op has default attributes, include tests that show that the defaults are
serialized and deserialized correctly.

### 4. Add Versioned Serialization Test

After adding tests to `stablehlo_legalize_to_vhlo.mlir`, copy the versioned test
file with the largest version into a new file at the new version, and add the
new tests to that file as well. You will also need to create an associated
bytecode file using `stablehlo-translate`:

```bash
export TARGET_VERSION=1.X.0
export TARGET_FILENAME=${TARGET_VERSION//./_}
stablehlo-translate --serialize --target=$TARGET_VERSION --strip-debuginfo stablehlo/tests/vhlo/stablehlo_legalize_to_vhlo.$TARGET_FILENAME.mlir > stablehlo/tests/vhlo/stablehlo_legalize_to_vhlo.$TARGET_FILENAME.mlir.bc
```
