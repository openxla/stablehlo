//===--------------------------------------------------------------------===//
// Empty Target

// RUN: not stablehlo-opt --vhlo-to-version %s 2>&1 | FileCheck %s --check-prefix=EMPTY_TARGET
// EMPTY_TARGET: No target version specified.
// EMPTY_TARGET-NEXT: Target version must be of the form `#.#.#`.

//===--------------------------------------------------------------------===//
// Future Target

// RUN: not stablehlo-opt --vhlo-to-version='target=100.10.10' %s 2>&1 | FileCheck %s --check-prefix=FUTURE_TARGET
// FUTURE_TARGET: target version 100.10.10 is greater than current version

//===--------------------------------------------------------------------===//
// Below Minimum Target

// RUN: not stablehlo-opt --vhlo-to-version='target=0.0.0' %s 2>&1 | FileCheck %s --check-prefix=MINIMUM_TARGET
// MINIMUM_TARGET: target version 0.0.0 is less than minimum supported

//===--------------------------------------------------------------------===//
// Not Version Target

// RUN: not stablehlo-opt --vhlo-to-version='target=x.y.z' %s 2>&1 | FileCheck %s --check-prefix=NOT_VERSION_TARGET
// NOT_VERSION_TARGET: Invalid target version argument 'x.y.z'
