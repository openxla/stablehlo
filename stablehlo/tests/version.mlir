// RUN: stablehlo-opt --version | FileCheck %s
// RUN: stablehlo-translate --version | FileCheck %s

// The embedded revision string varies per build and is not matched exactly.
// CHECK: LLVM version
// CHECK: StableHLO revision
// CHECK: StableHLO VHLO opset version {{[0-9]+\.[0-9]+\.[0-9]+}}
