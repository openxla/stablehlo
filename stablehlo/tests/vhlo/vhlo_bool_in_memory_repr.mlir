// RUN: stablehlo-opt %s --stablehlo-aggressive-folder | FileCheck %s
// RUN: stablehlo-translate %s --serialize --target=1.14.0 | stablehlo-translate --deserialize  | stablehlo-opt --stablehlo-aggressive-folder | FileCheck %s
// RUN-DISABLED(llvm/llvm-project#186178): stablehlo-opt %s -emit-bytecode | stablehlo-opt --stablehlo-aggressive-folder | FileCheck %s

// Bools are packed in bytecode, ensure that their unpacked memory repr matches
// what MLIR infra expects (see llvm/llvm-project#186178, or jax-ml/jax#35762).
func.func public @main() -> (tensor<32xi32>) {
  // CHECK: dense<1> : tensor<32xi32>
  %c = stablehlo.constant dense<true> : tensor<32xi1>
  %0 = stablehlo.convert %c : (tensor<32xi1>) -> tensor<32xi32>
  return %0 : tensor<32xi32>
}
