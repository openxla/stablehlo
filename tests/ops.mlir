// RUN: stablehlo-opt %s | FileCheck %s

// CHECK-LABEL: @dummy
ml_program.func @dummy() -> tensor<f32> {
  %0 = "stablehlo.dummy"() : () -> tensor<f32>
  ml_program.return %0 : tensor<f32>
}
