// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f32>>
    %1 = call @expected() : () -> tensor<2x3xcomplex<f32>>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.638406217,-3.81621552), (-0.575746477,-0.703999698), (0.460462809,6.67921638)], [(-0.901939809,3.45682478), (3.4841466,0.576016605), (-3.10126567,-5.34526205)]]> : tensor<2x3xcomplex<f32>>
    return %0 : tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<2x3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.638406217,-3.81621552), (-0.575746477,-0.703999698), (0.460462809,6.67921638)], [(-0.901939809,3.45682478), (3.4841466,0.576016605), (-3.10126567,-5.34526205)]]> : tensor<2x3xcomplex<f32>>
    return %0 : tensor<2x3xcomplex<f32>>
  }
}
