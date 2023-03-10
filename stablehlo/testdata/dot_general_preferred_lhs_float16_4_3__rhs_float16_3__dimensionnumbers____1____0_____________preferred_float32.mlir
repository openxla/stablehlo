// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xf16>, tensor<3xf16>)
    %1 = call @expected() : () -> tensor<4xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xf16>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : (tensor<3xf16>) -> tensor<3xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xf32>, tensor<3xf32>) -> tensor<4xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xf16>, tensor<3xf16>) {
    %0 = stablehlo.constant dense<[[-5.639650e-01, 1.504880e+00, 5.531250e+00], [3.902340e+00, -4.386720e+00, -1.755370e-01], [4.156250e+00, 2.948000e-02, 2.759770e+00], [2.771480e+00, -1.776370e+00, -1.197270e+00]]> : tensor<4x3xf16>
    %1 = stablehlo.constant dense<[3.384770e+00, -3.083500e-01, -3.320310e+00]> : tensor<3xf16>
    return %0, %1 : tensor<4x3xf16>, tensor<3xf16>
  }
  func.func private @expected() -> tensor<4xf32> {
    %0 = stablehlo.constant dense<[-20.7383976, 1.514400e+01, 4.8955574, 13.903863]> : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

