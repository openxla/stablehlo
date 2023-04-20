// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xbf16>, tensor<3xbf16>)
    %1 = call @expected() : () -> tensor<4xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xbf16>, tensor<3xbf16>) -> tensor<4xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xbf16>, tensor<3xbf16>) {
    %0 = stablehlo.constant dense<[[-1.148440e+00, 1.851560e+00, 2.671880e+00], [-1.794430e-02, 3.734380e+00, 4.687500e+00], [-7.593750e+00, 4.082030e-01, -5.937500e-01], [-3.687500e+00, -9.648430e-01, -3.750000e+00]]> : tensor<4x3xbf16>
    %1 = stablehlo.constant dense<[1.976560e+00, 6.757810e-01, 3.906250e+00]> : tensor<3xbf16>
    return %0, %1 : tensor<4x3xbf16>, tensor<3xbf16>
  }
  func.func private @expected() -> tensor<4xf32> {
    %0 = stablehlo.constant dense<[9.41830444, 20.7986984, -17.0530014, -22.589035]> : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

