// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xbf16>, tensor<3x6xbf16>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xbf16>, tensor<3x6xbf16>) {
    %0 = stablehlo.constant dense<[[6.796880e-01, 1.968750e+00, 3.125000e-01], [4.156250e+00, 2.011720e-01, 1.898440e+00], [1.257810e+00, 4.093750e+00, -1.453130e+00], [2.156250e+00, -3.484380e+00, -6.289060e-01]]> : tensor<4x3xbf16>
    %1 = stablehlo.constant dense<[[4.492190e-01, 4.937500e+00, 3.234380e+00, 6.250000e-01, 1.328130e+00, 5.273440e-02], [-3.250000e+00, -2.484380e+00, -1.578130e+00, 9.765620e-01, 2.796880e+00, -4.375000e+00], [-4.000000e+00, -3.437500e+00, 2.734380e+00, 3.437500e+00, 5.718750e+00, -1.390630e+00]]> : tensor<3x6xbf16>
    return %0, %1 : tensor<4x3xbf16>, tensor<3x6xbf16>
  }
  func.func private @expected() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[-7.34310913, -2.609375, -0.0540771484, 3.42163086, 8.19616699, -9.01200866], [-6.38049316, 13.4958191, 18.3164368, 9.31999206, 16.9393616, -3.30096436], [-6.92715454, 1.03515625, -6.36560059, -0.211181641, 4.81018066, -15.8230743], [14.8084717, 21.4648438, 10.7532349, -4.21691895, -10.4781494, 16.2324219]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
}

