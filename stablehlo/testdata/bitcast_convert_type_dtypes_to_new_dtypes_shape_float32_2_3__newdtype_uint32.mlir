// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xf32>
    %1 = call @expected() : () -> tensor<2x3xui32>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xf32>) -> tensor<2x3xui32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xui32>, tensor<2x3xui32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[5.429460e+00, -5.89350414, 1.64081824], [-1.19091403, 5.98908186, -4.70763731]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  func.func private @expected() -> tensor<2x3xui32> {
    %0 = stablehlo.constant dense<[[1085128227, 3233585046, 1070728789], [3214438367, 1086301839, 3231098103]]> : tensor<2x3xui32>
    return %0 : tensor<2x3xui32>
  }
}
