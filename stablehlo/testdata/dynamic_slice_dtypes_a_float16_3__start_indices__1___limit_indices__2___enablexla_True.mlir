// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xf16>, tensor<1xi32>)
    %1 = call @expected() : () -> tensor<1xf16>
    %2 = "stablehlo.slice"(%0#1) {limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>} : (tensor<1xi32>) -> tensor<1xi32>
    %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.compare  LT, %3, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = stablehlo.constant dense<3> : tensor<i32>
    %7 = stablehlo.add %3, %6 : tensor<i32>
    %8 = stablehlo.select %5, %7, %3 : tensor<i1>, tensor<i32>
    %9 = stablehlo.dynamic_slice %0#0, %8, sizes = [1] : (tensor<3xf16>, tensor<i32>) -> tensor<1xf16>
    %10 = stablehlo.custom_call @check.eq(%9, %1) : (tensor<1xf16>, tensor<1xf16>) -> tensor<i1>
    return %10 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xf16>, tensor<1xi32>) {
    %0 = stablehlo.constant dense<[-8.422850e-01, 3.076170e+00, 7.998050e-01]> : tensor<3xf16>
    %1 = stablehlo.constant dense<1> : tensor<1xi32>
    return %0, %1 : tensor<3xf16>, tensor<1xi32>
  }
  func.func private @expected() -> tensor<1xf16> {
    %0 = stablehlo.constant dense<3.076170e+00> : tensor<1xf16>
    return %0 : tensor<1xf16>
  }
}
