// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:4 = call @inputs() : () -> (tensor<2x3xi32>, tensor<2x3xbf16>, tensor<2x3xbf16>, tensor<2x3xbf16>)
    %1 = call @expected() : () -> tensor<2x3xbf16>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %4 = stablehlo.compare  LT, %0#0, %3,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %5 = stablehlo.constant dense<2> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %7 = stablehlo.compare  LT, %0#0, %6,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %8 = stablehlo.select %7, %0#2, %0#3 : tensor<2x3xi1>, tensor<2x3xbf16>
    %9 = stablehlo.select %4, %0#1, %8 : tensor<2x3xi1>, tensor<2x3xbf16>
    %10 = stablehlo.custom_call @check.eq(%9, %1) : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<i1>
    return %10 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xi32>, tensor<2x3xbf16>, tensor<2x3xbf16>, tensor<2x3xbf16>) {
    %0 = stablehlo.constant dense<[[0, 1, 1], [1, 0, 2]]> : tensor<2x3xi32>
    %1 = stablehlo.constant dense<[[1.562500e+00, -1.710940e+00, -1.226560e+00], [-3.574220e-01, -4.062500e-01, 1.546880e+00]]> : tensor<2x3xbf16>
    %2 = stablehlo.constant dense<[[-1.835940e+00, 7.125000e+00, -2.250000e+00], [-3.921880e+00, 2.000000e+00, -3.625000e+00]]> : tensor<2x3xbf16>
    %3 = stablehlo.constant dense<[[-3.046880e+00, -2.062500e+00, -1.015630e+00], [-1.390630e+00, -2.294920e-01, -1.765630e+00]]> : tensor<2x3xbf16>
    return %0, %1, %2, %3 : tensor<2x3xi32>, tensor<2x3xbf16>, tensor<2x3xbf16>, tensor<2x3xbf16>
  }
  func.func private @expected() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[1.562500e+00, 7.125000e+00, -2.250000e+00], [-3.921880e+00, -4.062500e-01, -1.765630e+00]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
}
