// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xf16>, tensor<f16>)
    %1 = call @expected() : () -> tensor<6x4xf16>
    %2 = stablehlo.pad %0#0, %0#1, low = [1, 0], high = [2, 1], interior = [1, 0] : (tensor<2x3xf16>, tensor<f16>) -> tensor<6x4xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<6x4xf16>, tensor<6x4xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf16>, tensor<f16>) {
    %0 = stablehlo.constant dense<[[-1.515390e-03, -1.670840e-03, 2.983090e-03], [6.544590e-05, 6.623260e-04, -1.194000e-03]]> : tensor<2x3xf16>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    return %0, %1 : tensor<2x3xf16>, tensor<f16>
  }
  func.func private @expected() -> tensor<6x4xf16> {
    %0 = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-1.515390e-03, -1.670840e-03, 2.983090e-03, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [6.544590e-05, 6.623260e-04, -1.194000e-03, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<6x4xf16>
    return %0 : tensor<6x4xf16>
  }
}
