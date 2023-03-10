// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x5xf16>
    %1 = call @expected() : () -> tensor<4x5xf16>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x5xf16>, tensor<4x5xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x5xf16> {
    %0 = stablehlo.constant dense<[[-3.843750e+00, -1.816410e+00, -3.042970e+00, 3.191410e+00, 9.692380e-01], [-4.926760e-01, -4.389650e-01, -3.982420e+00, 2.021480e+00, -2.193360e+00], [2.759770e+00, -1.366210e+00, -1.299800e+00, 1.706050e+00, 1.254880e+00], [-2.023930e-01, -1.726560e+00, 3.976560e+00, 1.223630e+00, -3.080080e+00]]> : tensor<4x5xf16>
    return %0 : tensor<4x5xf16>
  }
  func.func private @expected() -> tensor<4x5xf16> {
    %0 = stablehlo.constant dense<[[-2.023930e-01, -1.726560e+00, 3.976560e+00, 1.223630e+00, -3.080080e+00], [2.759770e+00, -1.366210e+00, -1.299800e+00, 1.706050e+00, 1.254880e+00], [-4.926760e-01, -4.389650e-01, -3.982420e+00, 2.021480e+00, -2.193360e+00], [-3.843750e+00, -1.816410e+00, -3.042970e+00, 3.191410e+00, 9.692380e-01]]> : tensor<4x5xf16>
    return %0 : tensor<4x5xf16>
  }
}
