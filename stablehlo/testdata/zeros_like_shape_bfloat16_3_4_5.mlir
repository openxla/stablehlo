// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4x5xbf16>
    %1 = call @expected() : () -> tensor<3x4x5xbf16>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<bf16>) -> tensor<3x4x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x4x5xbf16>, tensor<3x4x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4x5xbf16> {
    %0 = stablehlo.constant dense<[[[-1.269530e-01, 7.773430e-01, -3.476560e-01, 1.191410e-01, 3.453130e+00], [4.062500e+00, 1.367190e+00, 2.453130e+00, -1.304690e+00, -7.617180e-01], [1.429690e+00, -1.132810e+00, -1.242190e+00, 3.164060e-01, -4.257810e-01], [-7.695310e-01, 2.546880e+00, -3.789060e-01, -2.000000e+00, -9.804680e-01]], [[9.648430e-01, 2.953130e+00, 1.351560e+00, -2.625000e+00, -3.078130e+00], [1.492190e+00, -2.671880e+00, -9.218750e-01, 4.937500e+00, 3.171880e+00], [2.796880e+00, 4.875000e+00, 3.312500e+00, -2.796880e+00, -1.710940e+00], [-5.906250e+00, -6.312500e+00, 2.640630e+00, 3.500000e+00, 2.937500e+00]], [[-4.218750e+00, -2.875000e+00, 8.437500e-01, 8.312500e+00, -2.015630e+00], [6.718750e+00, 3.656250e+00, 3.468750e+00, -2.312500e+00, 5.437500e+00], [2.250000e+00, -3.281250e+00, -2.312500e+00, -1.078130e+00, 2.468750e+00], [1.593750e+00, -5.968750e+00, -4.648440e-01, 4.875000e+00, 1.656250e+00]]]> : tensor<3x4x5xbf16>
    return %0 : tensor<3x4x5xbf16>
  }
  func.func private @expected() -> tensor<3x4x5xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<3x4x5xbf16>
    return %0 : tensor<3x4x5xbf16>
  }
}
