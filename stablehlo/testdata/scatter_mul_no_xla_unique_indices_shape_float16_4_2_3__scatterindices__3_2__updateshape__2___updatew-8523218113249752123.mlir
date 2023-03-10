// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[3, 2]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3xf16>, tensor<2xf16>)
    %2 = call @expected() : () -> tensor<4x2x3xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf16>, tensor<2xi32>, tensor<2xf16>) -> tensor<4x2x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16>, tensor<2xf16>) {
    %0 = stablehlo.constant dense<[[[8.710930e+00, 3.892580e+00, -6.591800e-01], [8.671870e-01, -2.250980e-01, -1.601560e-01]], [[-5.558590e+00, 1.345210e-01, -3.341800e+00], [-5.175780e-01, 2.271480e+00, 1.391600e+00]], [[3.591800e+00, -4.074220e+00, 1.443360e+00], [-4.164060e+00, -1.551760e+00, -2.908200e+00]], [[-1.690670e-01, 7.075190e-01, 8.090820e-01], [-3.427730e+00, 6.148440e+00, 7.993160e-01]]]> : tensor<4x2x3xf16>
    %1 = stablehlo.constant dense<[1.501950e+00, 6.293950e-01]> : tensor<2xf16>
    return %0, %1 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> tensor<4x2x3xf16> {
    %0 = stablehlo.constant dense<[[[8.710930e+00, 3.892580e+00, -6.591800e-01], [8.671870e-01, -2.250980e-01, -1.601560e-01]], [[-5.558590e+00, 1.345210e-01, -3.341800e+00], [-5.175780e-01, 2.271480e+00, 1.391600e+00]], [[3.591800e+00, -4.074220e+00, 1.443360e+00], [-4.164060e+00, -1.551760e+00, -2.908200e+00]], [[-1.690670e-01, 7.075190e-01, 1.214840e+00], [-3.427730e+00, 6.148440e+00, 5.029300e-01]]]> : tensor<4x2x3xf16>
    return %0 : tensor<4x2x3xf16>
  }
}

