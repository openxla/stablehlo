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
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf16>, tensor<2xi32>, tensor<2xf16>) -> tensor<4x2x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16>, tensor<2xf16>) {
    %0 = stablehlo.constant dense<[[[-2.804690e+00, -1.310550e+00, 2.005860e+00], [-7.148430e-01, 2.271480e+00, 1.171880e+00]], [[4.804690e+00, 3.425780e+00, 9.819330e-01], [2.939450e+00, 4.123540e-01, -1.312500e+00]], [[8.481440e-01, -2.757810e+00, 1.665040e+00], [4.910160e+00, 7.304680e+00, -1.010740e+00]], [[-9.291990e-01, 1.258790e+00, 3.197270e+00], [8.666990e-01, 4.707030e+00, -3.478520e+00]]]> : tensor<4x2x3xf16>
    %1 = stablehlo.constant dense<[1.223630e+00, 1.197270e+00]> : tensor<2xf16>
    return %0, %1 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> tensor<4x2x3xf16> {
    %0 = stablehlo.constant dense<[[[-2.804690e+00, -1.310550e+00, 2.005860e+00], [-7.148430e-01, 2.271480e+00, 1.171880e+00]], [[4.804690e+00, 3.425780e+00, 9.819330e-01], [2.939450e+00, 4.123540e-01, -1.312500e+00]], [[8.481440e-01, -2.757810e+00, 1.665040e+00], [4.910160e+00, 7.304680e+00, -1.010740e+00]], [[-9.291990e-01, 1.258790e+00, 4.421880e+00], [8.666990e-01, 4.707030e+00, -2.281250e+00]]]> : tensor<4x2x3xf16>
    return %0 : tensor<4x2x3xf16>
  }
}

