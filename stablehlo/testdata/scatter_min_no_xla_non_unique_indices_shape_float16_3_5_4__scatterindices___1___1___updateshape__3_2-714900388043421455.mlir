// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<3x5x4xf16>, tensor<3x2x4xf16>)
    %2 = call @expected() : () -> tensor<3x5x4xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>} : (tensor<3x5x4xf16>, tensor<2x1xi32>, tensor<3x2x4xf16>) -> tensor<3x5x4xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<3x5x4xf16>, tensor<3x5x4xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x4xf16>, tensor<3x2x4xf16>) {
    %0 = stablehlo.constant dense<[[[4.164060e+00, -1.566410e+00, 1.443360e+00, 1.574220e+00], [2.730470e+00, -5.492190e+00, -3.286130e-01, -6.149290e-02], [-2.822270e+00, 1.877930e+00, -5.472650e+00, 1.980470e+00], [1.676760e+00, 2.412110e+00, -1.255860e+00, 1.949460e-01], [1.461910e+00, -2.162110e+00, 5.063480e-01, -3.062500e+00]], [[-3.328130e+00, 2.994140e+00, 4.695310e+00, -2.193360e+00], [-1.553710e+00, -6.933590e-01, 1.465820e+00, -5.249020e-01], [3.900390e+00, 2.816410e+00, 1.065830e-02, 5.828130e+00], [-3.848270e-02, 2.767580e+00, 2.146480e+00, -1.791990e+00], [3.048830e+00, 2.787110e+00, 1.415040e+00, 2.900390e+00]], [[-1.923830e+00, 1.573490e-01, -3.164060e+00, 5.152340e+00], [-2.310550e+00, -1.673830e+00, 1.250980e+00, 3.275390e+00], [-2.037110e+00, 4.292970e+00, -4.929690e+00, -3.783200e+00], [-3.443360e+00, 1.155270e+00, 2.177730e+00, 3.151860e-01], [-3.082030e+00, 1.514890e-01, -9.316400e-01, 2.310550e+00]]]> : tensor<3x5x4xf16>
    %1 = stablehlo.constant dense<[[[-8.242180e-01, 5.122070e-01, 6.707030e+00, 1.148440e+00], [-3.515630e+00, 1.549800e+00, -3.583980e-01, 4.843750e-01]], [[-6.308590e-01, -1.687010e-01, -2.125000e+00, -3.271480e-01], [2.863280e+00, -3.130860e+00, -7.070310e-01, -1.405270e+00]], [[-2.459720e-01, -1.547850e+00, 7.250970e-01, -8.002920e-01], [-2.174070e-01, 3.394530e+00, 1.902340e+00, 1.539060e+00]]]> : tensor<3x2x4xf16>
    return %0, %1 : tensor<3x5x4xf16>, tensor<3x2x4xf16>
  }
  func.func private @expected() -> tensor<3x5x4xf16> {
    %0 = stablehlo.constant dense<[[[4.164060e+00, -1.566410e+00, 1.443360e+00, 1.574220e+00], [-3.515630e+00, -5.492190e+00, -3.583980e-01, -6.149290e-02], [-2.822270e+00, 1.877930e+00, -5.472650e+00, 1.980470e+00], [1.676760e+00, 2.412110e+00, -1.255860e+00, 1.949460e-01], [1.461910e+00, -2.162110e+00, 5.063480e-01, -3.062500e+00]], [[-3.328130e+00, 2.994140e+00, 4.695310e+00, -2.193360e+00], [-1.553710e+00, -3.130860e+00, -2.125000e+00, -1.405270e+00], [3.900390e+00, 2.816410e+00, 1.065830e-02, 5.828130e+00], [-3.848270e-02, 2.767580e+00, 2.146480e+00, -1.791990e+00], [3.048830e+00, 2.787110e+00, 1.415040e+00, 2.900390e+00]], [[-1.923830e+00, 1.573490e-01, -3.164060e+00, 5.152340e+00], [-2.310550e+00, -1.673830e+00, 7.250970e-01, -8.002920e-01], [-2.037110e+00, 4.292970e+00, -4.929690e+00, -3.783200e+00], [-3.443360e+00, 1.155270e+00, 2.177730e+00, 3.151860e-01], [-3.082030e+00, 1.514890e-01, -9.316400e-01, 2.310550e+00]]]> : tensor<3x5x4xf16>
    return %0 : tensor<3x5x4xf16>
  }
}

