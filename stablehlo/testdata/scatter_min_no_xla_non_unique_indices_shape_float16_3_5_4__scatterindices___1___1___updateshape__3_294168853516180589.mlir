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
    %0 = stablehlo.constant dense<[[[-7.556150e-02, 1.252930e+00, -2.248050e+00, 2.369380e-01], [-7.664060e+00, -4.054690e+00, 9.184570e-01, -2.167970e+00], [-3.156250e+00, -1.679690e+00, 4.679690e+00, 1.528320e-01], [6.546880e+00, 3.984380e+00, -3.468750e+00, 1.938480e+00], [3.978520e+00, -1.143550e+00, -6.054690e+00, -3.617190e+00]], [[-6.928710e-01, -1.076170e+00, -2.687500e+00, -1.204100e+00], [-4.019530e+00, -1.184570e+00, 4.619140e-01, 7.382810e+00], [2.568360e+00, 5.130000e-02, 1.119140e+00, -7.667960e+00], [1.395510e+00, 3.099610e+00, -5.714840e+00, 6.557620e-01], [2.869140e+00, -3.876950e+00, 4.085940e+00, -8.032220e-01]], [[5.421880e+00, 1.021480e+00, -1.686520e+00, -3.830080e+00], [6.789060e+00, -4.431150e-01, 1.865230e+00, 4.753910e+00], [2.685550e+00, -2.349610e+00, 3.562500e+00, -1.050780e+00], [7.641600e-01, -4.016110e-01, 4.648440e+00, -2.925780e+00], [7.199210e+00, 3.164060e-01, 2.988280e+00, -2.041020e+00]]]> : tensor<3x5x4xf16>
    %1 = stablehlo.constant dense<[[[5.019530e+00, 6.225590e-01, 3.314450e+00, 1.093750e+00], [2.255860e+00, 5.761710e+00, 6.527340e+00, -2.988280e+00]], [[3.937500e+00, 2.642580e+00, -1.405270e+00, 1.593750e+00], [-4.443360e-01, 6.359380e+00, 2.533200e+00, 1.043950e+00]], [[3.271480e-01, 2.005860e+00, 7.104490e-01, 3.830080e+00], [-4.677730e-01, 1.700200e+00, 3.119140e+00, 1.831050e+00]]]> : tensor<3x2x4xf16>
    return %0, %1 : tensor<3x5x4xf16>, tensor<3x2x4xf16>
  }
  func.func private @expected() -> tensor<3x5x4xf16> {
    %0 = stablehlo.constant dense<[[[-7.556150e-02, 1.252930e+00, -2.248050e+00, 2.369380e-01], [-7.664060e+00, -4.054690e+00, 9.184570e-01, -2.988280e+00], [-3.156250e+00, -1.679690e+00, 4.679690e+00, 1.528320e-01], [6.546880e+00, 3.984380e+00, -3.468750e+00, 1.938480e+00], [3.978520e+00, -1.143550e+00, -6.054690e+00, -3.617190e+00]], [[-6.928710e-01, -1.076170e+00, -2.687500e+00, -1.204100e+00], [-4.019530e+00, -1.184570e+00, -1.405270e+00, 1.043950e+00], [2.568360e+00, 5.130000e-02, 1.119140e+00, -7.667960e+00], [1.395510e+00, 3.099610e+00, -5.714840e+00, 6.557620e-01], [2.869140e+00, -3.876950e+00, 4.085940e+00, -8.032220e-01]], [[5.421880e+00, 1.021480e+00, -1.686520e+00, -3.830080e+00], [-4.677730e-01, -4.431150e-01, 7.104490e-01, 1.831050e+00], [2.685550e+00, -2.349610e+00, 3.562500e+00, -1.050780e+00], [7.641600e-01, -4.016110e-01, 4.648440e+00, -2.925780e+00], [7.199210e+00, 3.164060e-01, 2.988280e+00, -2.041020e+00]]]> : tensor<3x5x4xf16>
    return %0 : tensor<3x5x4xf16>
  }
}

