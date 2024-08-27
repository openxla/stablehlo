// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3xf16>, tensor<2xf16>)
    %1 = call @expected() : () -> tensor<4x2x3xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<4x2x3xf16>, tensor<2xi64>, tensor<2xf16>) -> tensor<4x2x3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> ()
    return %2 : tensor<4x2x3xf16>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16> {mhlo.layout_mode = "default"}, tensor<2xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-4.195310e+00, -2.726560e+00, -2.300780e+00], [5.562500e+00, 5.890630e+00, 1.609380e+00]], [[-1.485350e+00, -1.921880e+00, -3.976560e+00], [2.164060e+00, -4.113280e+00, 1.083980e+00]], [[-1.351560e+00, 5.355470e+00, 7.287590e-02], [-4.378910e+00, 1.294920e+00, -1.518550e+00]], [[3.146480e+00, 3.517580e+00, -1.776120e-01], [-2.851560e+00, 6.285150e+00, -8.886710e-01]]]> : tensor<4x2x3xf16>
    %cst_0 = stablehlo.constant dense<[-2.132810e+00, -8.239740e-02]> : tensor<2xf16>
    return %cst, %cst_0 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> (tensor<4x2x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-4.195310e+00, -2.726560e+00, -2.300780e+00], [5.562500e+00, 5.890630e+00, 1.609380e+00]], [[-1.485350e+00, -1.921880e+00, -3.976560e+00], [2.164060e+00, -4.113280e+00, 1.083980e+00]], [[-1.351560e+00, 5.355470e+00, 7.287590e-02], [-4.378910e+00, 1.294920e+00, -1.518550e+00]], [[3.146480e+00, 3.517580e+00, -2.132810e+00], [-2.851560e+00, 6.285150e+00, -8.886710e-01]]]> : tensor<4x2x3xf16>
    return %cst : tensor<4x2x3xf16>
  }
}
