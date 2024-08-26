// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3x4xf32>
    %1 = call @expected() : () -> tensor<3x4x2xf32>
    %2 = stablehlo.transpose %0, dims = [1, 2, 0] : (tensor<2x3x4xf32>) -> tensor<3x4x2xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x4x2xf32>, tensor<3x4x2xf32>) -> ()
    return %2 : tensor<3x4x2xf32>
  }
  func.func private @inputs() -> (tensor<2x3x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-3.80841613, -3.58268166, -3.428400e-01, 1.12658536], [-1.35501349, 5.86460495, -2.9754107, -2.14586616], [2.40116119, 1.97292423, 0.00829547829, -2.05573058]], [[0.500765204, -1.24809313, -0.254042923, 0.0727051944], [-3.1472609, 6.320640e-01, 1.86220384, 2.29526496], [0.630201399, 3.82720256, 0.950467467, 0.584882617]]]> : tensor<2x3x4xf32>
    return %cst : tensor<2x3x4xf32>
  }
  func.func private @expected() -> (tensor<3x4x2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-3.80841613, 0.500765204], [-3.58268166, -1.24809313], [-3.428400e-01, -0.254042923], [1.12658536, 0.0727051944]], [[-1.35501349, -3.1472609], [5.86460495, 6.320640e-01], [-2.9754107, 1.86220384], [-2.14586616, 2.29526496]], [[2.40116119, 0.630201399], [1.97292423, 3.82720256], [0.00829547829, 0.950467467], [-2.05573058, 0.584882617]]]> : tensor<3x4x2xf32>
    return %cst : tensor<3x4x2xf32>
  }
}
