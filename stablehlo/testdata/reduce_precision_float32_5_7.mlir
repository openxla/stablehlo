// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xf32>
    %1 = call @expected() : () -> tensor<5x7xf32>
    %2 = stablehlo.reduce_precision %0, format = e11m52 : tensor<5x7xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<5x7xf32>, tensor<5x7xf32>) -> ()
    return %2 : tensor<5x7xf32>
  }
  func.func private @inputs() -> (tensor<5x7xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.0463174656, -0.359622896, 2.68061018, 2.73067665, -2.98465776, 2.37090635, 3.00272846], [-1.70035434, 0.561822534, 2.39265943, -3.71191573, -2.10235906, -3.70896864, 2.49592757], [2.0122726, 0.415704519, -0.829515696, 2.32972074, 1.1138587, 2.00742745, 2.4653914], [5.22061348, 2.39084625, 0.787524521, 0.837473809, 2.02389908, 0.432699025, 1.31516218], [-1.65173602, 2.73440528, 0.9528777, -0.0713662207, 2.9412961, -1.7066927, -3.27265668]]> : tensor<5x7xf32>
    return %cst : tensor<5x7xf32>
  }
  func.func private @expected() -> (tensor<5x7xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.0463174656, -0.359622896, 2.68061018, 2.73067665, -2.98465776, 2.37090635, 3.00272846], [-1.70035434, 0.561822534, 2.39265943, -3.71191573, -2.10235906, -3.70896864, 2.49592757], [2.0122726, 0.415704519, -0.829515696, 2.32972074, 1.1138587, 2.00742745, 2.4653914], [5.22061348, 2.39084625, 0.787524521, 0.837473809, 2.02389908, 0.432699025, 1.31516218], [-1.65173602, 2.73440528, 0.9528777, -0.0713662207, 2.9412961, -1.7066927, -3.27265668]]> : tensor<5x7xf32>
    return %cst : tensor<5x7xf32>
  }
}
