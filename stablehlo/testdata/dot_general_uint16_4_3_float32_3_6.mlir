// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui16>, tensor<3x6xf32>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui16>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf32>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    return %4 : tensor<4x6xf32>
  }
  func.func private @inputs() -> (tensor<4x3xui16> {mhlo.layout_mode = "default"}, tensor<3x6xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 1, 6], [3, 1, 0], [3, 3, 2], [2, 4, 2]]> : tensor<4x3xui16>
    %cst = stablehlo.constant dense<[[0.481567651, 1.21564889, -0.83783406, -6.77650261, -2.61862063, -1.68150115], [-1.98105204, -1.82323277, -2.40821362, -2.40491676, -0.84722644, -2.96528101], [-4.80308437, -3.1644187, 0.0673174784, -4.37394953, 5.07295513, -3.3313961]]> : tensor<3x6xf32>
    return %c, %cst : tensor<4x3xui16>, tensor<3x6xf32>
  }
  func.func private @expected() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-29.836422, -18.3784466, -3.67997694, -42.2016182, 24.3532639, -26.3166599], [-0.536349058, 1.8237139, -4.92171574, -22.7344246, -8.7030878, -8.0097847], [-14.1046219, -8.15158939, -9.60350799, -36.2921562, -0.251630783, -20.603138], [-16.5672417, -11.1904707, -11.1738873, -31.9205704, 1.51976299, -21.886919]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
}
