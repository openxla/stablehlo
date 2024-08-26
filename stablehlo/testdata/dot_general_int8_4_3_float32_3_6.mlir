// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi8>, tensor<3x6xf32>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi8>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf32>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    return %4 : tensor<4x6xf32>
  }
  func.func private @inputs() -> (tensor<4x3xi8> {mhlo.layout_mode = "default"}, tensor<3x6xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-3, -2, -1], [0, -3, 1], [-6, 0, 0], [2, -6, -4]]> : tensor<4x3xi8>
    %cst = stablehlo.constant dense<[[-3.99020648, 2.58919454, 0.433581144, 3.68860888, 2.73794389, -2.00574136], [-2.59321427, -1.43049741, -3.01954341, 0.79414606, 0.508764327, -0.239404649], [-0.140727609, -4.93321419, -0.293230742, -4.03748178, -1.0149287, 1.85012126]]> : tensor<3x6xf32>
    return %c, %cst : tensor<4x3xi8>, tensor<3x6xf32>
  }
  func.func private @expected() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[17.2977753, 0.0266251564, 5.03157377, -8.61663627, -8.21643161, 4.64591217], [7.63891554, -0.641721725, 8.76539897, -6.419920e+00, -2.54122162, 2.56833506], [23.9412384, -15.5351677, -2.60148692, -22.1316528, -16.4276638, 12.0344486], [8.14178276, 33.4942322, 20.1573448, 18.7622681, 6.48301649, -9.975540e+00]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
}
