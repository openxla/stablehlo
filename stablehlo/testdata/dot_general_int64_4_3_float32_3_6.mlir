// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi64>, tensor<3x6xf32>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi64>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf32>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    return %4 : tensor<4x6xf32>
  }
  func.func private @inputs() -> (tensor<4x3xi64> {mhlo.layout_mode = "default"}, tensor<3x6xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-5, 4, -3], [0, 0, 7], [0, 4, 6], [0, 0, 6]]> : tensor<4x3xi64>
    %cst = stablehlo.constant dense<[[-4.41832066, 2.30382681, 1.44504547, 2.03596401, -0.543255627, 5.14671564], [1.12803113, 2.10108399, 0.470255971, -1.63337147, -3.10091877, -0.58413136], [1.82461488, 4.01580429, 3.03422093, 0.731350958, -0.274075121, 1.52424252]]> : tensor<3x6xf32>
    return %c, %cst : tensor<4x3xi64>, tensor<3x6xf32>
  }
  func.func private @expected() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[21.1298828, -15.1622114, -14.446866, -18.9073601, -8.86517143, -32.6428299], [12.7723045, 28.11063, 21.2395458, 5.11945677, -1.91852582, 10.6696978], [15.4598141, 32.4991608, 20.0863495, -2.145380e+00, -14.0481262, 6.80892944], [10.9476891, 24.0948257, 18.2053261, 4.38810587, -1.64445066, 9.14545536]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
}
