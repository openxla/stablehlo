// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:4 = call @inputs() : () -> (tensor<2x3xi32>, tensor<2x3xf16>, tensor<2x3xf16>, tensor<2x3xf16>)
    %1 = call @expected() : () -> tensor<2x3xf16>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %3 = stablehlo.compare  LT, %0#0, %2,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %5 = stablehlo.compare  LT, %0#0, %4,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %6 = stablehlo.select %5, %0#2, %0#3 : tensor<2x3xi1>, tensor<2x3xf16>
    %7 = stablehlo.select %3, %0#1, %6 : tensor<2x3xi1>, tensor<2x3xf16>
    stablehlo.custom_call @check.expect_close(%7, %1) {has_side_effect = true} : (tensor<2x3xf16>, tensor<2x3xf16>) -> ()
    return %7 : tensor<2x3xf16>
  }
  func.func private @inputs() -> (tensor<2x3xi32> {mhlo.layout_mode = "default"}, tensor<2x3xf16> {mhlo.layout_mode = "default"}, tensor<2x3xf16> {mhlo.layout_mode = "default"}, tensor<2x3xf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 0, 2], [0, 2, 0]]> : tensor<2x3xi32>
    %cst = stablehlo.constant dense<[[1.390630e+00, 2.932130e-01, 5.960940e+00], [-2.246090e+00, -1.575200e+00, -1.703130e+00]]> : tensor<2x3xf16>
    %cst_0 = stablehlo.constant dense<[[5.703130e-01, 2.453130e+00, -1.626950e+00], [-1.898440e+00, -4.128910e+00, 4.449220e+00]]> : tensor<2x3xf16>
    %cst_1 = stablehlo.constant dense<[[-5.304690e+00, -1.771480e+00, 1.868160e+00], [-5.253910e+00, -1.725590e+00, -2.605470e+00]]> : tensor<2x3xf16>
    return %c, %cst, %cst_0, %cst_1 : tensor<2x3xi32>, tensor<2x3xf16>, tensor<2x3xf16>, tensor<2x3xf16>
  }
  func.func private @expected() -> (tensor<2x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-5.304690e+00, 2.932130e-01, 1.868160e+00], [-2.246090e+00, -1.725590e+00, -1.703130e+00]]> : tensor<2x3xf16>
    return %cst : tensor<2x3xf16>
  }
}
