// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:4 = call @inputs() : () -> (tensor<2x3xi32>, tensor<2x3xf64>, tensor<2x3xf64>, tensor<2x3xf64>)
    %1 = call @expected() : () -> tensor<2x3xf64>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %3 = stablehlo.compare  LT, %0#0, %2,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %5 = stablehlo.compare  LT, %0#0, %4,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %6 = stablehlo.select %5, %0#2, %0#3 : tensor<2x3xi1>, tensor<2x3xf64>
    %7 = stablehlo.select %3, %0#1, %6 : tensor<2x3xi1>, tensor<2x3xf64>
    stablehlo.custom_call @check.expect_close(%7, %1) {has_side_effect = true} : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    return %7 : tensor<2x3xf64>
  }
  func.func private @inputs() -> (tensor<2x3xi32> {mhlo.layout_mode = "default"}, tensor<2x3xf64> {mhlo.layout_mode = "default"}, tensor<2x3xf64> {mhlo.layout_mode = "default"}, tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 1, 2], [1, 0, 0]]> : tensor<2x3xi32>
    %cst = stablehlo.constant dense<[[-1.052776689176447, 4.8069485917233958, -3.7651505536904235], [-2.876392999881408, -1.7670196129089768, -3.6395192156358052]]> : tensor<2x3xf64>
    %cst_0 = stablehlo.constant dense<[[2.2510729191479317, 3.1269453789201296, -2.8835289165705293], [-3.3451178542512645, -2.4054327548529431, -4.0062719301466148]]> : tensor<2x3xf64>
    %cst_1 = stablehlo.constant dense<[[-0.032373297594009726, -0.90084348243514655, 0.75337834447118979], [0.48588967938531824, -4.7814668477506181, 3.2187831352023175]]> : tensor<2x3xf64>
    return %c, %cst, %cst_0, %cst_1 : tensor<2x3xi32>, tensor<2x3xf64>, tensor<2x3xf64>, tensor<2x3xf64>
  }
  func.func private @expected() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-0.032373297594009726, 3.1269453789201296, 0.75337834447118979], [-3.3451178542512645, -1.7670196129089768, -3.6395192156358052]]> : tensor<2x3xf64>
    return %cst : tensor<2x3xf64>
  }
}
