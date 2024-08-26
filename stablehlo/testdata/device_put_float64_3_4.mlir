// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3x4xf64>
    %1 = call @expected() : () -> tensor<3x4xf64>
    stablehlo.custom_call @check.expect_close(%0, %1) {has_side_effect = true} : (tensor<3x4xf64>, tensor<3x4xf64>) -> ()
    return %0 : tensor<3x4xf64>
  }
  func.func private @inputs() -> (tensor<3x4xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.93871766744870521, 2.8357154514833556, -1.9676780707688346, 3.0538035765256391], [1.2355984062438199, -4.6736242515759301, 2.8795909205256449, 2.1062487351838799], [1.4274756889977089, 2.6006261663243797, 6.7800422809940102, 4.5164546160017318]]> : tensor<3x4xf64>
    return %cst : tensor<3x4xf64>
  }
  func.func private @expected() -> (tensor<3x4xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.93871766744870521, 2.8357154514833556, -1.9676780707688346, 3.0538035765256391], [1.2355984062438199, -4.6736242515759301, 2.8795909205256449, 2.1062487351838799], [1.4274756889977089, 2.6006261663243797, 6.7800422809940102, 4.5164546160017318]]> : tensor<3x4xf64>
    return %cst : tensor<3x4xf64>
  }
}
