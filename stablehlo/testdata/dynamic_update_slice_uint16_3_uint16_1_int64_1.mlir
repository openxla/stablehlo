// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<3xui16>, tensor<1xui16>, tensor<1xi64>)
    %1 = call @expected() : () -> tensor<3xui16>
    %2 = stablehlo.slice %0#2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    %3 = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %4 = stablehlo.compare  LT, %3, %c,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %c_0 = stablehlo.constant dense<3> : tensor<i64>
    %5 = stablehlo.add %3, %c_0 : tensor<i64>
    %6 = stablehlo.select %4, %5, %3 : tensor<i1>, tensor<i64>
    %7 = stablehlo.dynamic_update_slice %0#0, %0#1, %6 : (tensor<3xui16>, tensor<1xui16>, tensor<i64>) -> tensor<3xui16>
    stablehlo.custom_call @check.expect_eq(%7, %1) {has_side_effect = true} : (tensor<3xui16>, tensor<3xui16>) -> ()
    return %7 : tensor<3xui16>
  }
  func.func private @inputs() -> (tensor<3xui16> {mhlo.layout_mode = "default"}, tensor<1xui16> {mhlo.layout_mode = "default"}, tensor<1xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[2, 2, 0]> : tensor<3xui16>
    %c_0 = stablehlo.constant dense<0> : tensor<1xui16>
    %c_1 = stablehlo.constant dense<1> : tensor<1xi64>
    return %c, %c_0, %c_1 : tensor<3xui16>, tensor<1xui16>, tensor<1xi64>
  }
  func.func private @expected() -> (tensor<3xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[2, 0, 0]> : tensor<3xui16>
    return %c : tensor<3xui16>
  }
}
