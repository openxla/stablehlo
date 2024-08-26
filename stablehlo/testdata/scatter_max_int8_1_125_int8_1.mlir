// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x125xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x125xi8>, tensor<1xi8>)
    %1 = call @expected() : () -> tensor<1x125xi8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<1x125xi8>, tensor<1xi64>, tensor<1xi8>) -> tensor<1x125xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xi8>, tensor<1x125xi8>) -> ()
    return %2 : tensor<1x125xi8>
  }
  func.func private @inputs() -> (tensor<1x125xi8> {mhlo.layout_mode = "default"}, tensor<1xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0101FEFBFD02FE03FE04FFFEFFFFFE01FF04FF030001FD00FE0303030002FF04FF00FE03FC020006FF010001FDFF0005000301FAFE03FDFB0202FEFE0002FE02000005FFFD010200FEFCF8030200FCFD03020300FF03FA030300FF01010403040105FDFEFE04FEFE030200020505FFFFFFFFFE00000301000100FB01FD"> : tensor<1x125xi8>
    %c_0 = stablehlo.constant dense<-2> : tensor<1xi8>
    return %c, %c_0 : tensor<1x125xi8>, tensor<1xi8>
  }
  func.func private @expected() -> (tensor<1x125xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0101FEFBFD02FE03FE04FFFEFFFFFE01FF04FF030001FD00FE0303030002FF04FF00FE03FC020006FF010001FDFF0005000301FAFE03FDFB0202FEFE0002FE02000005FFFD010200FEFCF8030200FCFD03020300FF03FA030300FF01010403040105FDFEFE04FEFE030200020505FFFFFFFFFE00000301000100FB01FD"> : tensor<1x125xi8>
    return %c : tensor<1x125xi8>
  }
}
