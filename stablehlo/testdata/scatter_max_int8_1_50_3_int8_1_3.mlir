// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x50x3xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<32> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x50x3xi8>, tensor<1x3xi8>)
    %1 = call @expected() : () -> tensor<1x50x3xi8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<1x50x3xi8>, tensor<1xi64>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> ()
    return %2 : tensor<1x50x3xi8>
  }
  func.func private @inputs() -> (tensor<1x50x3xi8> {mhlo.layout_mode = "default"}, tensor<1x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFDFD00FF00F802FCFD0002040302010007FE050202FEFE00060100FEFBFA0001FB00FF000000FF020105FDFCFFFEFF020002040100050000000000FC08FE02FE020403020301FEF60001FCFD0006F901FF0100FB010100FAFC0002FD000001010500FD00FC0001FCFE02010700FE00000101000003FBFCFA0103FF00FD040307020300FF000002FFFE0002FF0004FF0203FF01FF0001"> : tensor<1x50x3xi8>
    %c_0 = stablehlo.constant dense<[[-1, 4, 2]]> : tensor<1x3xi8>
    return %c, %c_0 : tensor<1x50x3xi8>, tensor<1x3xi8>
  }
  func.func private @expected() -> (tensor<1x50x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFDFD00FF00F802FCFD0002040302010007FE050202FEFE00060100FEFBFA0001FB00FF000000FF020105FDFCFFFEFF020002040100050000000000FC08FE02FE020403020301FEF60001FCFD0006F901FF0100FB010100FAFC0002FD0000010105040200FC0001FCFE02010700FE00000101000003FBFCFA0103FF00FD040307020300FF000002FFFE0002FF0004FF0203FF01FF0001"> : tensor<1x50x3xi8>
    return %c : tensor<1x50x3xi8>
  }
}
