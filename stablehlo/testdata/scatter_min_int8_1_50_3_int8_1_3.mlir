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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<1x50x3xi8>, tensor<1xi64>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> ()
    return %2 : tensor<1x50x3xi8>
  }
  func.func private @inputs() -> (tensor<1x50x3xi8> {mhlo.layout_mode = "default"}, tensor<1x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFAFFFE05FBFD0002FD04000200FB00F9FEFEFFFFFBFE000100FD000001FF00FF0001000200000203FCFD00FFFFFF05FD00020100000002FC0004FEFD00FFFD0001030000FE000301FAFEFBFD01FF03FFFF00FD000000FBFDFC0003FFFDFBFE000000FF0005FDFE01FD00FD01FDFF00FD0001FF0000FEFF0001FCFEFEFF03050000000003FF0000FE0000FC03FE010001FCFFFA000100"> : tensor<1x50x3xi8>
    %c_0 = stablehlo.constant dense<[[-3, 1, 3]]> : tensor<1x3xi8>
    return %c, %c_0 : tensor<1x50x3xi8>, tensor<1x3xi8>
  }
  func.func private @expected() -> (tensor<1x50x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFAFFFE05FBFD0002FD04000200FB00F9FEFEFFFFFBFE000100FD000001FF00FF0001000200000203FCFD00FFFFFF05FD00020100000002FC0004FEFD00FFFD0001030000FE000301FAFEFBFD01FF03FFFF00FD000000FBFDFC0003FFFDFBFE00FD00FF0005FDFE01FD00FD01FDFF00FD0001FF0000FEFF0001FCFEFEFF03050000000003FF0000FE0000FC03FE010001FCFFFA000100"> : tensor<1x50x3xi8>
    return %c : tensor<1x50x3xi8>
  }
}
