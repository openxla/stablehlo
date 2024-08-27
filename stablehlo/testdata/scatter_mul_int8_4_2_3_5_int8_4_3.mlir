// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>)
    %1 = call @expected() : () -> tensor<4x2x3x5xi8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<4x2x3x5xi8>, tensor<2xi64>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> ()
    return %2 : tensor<4x2x3x5xi8>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi8> {mhlo.layout_mode = "default"}, tensor<4x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x01F8000204010001FAFFFFFE00FF0101FF02FD00040006000000FDFF0002FD01020000000100FDFD03FF0201FFFB06FCFF0601FDFC00FD03010000FB0000FDFCFF040105FEFEFB0102FF01FF02FC00FE0003FF00FFFA0005FD00FE00FE05FCFC00000003FDFC0100FD00030202010005FD0000010200FF00"> : tensor<4x2x3x5xi8>
    %c_0 = stablehlo.constant dense<[[0, 1, 2], [-6, 0, 3], [0, -1, 0], [-3, 2, 1]]> : tensor<4x3xi8>
    return %c, %c_0 : tensor<4x2x3x5xi8>, tensor<4x3xi8>
  }
  func.func private @expected() -> (tensor<4x2x3x5xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x01F8000200010001FAFFFFFE00FF0201FF02FD00040006000000FDFF0002FD01020000000100FD0003FF0201FDFB06FCFF0601FDFC00FD03010000FB0000FDFC00040105FE02FB0102FF00FF02FC00FE0003FF00FFFA0005FD00FE00FE050CFC00000006FDFC0100FD00030202010005FD0000010200FF00"> : tensor<4x2x3x5xi8>
    return %c : tensor<4x2x3x5xi8>
  }
}
