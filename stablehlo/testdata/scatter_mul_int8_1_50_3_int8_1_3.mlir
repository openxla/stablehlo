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
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<1x50x3xi8>, tensor<1xi64>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> ()
    return %2 : tensor<1x50x3xi8>
  }
  func.func private @inputs() -> (tensor<1x50x3xi8> {mhlo.layout_mode = "default"}, tensor<1x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x01FFFE0101FFFE020100060203FE0003020300FEFE00FF02090406FF00FF0005000000020500FD000100FE0001FF0000010605FEFC01040402FE00FEFAFE01FEFC0007FCFC0001FF05010000FDFE01FDFB00FB00010201FC0001020200FFFDFFFE030501010001FFFCFBFB0402030106020000FF00FE01FAFF010003FCFC01FD07010204FF0102FE0203FE00010200FE010401030202"> : tensor<1x50x3xi8>
    %c_0 = stablehlo.constant dense<[[6, 2, -2]]> : tensor<1x3xi8>
    return %c, %c_0 : tensor<1x50x3xi8>, tensor<1x3xi8>
  }
  func.func private @expected() -> (tensor<1x50x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x01FFFE0101FFFE020100060203FE0003020300FEFE00FF02090406FF00FF0005000000020500FD000100FE0001FF0000010605FEFC01040402FE00FEFAFE01FEFC0007FCFC0001FF05010000FDFE01FDFB00FB00010201FC0001020200FFFDFFF406F601010001FFFCFBFB0402030106020000FF00FE01FAFF010003FCFC01FD07010204FF0102FE0203FE00010200FE010401030202"> : tensor<1x50x3xi8>
    return %c : tensor<1x50x3xi8>
  }
}
