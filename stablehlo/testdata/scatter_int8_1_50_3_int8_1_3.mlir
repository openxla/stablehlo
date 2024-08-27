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
      stablehlo.return %arg1 : tensor<i8>
    }) : (tensor<1x50x3xi8>, tensor<1xi64>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> ()
    return %2 : tensor<1x50x3xi8>
  }
  func.func private @inputs() -> (tensor<1x50x3xi8> {mhlo.layout_mode = "default"}, tensor<1x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFF0403020000FFFE03FD0300FE010300030707000101F603FA0000FE040106FE00FE0100030002FC04FDFD00060305020002FC0402000001FE01010304FD02FEFD01FEFB01FF02FF0100FF05FFFEFB00FF07020202FFFBFF05030000FF05010200000001FF000006FEFEFFFC03FFFEFD03030100FF07FF00FBFE0002FEFEFC0003FEFFFE0100040102FFFC00FEFF00FEFF03FEFD02FD"> : tensor<1x50x3xi8>
    %c_0 = stablehlo.constant dense<[[0, 1, 1]]> : tensor<1x3xi8>
    return %c, %c_0 : tensor<1x50x3xi8>, tensor<1x3xi8>
  }
  func.func private @expected() -> (tensor<1x50x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFF0403020000FFFE03FD0300FE010300030707000101F603FA0000FE040106FE00FE0100030002FC04FDFD00060305020002FC0402000001FE01010304FD02FEFD01FEFB01FF02FF0100FF05FFFEFB00FF07020202FFFBFF05030000FF05010200010101FF000006FEFEFFFC03FFFEFD03030100FF07FF00FBFE0002FEFEFC0003FEFFFE0100040102FFFC00FEFF00FEFF03FEFD02FD"> : tensor<1x50x3xi8>
    return %c : tensor<1x50x3xi8>
  }
}
