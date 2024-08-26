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
      stablehlo.return %arg1 : tensor<i8>
    }) : (tensor<4x2x3x5xi8>, tensor<2xi64>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> ()
    return %2 : tensor<4x2x3x5xi8>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi8> {mhlo.layout_mode = "default"}, tensor<4x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x02FDFF01000004FD05000207FC04FF02FF0102FA0003000000FF0402FF03FD0504000400FD0300FE04FF000103FF02FFFAF8FF000603030200F90000F7000101FEFF00000004000102FCFFFC02FDFE00000101FBFD010103FFFEFF000404F9000200000400FD0303010100FF02020600FD00FF00F800FCFE"> : tensor<4x2x3x5xi8>
    %c_0 = stablehlo.constant dense<[[-6, 1, 0], [1, 0, 3], [0, 2, 1], [4, 4, 3]]> : tensor<4x3xi8>
    return %c, %c_0 : tensor<4x2x3x5xi8>, tensor<4x3xi8>
  }
  func.func private @expected() -> (tensor<4x2x3x5xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x02FDFF01FA0004FD05010207FC040002FF0102FA0003000000FF0402FF03FD0504000100FD03000004FF000103FF02FFFAF8FF000603030200F90000F700010100FF00000002000102FC01FC02FDFE00000101FBFD010103FFFEFF00040404000200000400FD0303030100FF02020600FD00FF00F800FCFE"> : tensor<4x2x3x5xi8>
    return %c : tensor<4x2x3x5xi8>
  }
}
