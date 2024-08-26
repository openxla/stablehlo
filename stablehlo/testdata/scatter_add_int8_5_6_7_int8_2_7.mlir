// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xi8>, tensor<2x7xi8>)
    %1 = call @expected() : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<5x6x7xi8>, tensor<2x2xi64>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    return %2 : tensor<5x6x7xi8>
  }
  func.func private @inputs() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}, tensor<2x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFDFF0500030103000000FDFC050201030300FDFD00000200FF06000AFE00020003050404FBFBFC010103FF00010001FEFFFD0100FE00FE000501F802FEFF00FE0205FB02FC00010000FEFDFD0002F90301FE03FCFDFDFC01FAFF00FF00FE02FD01FD000005FE000102FFFA00FD00010302FEFDFEFE06000000FEFF0003FFFE0006FEFE010202FF020501FEFF02020002FF00FEFDFEFE01FD00FE0000FEFCFD030103FB00FF000202FEFCFF0009040101FD0100030204030201FEF70400FD00FFFEFFFF06010402FFFF030002020003FE0000"> : tensor<5x6x7xi8>
    %c_0 = stablehlo.constant dense<[[-2, 2, -3, 0, 1, 1, -2], [6, -1, 0, 1, -5, 0, -1]]> : tensor<2x7xi8>
    return %c, %c_0 : tensor<5x6x7xi8>, tensor<2x7xi8>
  }
  func.func private @expected() -> (tensor<5x6x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFDFF0500030103FE02FDFDFD060001030300FDFD00000200FF06000AFE00020003050404FBFBFC010103FF00010001FEFFFD0100FE00FE000501F802FEFF00FE0205FB02FC00010000FEFDFD0002F90301FE03FCFDFDFC01FAFF00FF00FE02FD01FD000005FE00010205F900FEFB010202FEFDFEFE06000000FEFF0003FFFE0006FEFE010202FF020501FEFF02020002FF00FEFDFEFE01FD00FE0000FEFCFD030103FB00FF000202FEFCFF0009040101FD0100030204030201FEF70400FD00FFFEFFFF06010402FFFF030002020003FE0000"> : tensor<5x6x7xi8>
    return %c : tensor<5x6x7xi8>
  }
}
