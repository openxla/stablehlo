// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xi8>, tensor<1x3xi8>)
    %2 = call @expected() : () -> tensor<1x50x3xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xi8>, tensor<1xi32>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xi8>, tensor<1x3xi8>) {
    %0 = stablehlo.constant dense<"0x0000FFFFFF020700020303FEFE00FF0205FF04FD02010302FD09FD000201000005030200FAF9FFFD00FF0201FE05FE0201FF0001FC0300090601FFFC030500000000FF0003FC01000000FF00FD01000200FE0001FE040101FE0000FC01FC0004FB0000FFFD0000FCFF03FCFC03FD050002FD0102FD0203020002F904FE0304FEFD0204020102000000FA04FE0106FFFF01FF02FCFCFD"> : tensor<1x50x3xi8>
    %1 = stablehlo.constant dense<[[4, -1, 0]]> : tensor<1x3xi8>
    return %0, %1 : tensor<1x50x3xi8>, tensor<1x3xi8>
  }
  func.func private @expected() -> tensor<1x50x3xi8> {
    %0 = stablehlo.constant dense<"0x0000FFFFFF020700020303FEFE00FF0205FF04FD02010302FD09FD000201000005030200FAF9FFFD00FF0201FE05FE0201FF0001FC0300090601FFFC030500000000FF0003FC01000000FF00FD01000200FE0001FE040101FE0000FC01FC0004040000FFFD0000FCFF03FCFC03FD050002FD0102FD0203020002F904FE0304FEFD0204020102000000FA04FE0106FFFF01FF02FCFCFD"> : tensor<1x50x3xi8>
    return %0 : tensor<1x50x3xi8>
  }
}

