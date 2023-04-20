// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>)
    %2 = call @expected() : () -> tensor<4x2x3x5xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi8>, tensor<2xi32>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>) {
    %0 = stablehlo.constant dense<"0x0302FFFC06000207FFFCFEFEFF0106000304FDFF01000102FF03FF00FF0000FF020002FB00FE0200FF01FB000102020100020002FB0003FEFE0403FF0000000000FF03000400FEFF0404FE02FEFEFF00000101000000FF0600000100FD0005FEFF02FE0000FF01000200FE000000FEFB000100020201FFFE"> : tensor<4x2x3x5xi8>
    %1 = stablehlo.constant dense<[[5, 1, 0], [-4, 3, 0], [0, 1, 0], [0, -4, 0]]> : tensor<4x3xi8>
    return %0, %1 : tensor<4x2x3x5xi8>, tensor<4x3xi8>
  }
  func.func private @expected() -> tensor<4x2x3x5xi8> {
    %0 = stablehlo.constant dense<"0x0302FFFC06000207FF01FEFEFF0106000304FDFF01000102FF03FF00FF0000FF020002FB00FE0203FF01FB000102020100020002FB0003FEFE0403FF0000000000FF03000401FEFF04040002FEFEFF00000101000000FF0600000100FD0005FEFF02FE0000FF01000200FE000000FEFB000100020201FFFE"> : tensor<4x2x3x5xi8>
    return %0 : tensor<4x2x3x5xi8>
  }
}

