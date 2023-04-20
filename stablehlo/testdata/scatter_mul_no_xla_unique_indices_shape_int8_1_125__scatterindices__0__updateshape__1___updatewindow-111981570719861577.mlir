// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xi8>, tensor<1xi8>)
    %2 = call @expected() : () -> tensor<1x125xi8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x125xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xi8>, tensor<1x125xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xi8>, tensor<1xi8>) {
    %0 = stablehlo.constant dense<"0x02FDFF030006060000FE0100FFFFFD03FF010300FF06FC00FF0003000403FC030100030403FE02FEFF00FE01FE000403020000FEFC01FCFE01FA0100010003040401FCFF00FF04FEFDFFFA020000FE02FEFF00FCFFFDFFFFFF0402FF00FFFBFDFF01020001FDFF000800FE00050104FDFFFE00FEFD00FFFFFE000000FE"> : tensor<1x125xi8>
    %1 = stablehlo.constant dense<0> : tensor<1xi8>
    return %0, %1 : tensor<1x125xi8>, tensor<1xi8>
  }
  func.func private @expected() -> tensor<1x125xi8> {
    %0 = stablehlo.constant dense<"0x00FDFF030006060000FE0100FFFFFD03FF010300FF06FC00FF0003000403FC030100030403FE02FEFF00FE01FE000403020000FEFC01FCFE01FA0100010003040401FCFF00FF04FEFDFFFA020000FE02FEFF00FCFFFDFFFFFF0402FF00FFFBFDFF01020001FDFF000800FE00050104FDFFFE00FEFD00FFFFFE000000FE"> : tensor<1x125xi8>
    return %0 : tensor<1x125xi8>
  }
}

