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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %5 : tensor<i8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xi8>, tensor<2xi32>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>) {
    %0 = stablehlo.constant dense<"0x01040000FEFCFF0000FC010201FFFC0701FE0300030100FFFC00FC0301FE0301FF0000030101000403FA000001FFFFFFFF000600FB0000FFFF00FFFEFC03FF03FDFF00000104FE0005FC0000000500FFFDFD0002000001FE000002020101020000000202000103FFFFFB0800010303FEFEF900FF01FDFD00"> : tensor<4x2x3x5xi8>
    %1 = stablehlo.constant dense<[[0, -1, 1], [1, -1, 0], [6, 1, -1], [0, 0, 3]]> : tensor<4x3xi8>
    return %0, %1 : tensor<4x2x3x5xi8>, tensor<4x3xi8>
  }
  func.func private @expected() -> tensor<4x2x3x5xi8> {
    %0 = stablehlo.constant dense<"0x01040000FEFCFF0000FC010201FFFC0701FE0300030100FFFC00FC0301FE0301FF000003010100FF03FA000000FFFFFFFF000600FB0000FFFF00FFFEFC03FF03FDFF00000101FE0005FCFF00000500FFFDFD0002000001FE000002020101000000000200000103FFFFFB0800010303FEFEF900FF01FDFD00"> : tensor<4x2x3x5xi8>
    return %0 : tensor<4x2x3x5xi8>
  }
}

