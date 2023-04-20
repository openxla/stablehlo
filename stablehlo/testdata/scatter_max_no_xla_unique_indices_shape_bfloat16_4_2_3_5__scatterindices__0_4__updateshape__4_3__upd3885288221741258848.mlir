// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>)
    %2 = call @expected() : () -> tensor<4x2x3x5xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xbf16>, tensor<2xi32>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>) {
    %0 = stablehlo.constant dense<"0xACBD194032C08A409ABD8ABE933FD03F8F400B3E1CC0DF3F5240D7BF0DC08940BF40B9BF45C05740ADBE33409E3FFCBF8DBFE1BFBE3FDFBFA23FA43F96BF54C0ABC0A9C0DD3FBCBD3B40483E90400DC0173EC4BF25C06AC0874007BFB8BF5BC092C08F3F3A3FA2C01FBF82BF0940C73E75BF8DC0E5BF29C0BC3F9540EDBF943F5840A8C00EBFEDBF0EC12F40203F4DBFBE3F20C062C01D3D41C072BE16C087BFF03F02C08E4088408A408B3E193F16BF8B3FAEBE8BBF87BE8CBFD03EB9C0B53D29BED03F82C0823F963F13C071C0B03F82BF1240FCBF97BE56408B3F194095C00F3F0CC039BF2DC07D405C40293E503E"> : tensor<4x2x3x5xbf16>
    %1 = stablehlo.constant dense<[[-1.437500e+00, -8.398430e-01, 3.574220e-01], [1.507810e+00, -2.437500e+00, -2.859380e+00], [9.375000e-01, -1.109380e+00, 2.171880e+00], [4.718750e+00, -1.179690e+00, 2.093750e+00]]> : tensor<4x3xbf16>
    return %0, %1 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xbf16> {
    %0 = stablehlo.constant dense<"0xACBD194032C08A409ABD8ABE933FD03F8F400B3E1CC0DF3F5240D7BFB73E8940BF40B9BF45C05740ADBE33409E3FFCBF8DBFE1BFBE3FDFBFA23FA43F96BF54C0ABC0A9C0DD3FBCBD3B40483E90400DC0173EC4BF25C06AC0874007BFB8BF5BC092C08F3F3A3FA2C01FBF82BF0940C73E75BF8DC0E5BF29C0BC3F9540EDBF943F5840A8C00EBFEDBF0EC12F40203F4DBFBE3F20C00B401D3D41C072BE16C087BFF03F02C08E4088408A408B3E193F16BF8B3FAEBE8BBF87BE8CBFD03E9740B53D29BED03F82C0823F963F13C071C0B03F06401240FCBF97BE56408B3F194095C00F3F0CC039BF2DC07D405C40293E503E"> : tensor<4x2x3x5xbf16>
    return %0 : tensor<4x2x3x5xbf16>
  }
}

