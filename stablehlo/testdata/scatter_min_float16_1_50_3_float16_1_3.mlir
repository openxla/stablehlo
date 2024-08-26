// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x50x3xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<32> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x50x3xf16>, tensor<1x3xf16>)
    %1 = call @expected() : () -> tensor<1x50x3xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<1x50x3xf16>, tensor<1xi64>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> ()
    return %2 : tensor<1x50x3xf16>
  }
  func.func private @inputs() -> (tensor<1x50x3xf16> {mhlo.layout_mode = "default"}, tensor<1x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x4A34B9BB1730ACB9364032444D3C63C09543A9B797C55CC42FC481C0B94596BCA6C09F423FC1603EB2454BB629C54445803E1C3ED9BE7E4043C69E45D4BA333282BCA5C572C0C742434363450248DA419840A03F1D4553356E4125B677374638BCC59F4445C05E416EC11DB4143913C209BE97BEE2B9A6CAC24026C4A538BA425148BEC451C25CBCCE3EF8B54BC0883A1940A23D0A39F0C01840F43875C06AB93FC02EC6D33C4BB49DC3573E6C3716C20F302641D94420B3ABBAFDC2B64102B035BFEDBB81C645AD4A40FF41FEBF9F46C93510BF383E633843C028432240BBBC38B73A415BBA0B3EA1C3253F82C29FAE6FC35636D1C3883DCCC2B8461FC68CBC30C2EBC4BA44A7C0524019C4F54398358EC353C2BE41913EE23832B861A6BCB6A6BFF73B27C4BD374DC0B9BF"> : tensor<1x50x3xf16>
    %cst_0 = stablehlo.constant dense<[[-1.188480e+00, 2.365230e+00, 4.773440e+00]]> : tensor<1x3xf16>
    return %cst, %cst_0 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> (tensor<1x50x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x4A34B9BB1730ACB9364032444D3C63C09543A9B797C55CC42FC481C0B94596BCA6C09F423FC1603EB2454BB629C54445803E1C3ED9BE7E4043C69E45D4BA333282BCA5C572C0C742434363450248DA419840A03F1D4553356E4125B677374638BCC59F4445C05E416EC11DB4143913C209BE97BEE2B9A6CAC24026C4A538BA425148BEC451C25CBCCE3EF8B54BC0883A1940A23D0A39F0C01840F43875C06AB93FC02EC6D33C4BB49DC3573E6C3716C20F302641D94420B3ABBAFDC2B64102B035BFEDBB81C645AD4A40FF41FEBF9F46C93510BF383E633843C028432240BBBC38B73A415BBA0B3EA1C3253F82C29FAE6FC35636D1C3883DCCC2B8461FC68CBC30C2EBC4BA44A7C0524019C4F54398358EC353C2BE41913EE23832B861A6BCB6A6BFF73B27C4BD374DC0B9BF"> : tensor<1x50x3xf16>
    return %cst : tensor<1x50x3xf16>
  }
}
