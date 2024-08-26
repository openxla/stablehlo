// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xf16>
    %1 = call @expected() : () -> tensor<3x5xf16>
    %cst = stablehlo.constant dense<0x7C00> : tensor<f16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<f16>
    %3 = "stablehlo.reduce_window"(%0, %2) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %4 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %4 : tensor<f16>
    }) : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<3x5xf16>, tensor<3x5xf16>) -> ()
    return %3 : tensor<3x5xf16>
  }
  func.func private @inputs() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.085940e+00, 1.006840e+00, 1.763670e+00, -5.851560e+00, 5.908200e-01, -2.433590e+00], [-5.878900e+00, -2.517090e-01, 3.554690e+00, 4.136720e+00, -2.068360e+00, -1.481450e+00], [3.496090e+00, 1.714840e+00, -6.791990e-01, -7.260740e-01, 3.552730e+00, -2.517580e+00], [-6.269530e+00, 6.977530e-01, 4.808590e+00, 2.189450e+00, -6.234380e+00, 5.789060e+00]]> : tensor<4x6xf16>
    return %cst : tensor<4x6xf16>
  }
  func.func private @expected() -> (tensor<3x5xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-5.878900e+00, -2.517090e-01, -5.851560e+00, -5.851560e+00, -2.433590e+00], [-5.878900e+00, -6.791990e-01, -7.260740e-01, -2.068360e+00, -2.517580e+00], [-6.269530e+00, -6.791990e-01, -7.260740e-01, -6.234380e+00, -6.234380e+00]]> : tensor<3x5xf16>
    return %cst : tensor<3x5xf16>
  }
}
