// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xbf16>
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %cst = stablehlo.constant dense<0x7F80> : tensor<bf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<bf16>
    %3 = "stablehlo.reduce_window"(%0, %2) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %4 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %4 : tensor<bf16>
    }) : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> ()
    return %3 : tensor<3x5xbf16>
  }
  func.func private @inputs() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-3.609380e+00, 3.031250e+00, 2.968750e+00, 2.093750e+00, 1.304690e+00, -1.765630e+00], [-4.937500e+00, 5.250000e+00, 4.648440e-01, -8.359380e-01, -6.328130e-01, -1.777340e-01], [5.468750e+00, -1.742190e+00, 4.687500e+00, -1.175000e+01, 9.570310e-01, 6.906250e+00], [8.164060e-01, -2.406250e+00, 3.375000e+00, 2.921880e+00, -4.937500e+00, -2.859380e+00]]> : tensor<4x6xbf16>
    return %cst : tensor<4x6xbf16>
  }
  func.func private @expected() -> (tensor<3x5xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.937500e+00, 4.648440e-01, -8.359380e-01, -8.359380e-01, -1.765630e+00], [-4.937500e+00, -1.742190e+00, -1.175000e+01, -1.175000e+01, -6.328130e-01], [-2.406250e+00, -2.406250e+00, -1.175000e+01, -1.175000e+01, -4.937500e+00]]> : tensor<3x5xbf16>
    return %cst : tensor<3x5xbf16>
  }
}
