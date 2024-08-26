// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xbf16>
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = "stablehlo.reduce_window"(%0, %cst) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> ()
    return %2 : tensor<3x5xbf16>
  }
  func.func private @inputs() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.023440e+00, 4.750000e+00, -4.218750e+00, -3.925780e-01, 4.593750e+00, -1.455080e-01], [2.093750e+00, -1.906250e+00, 2.093750e+00, 3.703130e+00, 8.906250e-01, -1.357420e-01], [2.953130e+00, -1.359380e+00, 2.984380e+00, 6.992180e-01, 4.250000e+00, -2.578130e+00], [-3.796880e+00, 2.062500e+00, 2.937500e+00, 1.109380e+00, -2.078130e+00, -2.203130e+00]]> : tensor<4x6xbf16>
    return %cst : tensor<4x6xbf16>
  }
  func.func private @expected() -> (tensor<3x5xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.906250e+00, 1.718750e+00, 2.187500e+00, 9.750000e+00, 6.187500e+00], [2.765630e+00, 2.812500e+00, 1.050000e+01, 1.050000e+01, 3.421880e+00], [8.593750e-01, 7.625000e+00, 8.750000e+00, 5.000000e+00, -1.609380e+00]]> : tensor<3x5xbf16>
    return %cst : tensor<3x5xbf16>
  }
}
