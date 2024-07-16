// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xf16>
    %1 = call @expected() : () -> tensor<3x5xf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %2 = "stablehlo.reduce_window"(%0, %cst) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5xf16>, tensor<3x5xf16>) -> ()
    return %2 : tensor<3x5xf16>
  }
  func.func private @inputs() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.824220e+00, -2.403560e-01, 3.193360e-01, 2.466800e+00, -1.509770e+00, 4.246090e+00], [-7.609380e+00, -5.324220e+00, -3.806150e-01, -2.015380e-01, 2.214840e+00, 5.929690e+00], [3.134770e+00, 2.857420e+00, 4.218750e+00, -1.056640e+00, 3.454590e-01, -2.564450e+00], [-1.566410e+00, 2.164060e+00, -3.417970e+00, 2.779300e+00, 3.009770e+00, -2.099610e+00]]> : tensor<4x6xf16>
    return %cst : tensor<4x6xf16>
  }
  func.func private @expected() -> (tensor<3x5xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.000000e+00, 1.000000e+00, 2.466800e+00, 2.466800e+00, 5.929690e+00], [3.134770e+00, 4.218750e+00, 4.218750e+00, 2.214840e+00, 5.929690e+00], [3.134770e+00, 4.218750e+00, 4.218750e+00, 3.009770e+00, 3.009770e+00]]> : tensor<3x5xf16>
    return %cst : tensor<3x5xf16>
  }
}
