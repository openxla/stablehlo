// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x6xf16>, tensor<4x6xf16>)
    %1 = call @expected() : () -> tensor<3x5xf16>
    %cst = stablehlo.constant dense<0x7C00> : tensor<f16>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %2:2 = "stablehlo.reduce_window"(%0#1, %0#0, %cst, %cst_0) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>, %arg2: tensor<f16>, %arg3: tensor<f16>):
      %3 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %4 = stablehlo.select %3, %arg0, %arg2 : tensor<i1>, tensor<f16>
      %5 = stablehlo.select %3, %arg1, %arg3 : tensor<i1>, tensor<f16>
      stablehlo.return %4, %5 : tensor<f16>, tensor<f16>
    }) : (tensor<4x6xf16>, tensor<4x6xf16>, tensor<f16>, tensor<f16>) -> (tensor<3x5xf16>, tensor<3x5xf16>)
    stablehlo.custom_call @check.expect_close(%2#1, %1) {has_side_effect = true} : (tensor<3x5xf16>, tensor<3x5xf16>) -> ()
    return %2#1 : tensor<3x5xf16>
  }
  func.func private @inputs() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}, tensor<4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.392580e+00, 3.939450e+00, -5.648440e+00, 8.652340e-01, 1.454100e+00, 6.093750e-01], [3.189450e+00, -7.421880e-01, -1.119140e+00, -2.177730e+00, 2.199220e+00, -7.075190e-01], [4.699710e-01, -8.984370e-01, -1.866210e+00, 7.580560e-02, -7.613280e+00, -7.242180e+00], [4.109380e+00, 3.443360e+00, 2.589840e+00, 1.947270e+00, -3.361820e-01, -1.671880e+00]]> : tensor<4x6xf16>
    %cst_0 = stablehlo.constant dense<[[2.197270e+00, 1.117190e+00, 3.562010e-01, 3.738280e+00, -2.197270e+00, 1.822270e+00], [8.984370e-01, -8.282470e-02, -1.631840e+00, -5.304690e+00, -2.890630e+00, 3.591800e+00], [5.402340e+00, 1.822270e+00, 7.626950e-01, 4.664060e+00, -9.718750e+00, -1.122070e+00], [3.025390e+00, 1.737060e-01, 1.925050e-01, -7.617180e-01, 4.625000e+00, 1.282230e+00]]> : tensor<4x6xf16>
    return %cst, %cst_0 : tensor<4x6xf16>, tensor<4x6xf16>
  }
  func.func private @expected() -> (tensor<3x5xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-7.421880e-01, -1.119140e+00, -2.177730e+00, -2.177730e+00, 2.199220e+00], [-7.421880e-01, -1.119140e+00, -2.177730e+00, -7.613280e+00, -7.613280e+00], [3.443360e+00, 3.443360e+00, 1.947270e+00, -7.613280e+00, -7.613280e+00]]> : tensor<3x5xf16>
    return %cst : tensor<3x5xf16>
  }
}
