// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xbf16> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xbf16>
    %1 = call @expected() : () -> tensor<5x7xbf16>
    %2 = "stablehlo.sort"(%0) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %3 = stablehlo.compare  EQ, %arg0, %cst,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %4 = stablehlo.select %3, %cst_0, %arg0 : tensor<i1>, tensor<bf16>
      %5 = stablehlo.compare  NE, %arg0, %arg0,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %cst_1 = stablehlo.constant dense<0x7FC0> : tensor<bf16>
      %6 = stablehlo.select %5, %cst_1, %4 : tensor<i1>, tensor<bf16>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %7 = stablehlo.compare  EQ, %arg1, %cst_2,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
      %8 = stablehlo.select %7, %cst_3, %arg1 : tensor<i1>, tensor<bf16>
      %9 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %cst_4 = stablehlo.constant dense<0x7FC0> : tensor<bf16>
      %10 = stablehlo.select %9, %cst_4, %8 : tensor<i1>, tensor<bf16>
      %11 = stablehlo.compare  LT, %6, %10,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %11 : tensor<i1>
    }) : (tensor<5x7xbf16>) -> tensor<5x7xbf16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x7xbf16>, tensor<5x7xbf16>) -> ()
    return %2 : tensor<5x7xbf16>
  }
  func.func private @inputs() -> (tensor<5x7xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.406250e+00, 3.421880e+00, 3.593750e+00, -3.750000e-01, -4.174800e-02, 1.982420e-01, 6.591800e-02], [-6.152340e-02, 1.031250e+00, -7.695310e-01, -8.945310e-01, 7.695310e-01, 2.203130e+00, -2.468750e+00], [4.343750e+00, -4.125000e+00, 3.140630e+00, -9.882810e-01, -1.070310e+00, -5.937500e-01, -3.750000e-01], [-1.240230e-01, -3.046880e+00, 1.937500e+00, -1.460940e+00, -6.884770e-02, -2.687500e+00, -3.468750e+00], [-4.156250e+00, -6.796880e-01, -4.312500e+00, 3.515630e-01, -6.312500e+00, -5.195310e-01, 1.531250e+00]]> : tensor<5x7xbf16>
    return %cst : tensor<5x7xbf16>
  }
  func.func private @expected() -> (tensor<5x7xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.156250e+00, -4.125000e+00, -4.312500e+00, -1.460940e+00, -6.312500e+00, -2.687500e+00, -3.468750e+00], [-1.240230e-01, -3.046880e+00, -7.695310e-01, -9.882810e-01, -1.070310e+00, -5.937500e-01, -2.468750e+00], [-6.152340e-02, -6.796880e-01, 1.937500e+00, -8.945310e-01, -6.884770e-02, -5.195310e-01, -3.750000e-01], [2.406250e+00, 1.031250e+00, 3.140630e+00, -3.750000e-01, -4.174800e-02, 1.982420e-01, 6.591800e-02], [4.343750e+00, 3.421880e+00, 3.593750e+00, 3.515630e-01, 7.695310e-01, 2.203130e+00, 1.531250e+00]]> : tensor<5x7xbf16>
    return %cst : tensor<5x7xbf16>
  }
}
