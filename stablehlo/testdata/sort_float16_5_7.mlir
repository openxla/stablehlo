// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xf16> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xf16>
    %1 = call @expected() : () -> tensor<5x7xf16>
    %2 = "stablehlo.sort"(%0) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %3 = stablehlo.compare  EQ, %arg0, %cst,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %4 = stablehlo.select %3, %cst_0, %arg0 : tensor<i1>, tensor<f16>
      %5 = stablehlo.compare  NE, %arg0, %arg0,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %cst_1 = stablehlo.constant dense<0x7E00> : tensor<f16>
      %6 = stablehlo.select %5, %cst_1, %4 : tensor<i1>, tensor<f16>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %7 = stablehlo.compare  EQ, %arg1, %cst_2,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
      %8 = stablehlo.select %7, %cst_3, %arg1 : tensor<i1>, tensor<f16>
      %9 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %cst_4 = stablehlo.constant dense<0x7E00> : tensor<f16>
      %10 = stablehlo.select %9, %cst_4, %8 : tensor<i1>, tensor<f16>
      %11 = stablehlo.compare  LT, %6, %10,  TOTALORDER : (tensor<f16>, tensor<f16>) -> tensor<i1>
      stablehlo.return %11 : tensor<i1>
    }) : (tensor<5x7xf16>) -> tensor<5x7xf16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x7xf16>, tensor<5x7xf16>) -> ()
    return %2 : tensor<5x7xf16>
  }
  func.func private @inputs() -> (tensor<5x7xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.148440e+00, 4.452510e-02, 6.804680e+00, 1.814450e+00, 3.199220e+00, -1.463870e+00, 4.503910e+00], [-4.371090e+00, -1.822270e+00, -1.518550e-01, 4.132810e+00, -3.310550e+00, -8.613280e-01, 1.857420e+00], [-2.072270e+00, -2.316410e+00, -3.261720e-01, -9.545890e-01, -5.367190e+00, 1.135740e+00, 3.908200e+00], [-5.170900e-01, -1.863280e+00, 1.348630e+00, -2.921880e+00, 8.007810e-01, 3.398440e+00, -1.192380e+00], [2.033200e+00, -4.953130e+00, 1.804690e+00, -6.308590e-01, -3.638670e+00, -2.794920e+00, -6.117190e+00]]> : tensor<5x7xf16>
    return %cst : tensor<5x7xf16>
  }
  func.func private @expected() -> (tensor<5x7xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.371090e+00, -4.953130e+00, -3.261720e-01, -2.921880e+00, -5.367190e+00, -2.794920e+00, -6.117190e+00], [-2.072270e+00, -2.316410e+00, -1.518550e-01, -9.545890e-01, -3.638670e+00, -1.463870e+00, -1.192380e+00], [-5.170900e-01, -1.863280e+00, 1.348630e+00, -6.308590e-01, -3.310550e+00, -8.613280e-01, 1.857420e+00], [2.033200e+00, -1.822270e+00, 1.804690e+00, 1.814450e+00, 8.007810e-01, 1.135740e+00, 3.908200e+00], [2.148440e+00, 4.452510e-02, 6.804680e+00, 4.132810e+00, 3.199220e+00, 3.398440e+00, 4.503910e+00]]> : tensor<5x7xf16>
    return %cst : tensor<5x7xf16>
  }
}
