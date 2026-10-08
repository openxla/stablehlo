// RUN: stablehlo-opt --chlo-legalize-to-stablehlo --split-input-file %s | stablehlo-translate --interpret -split-input-file

func.func @zeta_underflow_f32() {
  %s = stablehlo.constant dense<4.0> : tensor<f32>
  %q = stablehlo.constant dense<3.037000576E+09> : tensor<f32>
  %zero = stablehlo.constant dense<0.0> : tensor<f32>
  %scale = stablehlo.constant dense<1.0E+29> : tensor<f32>

  %result = chlo.zeta %s, %q : tensor<f32>, tensor<f32> -> tensor<f32>
  %positive = stablehlo.compare GT, %result, %zero : (tensor<f32>, tensor<f32>) -> tensor<i1>
  check.expect_eq_const %positive, dense<true> : tensor<i1>

  %scaled = stablehlo.multiply %result, %scale : tensor<f32>
  check.expect_almost_eq_const %scaled, dense<1.1899921> : tensor<f32>
  func.return
}

// -----

func.func @zeta_underflow_f64() {
  %s = stablehlo.constant dense<4.0> : tensor<f64>
  %q = stablehlo.constant dense<1.0E+77> : tensor<f64>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %scale = stablehlo.constant dense<1.0E+232> : tensor<f64>

  %result = chlo.zeta %s, %q : tensor<f64>, tensor<f64> -> tensor<f64>
  %positive = stablehlo.compare GT, %result, %zero : (tensor<f64>, tensor<f64>) -> tensor<i1>
  check.expect_eq_const %positive, dense<true> : tensor<i1>

  %scaled = stablehlo.multiply %result, %scale : tensor<f64>
  check.expect_almost_eq_const %scaled, dense<3.333333333333333> : tensor<f64>
  func.return
}
