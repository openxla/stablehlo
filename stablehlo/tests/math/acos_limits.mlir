// RUN: stablehlo-opt --chlo-legalize-to-stablehlo %s | stablehlo-translate --interpret

func.func @main() -> (tensor<f64>, tensor<complex<f64>>) {
  %cst = stablehlo.constant dense<-1.000000e+00> : tensor<f64>
  %cst_0 = stablehlo.constant dense<(-1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
  %zero = stablehlo.constant dense<0.0> : tensor<f64>
  %pi = stablehlo.constant dense<3.1415926535897931> : tensor<f64>
  %complex_pi = stablehlo.complex %pi, %zero : tensor<complex<f64>>
  %0 = chlo.acos %cst : tensor<f64> -> tensor<f64>
  %1 = chlo.acos %cst_0 : tensor<complex<f64>> -> tensor<complex<f64>>
  check.expect_close %0, %pi, max_ulp_difference = 1 : tensor<f64>, tensor<f64>
  check.expect_close %1, %complex_pi, max_ulp_difference = 1 : tensor<complex<f64>>, tensor<complex<f64>>
  return %0, %1 : tensor<f64>, tensor<complex<f64>>
}
