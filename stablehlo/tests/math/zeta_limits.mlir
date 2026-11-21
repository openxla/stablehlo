// RUN: stablehlo-opt --chlo-legalize-to-stablehlo %s | stablehlo-translate --interpret

// The Euler-Maclaurin approximation used by the zeta decomposition is only
// valid for q > 0. Check that zeta(x, q) for negative q and integer x, and
// hence polygamma(n, x) for negative x, is accurate also for q, x < -10.
// Expected values were computed with mpmath.

func.func @main() {
  %x_f64 = stablehlo.constant dense<[2.0, 2.0, 2.0, 3.0, 4.0, 7.0, 3.0]> : tensor<7xf64>
  %q_f64 = stablehlo.constant dense<[0.5, -0.5, -9.5, -20.3, -1000.25, -9.5, -22.5]> : tensor<7xf64>
  %zeta_f64 = chlo.zeta %x_f64, %q_f64 : tensor<7xf64>, tensor<7xf64> -> tensor<7xf64>
  %zeta_f64_expected = stablehlo.constant dense<[4.934802200544679, 8.934802200544679, 9.769687445030232, -34.41761682891865, 259.7575760903406, 1.638095413559251e-07, 9.447333934312955e-04]> : tensor<7xf64>
  check.expect_close %zeta_f64, %zeta_f64_expected, max_ulp_difference = 4 : tensor<7xf64>, tensor<7xf64>

  %n_f64 = stablehlo.constant dense<[1.0, 1.0, 1.0, 2.0, 4.0, 1.0]> : tensor<6xf64>
  %y_f64 = stablehlo.constant dense<[-16.5, -9.5, -100.3, -8.5, -14.5, 2.5]> : tensor<6xf64>
  %polygamma_f64 = chlo.polygamma %n_f64, %y_f64 : tensor<6xf64>, tensor<6xf64> -> tensor<6xf64>
  %polygamma_f64_expected = stablehlo.constant dense<[9.810797812993576, 9.769687445030232, 15.06949314924401, -1.2307845807709338e-02, -1.180819310106e-04, 0.49035775610023485]> : tensor<6xf64>
  check.expect_close %polygamma_f64, %polygamma_f64_expected, max_ulp_difference = 16 : tensor<6xf64>, tensor<6xf64>

  %x_f32 = stablehlo.constant dense<[2.0, 2.0, 2.0, 3.0, 4.0, 7.0, 3.0]> : tensor<7xf32>
  %q_f32 = stablehlo.constant dense<[0.5, -0.5, -9.5, -20.3, -1000.25, -9.5, -22.5]> : tensor<7xf32>
  %zeta_f32 = chlo.zeta %x_f32, %q_f32 : tensor<7xf32>, tensor<7xf32> -> tensor<7xf32>
  %zeta_f32_expected = stablehlo.constant dense<[4.934802, 8.934802, 9.769688, -34.41791, 259.75757, 1.6380955e-07, 9.447334e-04]> : tensor<7xf32>
  check.expect_close %zeta_f32, %zeta_f32_expected, max_ulp_difference = 4 : tensor<7xf32>, tensor<7xf32>

  %n_f32 = stablehlo.constant dense<[1.0, 1.0, 1.0, 2.0, 4.0, 1.0]> : tensor<6xf32>
  %y_f32 = stablehlo.constant dense<[-16.5, -9.5, -100.3, -8.5, -14.5, 2.5]> : tensor<6xf32>
  %polygamma_f32 = chlo.polygamma %n_f32, %y_f32 : tensor<6xf32>, tensor<6xf32> -> tensor<6xf32>
  %polygamma_f32_expected = stablehlo.constant dense<[9.810798, 9.769688, 15.0692835, -1.2307846e-02, -1.1808193e-04, 0.49035776]> : tensor<6xf32>
  check.expect_close %polygamma_f32, %polygamma_f32_expected, max_ulp_difference = 32 : tensor<6xf32>, tensor<6xf32>
  func.return
}
