// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xf16>, tensor<4x6xf16>)
    %1 = call @expected() : () -> tensor<3x5xf16>
    %2 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>, %arg2: tensor<f16>, %arg3: tensor<f16>):
      %6 = stablehlo.compare  GE, %arg0, %arg2,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<f16>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f16>
      stablehlo.return %7, %8 : tensor<f16>, tensor<f16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf16>, tensor<4x6xf16>, tensor<f16>, tensor<f16>) -> (tensor<3x5xf16>, tensor<3x5xf16>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<3x5xf16>, tensor<3x5xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xf16>, tensor<4x6xf16>) {
    %0 = stablehlo.constant dense<[[1.569340e+00, -2.009770e+00, 7.700190e-01, 2.480470e+00, 3.242190e+00, 1.518550e+00], [7.589840e+00, 2.556640e+00, 1.146480e+00, -1.239260e+00, 4.360350e-01, -4.656250e+00], [3.835940e+00, 1.017460e-01, 5.343750e+00, 3.216800e+00, -8.295890e-01, 5.750000e+00], [7.281250e+00, -1.558590e+00, -3.376950e+00, -7.534170e-01, 1.650390e+00, -3.751950e+00]]> : tensor<4x6xf16>
    %1 = stablehlo.constant dense<[[1.895510e+00, -1.314450e+00, 4.328130e+00, -3.505860e+00, 4.042970e+00, -1.884770e+00], [-7.543940e-01, -2.771000e-01, 2.904300e+00, 1.837890e+00, -1.598630e+00, -2.759770e+00], [-5.874020e-01, 5.238280e+00, -1.217770e+00, -4.187500e+00, -2.304690e+00, -3.783200e+00], [-3.337890e+00, 6.049800e-01, 4.937500e+00, -2.716800e+00, -3.718750e+00, -9.487300e-01]]> : tensor<4x6xf16>
    return %0, %1 : tensor<4x6xf16>, tensor<4x6xf16>
  }
  func.func private @expected() -> tensor<3x5xf16> {
    %0 = stablehlo.constant dense<[[1.569340e+00, 7.700190e-01, 7.700190e-01, 3.242190e+00, 3.242190e+00], [1.017460e-01, 1.017460e-01, 1.146480e+00, -1.239260e+00, 4.360350e-01], [1.017460e-01, 1.017460e-01, -3.376950e+00, -8.295890e-01, -3.751950e+00]]> : tensor<3x5xf16>
    return %0 : tensor<3x5xf16>
  }
}

