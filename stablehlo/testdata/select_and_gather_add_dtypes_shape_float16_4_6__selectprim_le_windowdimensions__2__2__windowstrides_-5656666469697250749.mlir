// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xf16>, tensor<4x6xf16>)
    %1 = call @expected() : () -> tensor<3x5xf16>
    %2 = stablehlo.constant dense<0x7C00> : tensor<f16>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>, %arg2: tensor<f16>, %arg3: tensor<f16>):
      %6 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<f16>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f16>
      stablehlo.return %7, %8 : tensor<f16>, tensor<f16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf16>, tensor<4x6xf16>, tensor<f16>, tensor<f16>) -> (tensor<3x5xf16>, tensor<3x5xf16>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<3x5xf16>, tensor<3x5xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xf16>, tensor<4x6xf16>) {
    %0 = stablehlo.constant dense<[[4.468750e+00, 4.007810e+00, 1.850590e+00, 3.376950e+00, -1.212890e+00, 2.431640e+00], [2.703130e+00, -1.466800e+00, 1.662110e+00, 5.839840e-01, 3.337890e+00, -1.627930e+00], [-4.936520e-01, -7.226560e-01, 1.649170e-01, -3.884770e+00, 7.230460e+00, -2.349610e+00], [2.475590e-01, 2.248050e+00, 3.415530e-01, -2.958980e-01, -2.945310e+00, 1.801760e+00]]> : tensor<4x6xf16>
    %1 = stablehlo.constant dense<[[9.877920e-01, 4.013670e-01, -6.843750e+00, 2.652340e+00, 5.902340e+00, -3.699220e+00], [5.374150e-02, 2.310550e+00, 3.777340e+00, -6.157230e-01, -6.010740e-01, -1.607420e+00], [9.291990e-01, -1.813480e+00, -6.243900e-02, -1.610350e+00, -2.949220e+00, -1.679690e+00], [-7.597650e+00, -9.326170e-01, 6.464840e-01, -4.171880e+00, 2.242190e+00, 2.251950e+00]]> : tensor<4x6xf16>
    return %0, %1 : tensor<4x6xf16>, tensor<4x6xf16>
  }
  func.func private @expected() -> tensor<3x5xf16> {
    %0 = stablehlo.constant dense<[[2.703130e+00, 1.850590e+00, 1.850590e+00, 5.839840e-01, 2.431640e+00], [-7.226560e-01, -7.226560e-01, -3.884770e+00, 7.230460e+00, 7.230460e+00], [2.475590e-01, -7.226560e-01, -2.958980e-01, -2.958980e-01, 7.230460e+00]]> : tensor<3x5xf16>
    return %0 : tensor<3x5xf16>
  }
}

