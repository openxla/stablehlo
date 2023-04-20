// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xf16>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xf16>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : (tensor<3x6xf16>) -> tensor<3x6xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xf16>, tensor<3x6xf16>) {
    %0 = stablehlo.constant dense<[[9.077140e-01, 1.049800e+00, -4.757810e+00], [-2.894530e+00, 2.638670e+00, 1.275630e-01], [1.354490e+00, 4.488280e+00, 3.138670e+00], [-4.228520e-01, 4.218750e+00, 1.034180e+00]]> : tensor<4x3xf16>
    %1 = stablehlo.constant dense<[[-9.531250e-01, -3.810550e+00, -2.663570e-01, 2.607420e+00, -2.109380e+00, 2.542970e+00], [-1.798830e+00, 4.289060e+00, -9.238280e-01, 7.402340e-01, 9.189450e-01, 2.771480e+00], [-6.113280e-01, 1.085820e-01, -5.222660e+00, 3.832030e+00, 5.183590e+00, -4.589840e-01]]> : tensor<3x6xf16>
    return %0, %1 : tensor<4x3xf16>, tensor<3x6xf16>
  }
  func.func private @expected() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[0.155000687, 0.527177334, 23.6368027, -15.0881891, -25.6125641, 7.40156937], [-2.06565022, 22.3610268, -2.3329196, -5.10520124, 9.19168472, -0.106214285], [-11.2834053, 14.429965, -20.8993835, 18.8816013, 17.536953, 14.4430313], [-7.81799888, 19.8180714, -9.18593502, 5.983320e+00, 10.1295204, 10.1422291]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
}

