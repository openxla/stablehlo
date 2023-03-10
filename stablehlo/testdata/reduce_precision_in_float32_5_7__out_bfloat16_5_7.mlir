// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5x7xf32>
    %1 = call @expected() : () -> tensor<5x7xf32>
    %2 = stablehlo.reduce_precision %0, format = e8m7 : tensor<5x7xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<5x7xf32>, tensor<5x7xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5x7xf32> {
    %0 = stablehlo.constant dense<[[2.25389457, -3.57496953, 2.07120371, -3.46210718, -2.64704967, 1.82141495, 2.90927529], [0.432021976, 4.74591446, 0.55090344, 3.30714226, 0.176475316, -4.552080e+00, -0.0443474688], [0.0792889595, 1.43216443, -0.969412326, 0.720816671, 0.984933853, -5.63764048, -3.35218692], [2.61366272, -3.74551773, -1.4951669, -2.35648894, -0.951688528, -1.53872073, -2.18088317], [5.72941971, -0.0653350875, -0.305840045, -3.62638831, 4.48467255, -0.989226281, -4.62714243]]> : tensor<5x7xf32>
    return %0 : tensor<5x7xf32>
  }
  func.func private @expected() -> tensor<5x7xf32> {
    %0 = stablehlo.constant dense<[[2.250000e+00, -3.578125, 2.078125, -3.468750e+00, -2.640625, 1.8203125, 2.906250e+00], [0.431640625, 4.750000e+00, 0.55078125, 3.312500e+00, 0.176757813, -4.562500e+00, -0.0444335938], [0.0791015625, 1.4296875, -9.687500e-01, 0.72265625, 9.843750e-01, -5.625000e+00, -3.359375], [2.609375, -3.750000e+00, -1.4921875, -2.359375, -9.531250e-01, -1.5390625, -2.187500e+00], [5.718750e+00, -0.0654296875, -0.306640625, -3.625000e+00, 4.500000e+00, -0.98828125, -4.625000e+00]]> : tensor<5x7xf32>
    return %0 : tensor<5x7xf32>
  }
}
