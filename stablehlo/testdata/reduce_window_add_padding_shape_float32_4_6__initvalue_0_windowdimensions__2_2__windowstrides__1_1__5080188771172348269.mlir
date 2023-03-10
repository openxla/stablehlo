// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<6x8xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {padding = dense<[[1, 2], [0, 3]]> : tensor<2x2xi64>, window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<6x8xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<6x8xf32>, tensor<6x8xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[-3.68880367, -4.02331495, -2.25862646, -1.11733317, 0.8114627, -3.74134445], [4.84204865, 3.72955298, 2.25633454, -2.72418094, -0.285849363, -2.41332364], [0.84224981, -3.04229522, 1.13644135, 1.40966439, -2.73121357, 1.85701787], [-4.51306868, 2.05381179, 0.81867963, 4.09194469, -0.237238407, 2.78918052]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<6x8xf32> {
    %0 = stablehlo.constant dense<[[-7.71211863, -6.28194141, -3.37595963, -0.305870473, -2.92988181, -3.74134445, 0.000000e+00, 0.000000e+00], [8.594830e-01, -0.296053886, -3.84380603, -3.3159008, -5.62905502, -6.15466785, 0.000000e+00, 0.000000e+00], [6.37155628, 4.08003378, 2.07825947, -4.33157921, -3.57336855, -0.556305766, 0.000000e+00, 0.000000e+00], [-4.65930176, 0.966637551, 7.456730e+00, 2.53315711, 1.67774642, 4.64619827, 0.000000e+00, 0.000000e+00], [-2.45925689, 2.87249136, 4.9106245, 3.85470629, 2.55194211, 2.78918052, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<6x8xf32>
    return %0 : tensor<6x8xf32>
  }
}

