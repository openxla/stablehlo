// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<1x2x2xf32>, tensor<2x4x6xf32>)
    %1 = call @expected() : () -> tensor<2x4x6xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.pad %0#1, %cst, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "stablehlo.select_and_scatter"(%2, %0#0, %cst_0) <{window_dimensions = array<i64: 2, 2, 2>, window_strides = array<i64: 1, 2, 3>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<2x4x6xf32>, tensor<1x2x2xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %4 = stablehlo.slice %3 [0:2, 0:4, 0:6] : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> ()
    return %4 : tensor<2x4x6xf32>
  }
  func.func private @inputs() -> (tensor<1x2x2xf32> {mhlo.layout_mode = "default"}, tensor<2x4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-4.74842882, -3.46046281], [-5.42686605, 0.153474241]]]> : tensor<1x2x2xf32>
    %cst_0 = stablehlo.constant dense<[[[-2.15483785, -1.2998265, -2.06859159, 2.06262922, -5.04161739, -2.80120754], [-0.731763124, -0.849812805, 1.3785845, -3.86032128, 4.344930e+00, -0.56643635], [-4.96906567, -4.41274834, 0.0296598822, -0.221065283, 2.1941483, -3.0988338], [-6.20206594, -2.26923609, -3.92311668, -5.02652597, 0.513001502, -3.97359276]], [[-6.76744461, 0.523507416, 1.75738323, 1.93691206, -1.67728722, 0.715135514], [-4.50445032, 0.349508524, -2.15930676, 2.86179519, -1.65786111, -0.0169575941], [2.90006447, 1.82748401, -0.0447356291, 5.61337233, -1.01589382, 2.75839496], [-3.58221126, -3.38783431, 2.8053546, 5.55537176, 0.202345371, 2.61038017]]]> : tensor<2x4x6xf32>
    return %cst, %cst_0 : tensor<1x2x2xf32>, tensor<2x4x6xf32>
  }
  func.func private @expected() -> (tensor<2x4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -3.46046281, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, -4.74842882, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-5.42686605, 0.000000e+00, 0.000000e+00, 0.153474241, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>
    return %cst : tensor<2x4x6xf32>
  }
}
