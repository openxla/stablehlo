// RUN-DISABLED(inaccurate) stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui32>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<4x6xf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui32>) -> tensor<4x3xf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    stablehlo.custom_call @check.expect_almost_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    return %4 : tensor<4x6xf16>
  }
  func.func private @inputs() -> (tensor<4x3xui32> {mhlo.layout_mode = "default"}, tensor<3x6xf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4, 4, 0], [0, 3, 0], [1, 0, 0], [2, 3, 6]]> : tensor<4x3xui32>
    %cst = stablehlo.constant dense<[[-2.898440e+00, 1.852540e+00, -3.498050e+00, -1.629880e+00, -7.958980e-01, -4.046880e+00], [1.198240e+00, -2.576170e+00, -3.648440e+00, -1.153320e+00, -2.929690e+00, -5.292970e-01], [-7.308590e+00, 2.611330e+00, 3.431640e+00, 9.500000e+00, 4.707030e+00, 2.431640e+00]]> : tensor<3x6xf16>
    return %c, %cst : tensor<4x3xui32>, tensor<3x6xf16>
  }
  func.func private @expected() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-6.800780e+00, -2.894530e+00, -2.859380e+01, -1.113280e+01, -1.490630e+01, -1.831250e+01], [3.593750e+00, -7.726560e+00, -1.094530e+01, -3.460940e+00, -8.789060e+00, -1.587890e+00], [-2.898440e+00, 1.852540e+00, -3.498050e+00, -1.629880e+00, -7.958980e-01, -4.046880e+00], [-4.606250e+01, 1.164060e+01, 2.648440e+00, 5.028130e+01, 1.785940e+01, 4.906250e+00]]> : tensor<4x6xf16>
    return %cst : tensor<4x6xf16>
  }
}
