// RUN-DISABLED(inaccurate) stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi32>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<4x6xf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi32>) -> tensor<4x3xf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    stablehlo.custom_call @check.expect_almost_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    return %4 : tensor<4x6xf16>
  }
  func.func private @inputs() -> (tensor<4x3xi32> {mhlo.layout_mode = "default"}, tensor<3x6xf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-3, 3, -1], [-3, 3, -2], [1, 1, 0], [-1, 1, -2]]> : tensor<4x3xi32>
    %cst = stablehlo.constant dense<[[-8.813470e-01, -3.654790e-01, 4.937500e+00, 9.712210e-03, -2.025390e+00, 2.474610e+00], [1.463870e+00, -8.090820e-01, -5.164060e+00, -6.535150e+00, -5.908200e-01, -1.177730e+00], [-1.011720e+00, 1.716800e+00, -7.593750e+00, -3.251950e+00, 4.164060e+00, 5.843750e+00]]> : tensor<3x6xf16>
    return %c, %cst : tensor<4x3xi32>, tensor<3x6xf16>
  }
  func.func private @expected() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[8.046880e+00, -3.046880e+00, -2.271880e+01, -1.637500e+01, 1.396480e-01, -1.679690e+01], [9.062500e+00, -4.765630e+00, -1.511720e+01, -1.313280e+01, -4.023440e+00, -2.264060e+01], [5.825200e-01, -1.174800e+00, -2.265630e-01, -6.527340e+00, -2.617190e+00, 1.296880e+00], [4.367190e+00, -3.876950e+00, 5.085940e+00, -4.095460e-02, -6.894530e+00, -1.534380e+01]]> : tensor<4x6xf16>
    return %cst : tensor<4x6xf16>
  }
}
