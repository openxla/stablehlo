// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xf16>, tensor<3x6xf32>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xf16>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf32>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    return %4 : tensor<4x6xf32>
  }
  func.func private @inputs() -> (tensor<4x3xf16> {mhlo.layout_mode = "default"}, tensor<3x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[6.289060e-01, -1.862300e+00, 4.878910e+00], [-1.801760e+00, -4.636230e-01, 2.792970e-01], [1.379880e+00, 5.246090e+00, -9.033200e-01], [-7.660150e+00, -2.015630e+00, -7.167960e-01]]> : tensor<4x3xf16>
    %cst_0 = stablehlo.constant dense<[[1.03904819, -0.390662134, 0.586938202, 0.189655736, -3.34397602, -1.60065711], [-1.24223864, -4.05648661, 4.49510813, 0.90029788, -1.29326081, 3.30721259], [-1.39605582, 1.74161911, 1.90680158, -0.809361219, 1.66872275, 0.672101855]]> : tensor<3x6xf32>
    return %cst, %cst_0 : tensor<4x3xf16>, tensor<3x6xf32>
  }
  func.func private @expected() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-3.8443346, 15.8059206, 1.30097473, -5.50615072, 8.446940e+00, -3.88657904], [-1.68609679, 3.07098794, -2.60899258, -0.985164582, 7.09068965, 1.53841257], [-3.822050e+00, -23.3930168, 22.6692123, 5.71586227, -12.9062538, 14.5341043], [-4.4546957, 9.9205017, -14.9232807, -2.68730783, 27.0259743, 5.11342287]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
}
