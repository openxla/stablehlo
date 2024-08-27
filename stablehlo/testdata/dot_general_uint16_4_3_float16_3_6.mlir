// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui16>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<4x6xf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui16>) -> tensor<4x3xf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    return %4 : tensor<4x6xf16>
  }
  func.func private @inputs() -> (tensor<4x3xui16> {mhlo.layout_mode = "default"}, tensor<3x6xf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 0, 0], [5, 1, 3], [3, 3, 5], [1, 0, 6]]> : tensor<4x3xui16>
    %cst = stablehlo.constant dense<[[3.820310e+00, 2.255860e+00, 5.359380e+00, 6.479490e-01, -2.013670e+00, -2.236330e+00], [-9.077140e-01, -4.492190e-01, 2.250000e+00, 4.054690e+00, 3.130860e+00, 6.831050e-01], [-3.121090e+00, 3.070310e+00, -2.919920e+00, 1.774410e+00, -3.007810e-01, -1.563480e+00]]> : tensor<3x6xf16>
    return %c, %cst : tensor<4x3xui16>, tensor<3x6xf16>
  }
  func.func private @expected() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[7.640630e+00, 4.511720e+00, 1.071880e+01, 1.295900e+00, -4.027340e+00, -4.472660e+00], [8.828120e+00, 2.004690e+01, 2.028130e+01, 1.261720e+01, -7.839840e+00, -1.518750e+01], [-6.867180e+00, 2.076560e+01, 8.226560e+00, 2.298440e+01, 1.847660e+00, -1.247660e+01], [-1.490630e+01, 2.067190e+01, -1.215630e+01, 1.129690e+01, -3.818360e+00, -1.161720e+01]]> : tensor<4x6xf16>
    return %cst : tensor<4x6xf16>
  }
}
