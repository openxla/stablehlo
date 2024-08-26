// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui8>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<4x6xf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui8>) -> tensor<4x3xf16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    return %4 : tensor<4x6xf16>
  }
  func.func private @inputs() -> (tensor<4x3xui8> {mhlo.layout_mode = "default"}, tensor<3x6xf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 0, 3], [1, 1, 0], [0, 1, 0], [0, 4, 0]]> : tensor<4x3xui8>
    %cst = stablehlo.constant dense<[[4.367190e+00, -6.535150e+00, 4.644530e+00, 8.007810e-01, 5.117190e+00, -3.431640e+00], [-1.705080e+00, -4.338380e-01, 3.429690e+00, -2.292480e-01, -1.335940e+00, 1.079100e+00], [5.113280e+00, -4.445310e+00, 2.050780e-01, -2.207030e+00, -1.338870e+00, 1.314450e+00]]> : tensor<3x6xf16>
    return %c, %cst : tensor<4x3xui8>, tensor<3x6xf16>
  }
  func.func private @expected() -> (tensor<4x6xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.407810e+01, -2.640630e+01, 9.906250e+00, -5.019530e+00, 6.218750e+00, -2.919920e+00], [2.662110e+00, -6.968750e+00, 8.078130e+00, 5.712890e-01, 3.781250e+00, -2.351560e+00], [-1.705080e+00, -4.338380e-01, 3.429690e+00, -2.292480e-01, -1.335940e+00, 1.079100e+00], [-6.820310e+00, -1.735350e+00, 1.371880e+01, -9.169920e-01, -5.343750e+00, 4.316410e+00]]> : tensor<4x6xf16>
    return %cst : tensor<4x6xf16>
  }
}
