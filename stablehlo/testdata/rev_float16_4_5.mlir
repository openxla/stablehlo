// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x5xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x5xf16>
    %1 = call @expected() : () -> tensor<4x5xf16>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x5xf16>, tensor<4x5xf16>) -> ()
    return %2 : tensor<4x5xf16>
  }
  func.func private @inputs() -> (tensor<4x5xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-6.968750e+00, -1.627930e+00, -1.103520e+00, 3.359380e+00, 2.449950e-01], [-5.144530e+00, 9.155270e-01, -1.909180e+00, -1.003910e+00, -3.773440e+00], [-8.979490e-01, 8.730460e-01, 3.382810e+00, 2.332030e+00, -1.281250e+00], [-4.339840e+00, 1.500980e+00, 5.791020e-01, 4.203130e+00, 2.228520e+00]]> : tensor<4x5xf16>
    return %cst : tensor<4x5xf16>
  }
  func.func private @expected() -> (tensor<4x5xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.339840e+00, 1.500980e+00, 5.791020e-01, 4.203130e+00, 2.228520e+00], [-8.979490e-01, 8.730460e-01, 3.382810e+00, 2.332030e+00, -1.281250e+00], [-5.144530e+00, 9.155270e-01, -1.909180e+00, -1.003910e+00, -3.773440e+00], [-6.968750e+00, -1.627930e+00, -1.103520e+00, 3.359380e+00, 2.449950e-01]]> : tensor<4x5xf16>
    return %cst : tensor<4x5xf16>
  }
}
