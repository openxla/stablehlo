// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3x4xf16>
    %1 = call @expected() : () -> tensor<3x4xf16>
    stablehlo.custom_call @check.expect_close(%0, %1) {has_side_effect = true} : (tensor<3x4xf16>, tensor<3x4xf16>) -> ()
    return %0 : tensor<3x4xf16>
  }
  func.func private @inputs() -> (tensor<3x4xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[9.433590e-01, -1.404300e+00, -4.335940e-01, -4.843750e+00], [-2.107420e+00, -1.970700e+00, -7.065420e-01, -9.868160e-01], [-7.929680e+00, 2.261720e+00, 3.748050e+00, 1.697270e+00]]> : tensor<3x4xf16>
    return %cst : tensor<3x4xf16>
  }
  func.func private @expected() -> (tensor<3x4xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[9.433590e-01, -1.404300e+00, -4.335940e-01, -4.843750e+00], [-2.107420e+00, -1.970700e+00, -7.065420e-01, -9.868160e-01], [-7.929680e+00, 2.261720e+00, 3.748050e+00, 1.697270e+00]]> : tensor<3x4xf16>
    return %cst : tensor<3x4xf16>
  }
}
