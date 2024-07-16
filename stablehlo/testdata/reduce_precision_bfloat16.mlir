// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<bf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<bf16>
    %1 = call @expected() : () -> tensor<bf16>
    %2 = stablehlo.reduce_precision %0, format = e11m52 : tensor<bf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<bf16>, tensor<bf16>) -> ()
    return %2 : tensor<bf16>
  }
  func.func private @inputs() -> (tensor<bf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<1.046880e+00> : tensor<bf16>
    return %cst : tensor<bf16>
  }
  func.func private @expected() -> (tensor<bf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<1.046880e+00> : tensor<bf16>
    return %cst : tensor<bf16>
  }
}
