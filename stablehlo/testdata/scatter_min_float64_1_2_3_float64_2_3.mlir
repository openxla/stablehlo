// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x2x3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x2x3xf64>, tensor<2x3xf64>)
    %1 = call @expected() : () -> tensor<1x2x3xf64>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<f64>
      stablehlo.return %3 : tensor<f64>
    }) : (tensor<1x2x3xf64>, tensor<1xi64>, tensor<2x3xf64>) -> tensor<1x2x3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x2x3xf64>, tensor<1x2x3xf64>) -> ()
    return %2 : tensor<1x2x3xf64>
  }
  func.func private @inputs() -> (tensor<1x2x3xf64> {mhlo.layout_mode = "default"}, tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[3.0225025851936644, -1.5950665390699252, -0.95293647350988098], [1.7268214008649281, -0.31037898543681103, 4.6791721730572959]]]> : tensor<1x2x3xf64>
    %cst_0 = stablehlo.constant dense<[[-0.27851742227736365, 3.3639292928448077, -7.0152414034484547], [-1.7853868218551712, 1.3020282295664241, 2.3475170431238057]]> : tensor<2x3xf64>
    return %cst, %cst_0 : tensor<1x2x3xf64>, tensor<2x3xf64>
  }
  func.func private @expected() -> (tensor<1x2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-0.27851742227736365, -1.5950665390699252, -7.0152414034484547], [-1.7853868218551712, -0.31037898543681103, 2.3475170431238057]]]> : tensor<1x2x3xf64>
    return %cst : tensor<1x2x3xf64>
  }
}
