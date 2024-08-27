// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6x4xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3xf64>, tensor<f64>)
    %1 = call @expected() : () -> tensor<6x4xf64>
    %2 = stablehlo.pad %0#0, %0#1, low = [1, 0], high = [2, 1], interior = [1, 0] : (tensor<2x3xf64>, tensor<f64>) -> tensor<6x4xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<6x4xf64>, tensor<6x4xf64>) -> ()
    return %2 : tensor<6x4xf64>
  }
  func.func private @inputs() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}, tensor<f64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.0932493141393282E-4, -0.0016204126469074691, -4.7338837888897281E-4], [0.0021147984141223018, -1.9758004860187268E-4, -3.4312493334703604E-4]]> : tensor<2x3xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    return %cst, %cst_0 : tensor<2x3xf64>, tensor<f64>
  }
  func.func private @expected() -> (tensor<6x4xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-4.0932493141393282E-4, -0.0016204126469074691, -4.7338837888897281E-4, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.0021147984141223018, -1.9758004860187268E-4, -3.4312493334703604E-4, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<6x4xf64>
    return %cst : tensor<6x4xf64>
  }
}
