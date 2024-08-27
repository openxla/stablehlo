// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3xf64>, tensor<2xf64>)
    %1 = call @expected() : () -> tensor<4x2x3xf64>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<f64>
      stablehlo.return %3 : tensor<f64>
    }) : (tensor<4x2x3xf64>, tensor<2xi64>, tensor<2xf64>) -> tensor<4x2x3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xf64>, tensor<4x2x3xf64>) -> ()
    return %2 : tensor<4x2x3xf64>
  }
  func.func private @inputs() -> (tensor<4x2x3xf64> {mhlo.layout_mode = "default"}, tensor<2xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-0.70814289104667005, 7.6062126157397234, 0.18761567730500808], [-2.924071095806299, 0.86538630749576595, 0.67695919319953013]], [[0.84498750159959046, -4.8441990984275884, 1.6374197671206578], [1.9110378539069801, 0.11505512302261872, -6.4754125231782851]], [[-3.3276858487725334, 2.2768390412490298, 3.4226021604365693], [1.3775832163550565, -2.1804800331415279, -0.15284536973282603]], [[-4.5388283084287622, -3.3391713172917714, -0.013146948770895365], [-5.1108747593091044, 2.6479519397382552, -2.1067668637065649]]]> : tensor<4x2x3xf64>
    %cst_0 = stablehlo.constant dense<[1.42729657139038, -1.73800371155943]> : tensor<2xf64>
    return %cst, %cst_0 : tensor<4x2x3xf64>, tensor<2xf64>
  }
  func.func private @expected() -> (tensor<4x2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-0.70814289104667005, 7.6062126157397234, 0.18761567730500808], [-2.924071095806299, 0.86538630749576595, 0.67695919319953013]], [[0.84498750159959046, -4.8441990984275884, 1.6374197671206578], [1.9110378539069801, 0.11505512302261872, -6.4754125231782851]], [[-3.3276858487725334, 2.2768390412490298, 3.4226021604365693], [1.3775832163550565, -2.1804800331415279, -0.15284536973282603]], [[-4.5388283084287622, -3.3391713172917714, 1.42729657139038], [-5.1108747593091044, 2.6479519397382552, -1.73800371155943]]]> : tensor<4x2x3xf64>
    return %cst : tensor<4x2x3xf64>
  }
}
