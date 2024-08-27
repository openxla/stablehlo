// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui16>, tensor<3x6xf64>)
    %1 = call @expected() : () -> tensor<4x6xf64>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui16>) -> tensor<4x3xf64>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf64>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    return %4 : tensor<4x6xf64>
  }
  func.func private @inputs() -> (tensor<4x3xui16> {mhlo.layout_mode = "default"}, tensor<3x6xf64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4, 5, 1], [1, 1, 1], [1, 0, 1], [4, 3, 1]]> : tensor<4x3xui16>
    %cst = stablehlo.constant dense<[[-7.3898443007284609, 0.19670499395453581, -1.7267214064489096, -5.5554539393575659, -4.3368963353758403, 1.4177000540488949], [-4.2871662468944489, 1.4031557079413877, -2.2210025607289889, 0.013673718816889249, 1.81969453715385, 1.804112442445156], [1.4987380528153229, 3.4588654567804267, 2.6412790852210235, -0.64853344011468306, 3.0950655806641274, -0.028863570797704914]]> : tensor<3x6xf64>
    return %c, %cst : tensor<4x3xui16>, tensor<3x6xf64>
  }
  func.func private @expected() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-49.49647038457077, 11.261463972305508, -15.370619344219561, -22.8019806034605, -5.1540470750699834, 14.662498857623655], [-10.178272494807587, 5.0587261586763503, -1.306444881956875, -6.1903136606553595, 0.5778637824421371, 3.1929489256963461], [-5.8911062479131378, 3.6555704507349627, 0.91455767877211391, -6.2039873794722489, -1.2418307547117129, 1.3888364832511899], [-40.922137890781869, 8.4551525564227337, -10.928614222761581, -22.829328041094278, -8.7934361493776834, 11.054273972733343]]> : tensor<4x6xf64>
    return %cst : tensor<4x6xf64>
  }
}
