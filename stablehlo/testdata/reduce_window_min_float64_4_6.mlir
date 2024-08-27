// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xf64>
    %1 = call @expected() : () -> tensor<3x5xf64>
    %cst = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<f64>
    %3 = "stablehlo.reduce_window"(%0, %2) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %4 = stablehlo.minimum %arg0, %arg1 : tensor<f64>
      stablehlo.return %4 : tensor<f64>
    }) : (tensor<4x6xf64>, tensor<f64>) -> tensor<3x5xf64>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<3x5xf64>, tensor<3x5xf64>) -> ()
    return %3 : tensor<3x5xf64>
  }
  func.func private @inputs() -> (tensor<4x6xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-5.340512701557711, 5.6640757346528865, -0.66274802564085955, 1.4868715820375751, -1.9026789543114404, 2.057134815565874], [-3.3202575744029352, 0.3349930575415731, 2.85193019745576, 2.846085033599338, -1.7546266468898608, -3.5823896338146968], [4.1471230365198188, 2.9182955017124028, -1.5072543490921961, 5.0012649170837218, -2.6999111295778397, -0.92528331781930073], [-0.70220192940453985, 1.6593889203822185, 5.7134344530849894, -0.063442846841271355, 1.9046710707445405, 1.1704310990119449]]> : tensor<4x6xf64>
    return %cst : tensor<4x6xf64>
  }
  func.func private @expected() -> (tensor<3x5xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-5.340512701557711, -0.66274802564085955, -0.66274802564085955, -1.9026789543114404, -3.5823896338146968], [-3.3202575744029352, -1.5072543490921961, -1.5072543490921961, -2.6999111295778397, -3.5823896338146968], [-0.70220192940453985, -1.5072543490921961, -1.5072543490921961, -2.6999111295778397, -2.6999111295778397]]> : tensor<3x5xf64>
    return %cst : tensor<3x5xf64>
  }
}
