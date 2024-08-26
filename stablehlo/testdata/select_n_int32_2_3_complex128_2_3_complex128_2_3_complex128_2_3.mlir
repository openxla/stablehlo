// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:4 = call @inputs() : () -> (tensor<2x3xi32>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>)
    %1 = call @expected() : () -> tensor<2x3xcomplex<f64>>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %3 = stablehlo.compare  LT, %0#0, %2,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %5 = stablehlo.compare  LT, %0#0, %4,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %6 = stablehlo.select %5, %0#2, %0#3 : tensor<2x3xi1>, tensor<2x3xcomplex<f64>>
    %7 = stablehlo.select %3, %0#1, %6 : tensor<2x3xi1>, tensor<2x3xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%7, %1) {has_side_effect = true} : (tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> ()
    return %7 : tensor<2x3xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<2x3xi32> {mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 0, 1], [2, 0, 2]]> : tensor<2x3xi32>
    %cst = stablehlo.constant dense<[[(-0.972899569715164,-0.19439260647629059), (1.4285494060489889,2.1200944122254861), (1.6992263098054345,5.0060033963738082)], [(-0.48134484304966618,3.0533784598258444), (3.2369772756671238,0.53969119388873188), (-2.2675026763114801,1.1807463285403068)]]> : tensor<2x3xcomplex<f64>>
    %cst_0 = stablehlo.constant dense<[[(0.33812031270166215,5.3939481387789412), (-1.6779673303524247,-3.6235118957269483), (2.1222088495553262,-1.2765837493591281)], [(3.2277343611739111,-0.57648162850785989), (6.0459979857659665,-1.3776613608161286), (2.324377849354657,-1.6803171534767465)]]> : tensor<2x3xcomplex<f64>>
    %cst_1 = stablehlo.constant dense<[[(-4.4332885042668408,-1.3333811809215232), (-2.7715994742635184,2.7833861511184836), (2.2669882206619563,1.9938611030949978)], [(-0.36411930119219904,5.1011209546411163), (4.6669742223605475,0.6042069681966763), (-0.76515931732154541,-0.054732358304975484)]]> : tensor<2x3xcomplex<f64>>
    return %c, %cst, %cst_0, %cst_1 : tensor<2x3xi32>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(0.33812031270166215,5.3939481387789412), (1.4285494060489889,2.1200944122254861), (2.1222088495553262,-1.2765837493591281)], [(-0.36411930119219904,5.1011209546411163), (3.2369772756671238,0.53969119388873188), (-0.76515931732154541,-0.054732358304975484)]]> : tensor<2x3xcomplex<f64>>
    return %cst : tensor<2x3xcomplex<f64>>
  }
}
