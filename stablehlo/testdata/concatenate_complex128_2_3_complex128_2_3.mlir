// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x3xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>)
    %1 = call @expected() : () -> tensor<4x3xcomplex<f64>>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 0 : (tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> tensor<4x3xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x3xcomplex<f64>>, tensor<4x3xcomplex<f64>>) -> ()
    return %2 : tensor<4x3xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(0.12093412311353263,0.38010149284511113), (-1.9093750408113137,1.1698267132297944), (3.3517893977389743,-5.5234094816836858)], [(-1.058133517624317,-4.1777904231767859), (1.6058419047578061,-1.8533674523699413), (-4.0672779033135003,6.3303697566098354)]]> : tensor<2x3xcomplex<f64>>
    %cst_0 = stablehlo.constant dense<[[(-0.23924756010979267,1.6093707259284415), (6.5674842892777754,-2.0374352217249356), (-3.8670516047106598,-2.6247801011694483)], [(2.9389038494109894,-3.1402112199128149), (3.7405834733626224,-5.3309082802223058), (0.54943704590034437,2.2843780882730322)]]> : tensor<2x3xcomplex<f64>>
    return %cst, %cst_0 : tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<4x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(0.12093412311353263,0.38010149284511113), (-1.9093750408113137,1.1698267132297944), (3.3517893977389743,-5.5234094816836858)], [(-1.058133517624317,-4.1777904231767859), (1.6058419047578061,-1.8533674523699413), (-4.0672779033135003,6.3303697566098354)], [(-0.23924756010979267,1.6093707259284415), (6.5674842892777754,-2.0374352217249356), (-3.8670516047106598,-2.6247801011694483)], [(2.9389038494109894,-3.1402112199128149), (3.7405834733626224,-5.3309082802223058), (0.54943704590034437,2.2843780882730322)]]> : tensor<4x3xcomplex<f64>>
    return %cst : tensor<4x3xcomplex<f64>>
  }
}
