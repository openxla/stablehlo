// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3x4xcomplex<f64>>
    %1 = call @expected() : () -> tensor<3x4xcomplex<f64>>
    %2 = stablehlo.real %0 : (tensor<3x4xcomplex<f64>>) -> tensor<3x4xf64>
    %3 = stablehlo.imag %0 : (tensor<3x4xcomplex<f64>>) -> tensor<3x4xf64>
    %4 = stablehlo.negate %3 : tensor<3x4xf64>
    %5 = stablehlo.complex %2, %4 : tensor<3x4xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%5, %1) {has_side_effect = true} : (tensor<3x4xcomplex<f64>>, tensor<3x4xcomplex<f64>>) -> ()
    return %5 : tensor<3x4xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<3x4xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-2.7757113803979814,2.5267782869731414), (1.992888780677561,0.71865309190141702), (-0.7875001393193215,-2.7237443061422169), (1.9000941560104727,-2.4956603839421039)], [(1.450950019069186,8.1688934962764747), (5.5157213985318672,-5.2448518436290303), (0.66197619089797932,3.3153452667243641), (-1.606452313252019,-1.4369754941605484)], [(-4.6290449066973105,3.8151155866122046), (6.4014787745251205,-1.0610978321015645), (-1.8782692857676464,-2.0500741375956322), (0.014332066692161645,-1.1266138718597651)]]> : tensor<3x4xcomplex<f64>>
    return %cst : tensor<3x4xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<3x4xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-2.7757113803979814,-2.5267782869731414), (1.992888780677561,-0.71865309190141702), (-0.7875001393193215,2.7237443061422169), (1.9000941560104727,2.4956603839421039)], [(1.450950019069186,-8.1688934962764747), (5.5157213985318672,5.2448518436290303), (0.66197619089797932,-3.3153452667243641), (-1.606452313252019,1.4369754941605484)], [(-4.6290449066973105,-3.8151155866122046), (6.4014787745251205,1.0610978321015645), (-1.8782692857676464,2.0500741375956322), (0.014332066692161645,1.1266138718597651)]]> : tensor<3x4xcomplex<f64>>
    return %cst : tensor<3x4xcomplex<f64>>
  }
}
